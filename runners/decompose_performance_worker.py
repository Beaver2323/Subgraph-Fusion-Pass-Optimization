#!/usr/bin/env python3
"""T-077 decompose-mm/bmm/addmm 的 fresh-process NPU 性能 worker。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

# 后端注册有进程级生命周期，必须先于 torch/torch_npu 导入。
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"

import torch
import torch_npu
from torch._dynamo.utils import counters
from torch._inductor.utils import run_and_get_code


COMMUNITY_CASES = {
    "bmm": {
        "acceptance_unit_id": "AU-decompose-mem-bound-mm-decompose-bmm",
        "nodeid": (
            "test/inductor/test_decompose_mem_bound_mm.py::"
            "TestDecomposeMemMM.test_decompose_bmm"
        ),
        "shape_contract": "B=10240,M=K=N=2 positive",
        "counter": "decompose_bmm",
    },
    "mm": {
        "acceptance_unit_id": "AU-decompose-mem-bound-mm-decompose-mm",
        "nodeid": (
            "test/inductor/test_decompose_mem_bound_mm.py::"
            "TestDecomposeMemMM.test_decompose_mm"
        ),
        "shape_contract": "M=20480,K=5,N=2 fp32 positive",
        "counter": "decompose_mm",
    },
    "addmm": {
        "acceptance_unit_id": "AU-decompose-mem-bound-mm-decompose-addmm",
        "nodeid": (
            "test/inductor/test_decompose_mem_bound_mm.py::"
            "TestDecomposeMemMM.test_dynamic_shape_decompose_addmm"
        ),
        "shape_contract": "M=19494144,K=N=8 dynamic positive",
        "counter": "decompose_addmm",
    },
}


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50_ms": percentile(values, 0.50),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def install_npu_capability_adapter() -> None:
    """仅为 capability/perf 探针让 NPU 复用 upstream CUDA shape policy。"""
    from torch._inductor.fx_passes import decompose_mem_bound_mm

    original = decompose_mem_bound_mm.check_device

    def check_device_with_npu(a, b, device="cuda"):
        if (
            device == "cuda"
            and a.device.type == "npu"
            and b.device.type == "npu"
        ):
            return True
        return original(a, b, device=device)

    decompose_mem_bound_mm.check_device = check_device_with_npu


def install_small_mm_correctness_guard() -> None:
    """复用已验证 dfbcc25 的测试态边界，隔离已知 NPU lowering 回归。"""
    from torch._inductor.kernel import mm as upstream_mm_kernel

    original = upstream_mm_kernel._use_small_mm_pointwise

    def npu_safe_small_mm_guard(m, k, n, layout):
        if layout.device.type == "npu":
            return False
        return original(m, k, n, layout)

    upstream_mm_kernel._use_small_mm_pointwise = npu_safe_small_mm_guard


def make_case(unit: str):
    if unit == "bmm":
        def fn(left, right):
            return torch.bmm(left, right)

        inputs = (
            torch.randn(10240, 2, 2, device="npu", dtype=torch.float32),
            torch.randn(10240, 2, 2, device="npu", dtype=torch.float32),
        )
        return fn, inputs, False
    if unit == "mm":
        def fn(left, right):
            return torch.mm(left, right)

        inputs = (
            torch.randn(20480, 5, device="npu", dtype=torch.float32),
            torch.randn(5, 2, device="npu", dtype=torch.float32),
        )
        return fn, inputs, False

    def fn(bias, left, right):
        return torch.ops.aten.addmm.default(bias, left, right)

    inputs = (
        torch.randn(8, device="npu", dtype=torch.float32),
        torch.randn(19_494_144, 8, device="npu", dtype=torch.float32),
        torch.randn(8, 8, device="npu", dtype=torch.float32),
    )
    return fn, inputs, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", choices=tuple(COMMUNITY_CASES), required=True)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    if Path.cwd().resolve() != Path("/home/z50063656/tmp"):
        raise RuntimeError("NPU 性能测试必须从 /home/z50063656/tmp 启动")
    if not torch.npu.is_available():
        raise RuntimeError("当前进程不可见 NPU")
    torch.npu.set_device("npu:0")
    from torch_npu.utils._dynamo import register_inductor_npu

    register_inductor_npu()
    install_small_mm_correctness_guard()
    if args.mode == "on":
        install_npu_capability_adapter()

    torch.manual_seed(20260903)
    fn, inputs, dynamic = make_case(args.unit)
    expected = fn(*inputs)
    counters.clear()
    enabled = args.mode == "on"
    pass_options = (
        {
            "decompose_mm_pass": {
                "skip_dynamic_shape_dim_check": args.unit == "addmm",
            }
        }
        if enabled
        else {}
    )

    torch.npu.reset_peak_memory_stats()
    compile_started = time.perf_counter_ns()
    with torch._inductor.config.patch(post_grad_fusion_options=pass_options):
        compiled = torch.compile(
            fn,
            dynamic=dynamic,
            options={"npu_backend": "triton_experimental"},
        )
        actual, codes = run_and_get_code(compiled, *inputs)
        torch.npu.synchronize()
        compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

        for _ in range(args.warmup):
            actual = compiled(*inputs)
        torch.npu.synchronize()

        host_samples: list[float] = []
        device_samples: list[float] = []
        for _ in range(args.runs):
            start_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)
            host_started = time.perf_counter_ns()
            start_event.record()
            actual = compiled(*inputs)
            end_event.record()
            torch.npu.synchronize()
            host_samples.append(
                (time.perf_counter_ns() - host_started) / 1_000_000
            )
            device_samples.append(start_event.elapsed_time(end_event))

    counter_name = COMMUNITY_CASES[args.unit]["counter"]
    target_count = counters["inductor"][counter_name]
    expected_count = 1 if enabled else 0
    if target_count != expected_count:
        raise AssertionError(
            f"{args.unit} {args.mode} 的 {counter_name} 应为 {expected_count}，"
            f"实际为 {target_count}"
        )
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    code_text = "\n\n".join(codes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    code_path = args.output.with_name("generated_code.py")
    code_path.write_text(code_text, encoding="utf-8")
    source = COMMUNITY_CASES[args.unit]
    result = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_id": "T-077",
        "acceptance_unit_id": source["acceptance_unit_id"],
        "performance_case_source": {
            "kind": "derived-from-community-functional-positive",
            "nodeid": source["nodeid"],
            "reused": [source["shape_contract"], "float32 dtype", "1e-3 atol/rtol"],
            "added": [
                "pass OFF/ON fresh-process isolation",
                "host/NPU Event timing",
                "peak memory and generated code",
            ],
            "community_performance_test": "absent",
        },
        "mode": args.mode,
        "round": args.round,
        "warmup": args.warmup,
        "runs": args.runs,
        "input_shapes": [list(value.shape) for value in inputs],
        "dynamic": dynamic,
        "correctness": "passed",
        "pattern": {
            "counter": counter_name,
            "expected_count": expected_count,
            "actual_count": target_count,
            "generated_code": str(code_path),
        },
        "candidate_scope": {
            "npu_device_adapter": enabled,
            "cuda_shape_policy_reused": enabled,
            "small_mm_correctness_guard": "dfbcc25-equivalent-test-patch",
            "product_source_modified": False,
        },
        "timing": {
            "compile_and_first_run_ms": compile_ms,
            "host": summarize(host_samples),
            "device_event": summarize(device_samples),
        },
        "memory": {
            "max_allocated_bytes": torch.npu.max_memory_allocated(),
            "max_reserved_bytes": torch.npu.max_memory_reserved(),
        },
        "environment": {
            "backend": "triton_experimental",
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(0),
            "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "python": sys.version,
            "cwd": str(Path.cwd()),
        },
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("T077_DECOMPOSE_PERFORMANCE=" + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
