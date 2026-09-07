#!/usr/bin/env python3
"""T-080 单臂 NPU 性能 worker；只测未被产品显式关闭的单元。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"
WORK = Path("/home/z50063656/tmp")
if Path.cwd().resolve() != WORK:
    raise RuntimeError("必须从 /home/z50063656/tmp 启动")

import torch
import torch_npu
from torch._inductor import config, metrics
from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu


PYTORCH_COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
UNITS = {
    "scatter": "AU-joint-graph-scatter-upon-const-tensor",
    "constructors": "AU-post-grad-move-constructors-to-gpu",
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


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def install_constructor_control(mode: str, state: dict[str, int]) -> None:
    from torch._inductor.fx_passes import post_grad

    original = post_grad.move_constructors_to_gpu

    def controlled(graph):
        state["calls"] = state.get("calls", 0) + 1
        if mode == "on":
            return original(graph)
        return None

    post_grad.move_constructors_to_gpu = controlled


def time_callable(fn, warmup: int, runs: int) -> tuple[dict, dict]:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    event_ms = []
    for _ in range(runs):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        event_ms.append(start.elapsed_time(end))
    host_ms = []
    for _ in range(runs):
        before = time.perf_counter()
        fn()
        torch.npu.synchronize()
        host_ms.append((time.perf_counter() - before) * 1000)
    return summarize(event_ms), summarize(host_ms)


def scatter_workload(mode: str, warmup: int, runs: int) -> list[dict]:
    # 原社区 DO_PERF_TEST=1 的完整合同，不能静默缩小后仍称 community。
    B, T, D, V = 32, 1024, 768, 50257
    ref_model = torch.nn.Linear(D, V).to(torch.bfloat16).to("npu")
    opt_model = copy.deepcopy(ref_model)
    ce = torch.nn.CrossEntropyLoss()
    x = torch.randn(B, T, D, dtype=torch.bfloat16, device="npu")
    label = torch.randint(0, V, (B, T), dtype=torch.int64, device="npu")

    def fn(model, input_tensor, target):
        ce(model(input_tensor).view(-1, V), target.view(-1)).backward()

    ref_model.zero_grad(set_to_none=True)
    fn(ref_model, x, label)
    ref_grad = ref_model.weight.grad.clone()
    metrics.reset()
    with config.patch(optimize_scatter_upon_const_tensor=mode == "on"):
        started = time.perf_counter()
        compiled = torch.compile(fn, options={"npu_backend": "triton_experimental"})
        compiled(opt_model, x, label)
        torch.npu.synchronize()
        compile_ms = (time.perf_counter() - started) * 1000
        torch.testing.assert_close(
            opt_model.weight.grad, ref_grad, atol=1e-3, rtol=1e-3
        )
        target_hits = metrics.num_matches_for_scatter_upon_const_tensor
        if mode == "on" and target_hits < 1:
            raise RuntimeError("scatter ON 未命中 target metric")
        if mode == "off" and target_hits != 0:
            raise RuntimeError("scatter OFF 仍命中 target metric")
        torch.npu.reset_peak_memory_stats()
        event, host = time_callable(
            lambda: compiled(opt_model, x, label), warmup, runs
        )
    return [{
        "workload_id": "scatter-cross-entropy-community-full",
        "shape": {"B": B, "T": T, "D": D, "V": V},
        "community_primary": True,
        "correctness": "passed",
        "compile_ms": compile_ms,
        "target_hits": target_hits,
        "event": event,
        "host": host,
        "max_memory_allocated": torch.npu.max_memory_allocated(),
        "max_memory_reserved": torch.npu.max_memory_reserved(),
    }]


def constructor_workloads(
    mode: str, warmup: int, runs: int, state: dict[str, int]
) -> list[dict]:
    del mode

    def fn(x):
        return torch.arange(len(x), device="cpu").to(x.device) + x

    output = []
    for length in (32, 1024, 65536, 1048576):
        torch._dynamo.reset()
        x = torch.randn(length, device="npu")
        expected = fn(x)
        calls_before = state.get("calls", 0)
        started = time.perf_counter()
        compiled = torch.compile(fn, options={"npu_backend": "triton_experimental"})
        actual = compiled(x)
        torch.npu.synchronize()
        compile_ms = (time.perf_counter() - started) * 1000
        torch.testing.assert_close(actual, expected)
        calls = state.get("calls", 0) - calls_before
        if calls < 1:
            raise RuntimeError("constructor target pass 未进入")
        torch.npu.reset_peak_memory_stats()
        event, host = time_callable(lambda: compiled(x), warmup, runs)
        output.append({
            "workload_id": (
                "move-arange-community-shape"
                if length == 32
                else f"move-arange-length-{length}-sensitivity"
            ),
            "shape": [length],
            "community_primary": length == 32,
            "correctness": "passed",
            "compile_ms": compile_ms,
            "target_pass_calls": calls,
            "event": event,
            "host": host,
            "max_memory_allocated": torch.npu.max_memory_allocated(),
            "max_memory_reserved": torch.npu.max_memory_reserved(),
        })
    return output


def count_codegen(debug_root: Path) -> dict:
    files = sorted(debug_root.rglob("output_code.py"))
    joined = "\n".join(path.read_text(errors="replace") for path in files)
    return {
        "output_code_files": len(files),
        "triton_kernels": joined.count("@triton.jit"),
        "device_put_count": joined.count("prims.device_put"),
        "empty_strided_npu_scalar_count": joined.count("empty_strided_npu(())"),
        "sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in files],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", choices=sorted(UNITS), required=True)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    register_inductor_npu()
    if _InductorNpuRegistry._loaded_backend != "triton_experimental":
        raise RuntimeError("实际 backend 不是 triton_experimental")
    if torch.version.git_version != PYTORCH_COMMIT:
        raise RuntimeError("PyTorch commit 不匹配")
    torch.npu.set_device(0)
    state: dict[str, int] = {}
    if args.unit == "constructors":
        install_constructor_control(args.mode, state)
        workloads = constructor_workloads(
            args.mode, args.warmup, args.runs, state
        )
    else:
        workloads = scatter_workload(args.mode, args.warmup, args.runs)

    debug_root = Path(os.environ["TORCH_COMPILE_DEBUG_DIR"])
    result = {
        "schema_version": "1.0",
        "task_id": "T-080",
        "acceptance_unit_id": UNITS[args.unit],
        "unit": args.unit,
        "mode": args.mode,
        "round": args.round,
        "backend": _InductorNpuRegistry._loaded_backend,
        "torch_commit": torch.version.git_version,
        "torch_npu_commit": git_head(Path("/home/z50063656/Pass/src/torch_npu")),
        "npu_triton_heuristics_file": __import__(
            "torch_npu._inductor.triton_experimental.npu_triton_heuristics",
            fromlist=["__file__"],
        ).__file__,
        "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "product_gate_bypassed": False,
        "warmup": args.warmup,
        "runs": args.runs,
        "workloads": workloads,
        "codegen": count_codegen(debug_root),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("T080_PERFORMANCE=" + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
