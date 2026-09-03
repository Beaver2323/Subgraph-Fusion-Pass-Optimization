#!/usr/bin/env python3
"""T-077 Gumbel-max fresh-process OFF/ON 性能 worker。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

# 后端注册具有进程级生命周期；必须在导入 torch/torch_npu 前固定。
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"

import torch
import torch_npu
from torch._dynamo.utils import counters
from torch._inductor.utils import run_and_get_code

COMMUNITY_SOURCE_NODEID = (
    "test/inductor/test_pattern_matcher.py::"
    "TestPatternMatcherLogging.test_gumbel_max_trick"
)


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


def main() -> int:
    parser = argparse.ArgumentParser()
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
    counters.clear()
    torch.manual_seed(20260903)

    def sample(logits, temperature):
        logits = logits / max(temperature, 1e-5)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        q = torch.empty_like(logits).exponential_(1)
        return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

    categories = 10
    samples = 1_000_000
    temperature = 0.8
    row = (
        torch.arange(1, categories + 1, dtype=torch.float, device="npu").log()
        * temperature
    )
    logits = row[None, :].repeat(samples, 1)
    enabled = args.mode == "on"

    torch.npu.reset_peak_memory_stats()
    compile_started = time.perf_counter_ns()
    with torch._inductor.config.patch(apply_gumbel_max_trick=enabled):
        compiled = torch.compile(
            sample,
            options={"npu_backend": "triton_experimental"},
        )
        output, codes = run_and_get_code(compiled, logits, temperature)
        torch.npu.synchronize()
        compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000

        for _ in range(args.warmup):
            output = compiled(logits, temperature)
        torch.npu.synchronize()

        host_samples: list[float] = []
        device_samples: list[float] = []
        for _ in range(args.runs):
            start_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)
            host_started = time.perf_counter_ns()
            start_event.record()
            output = compiled(logits, temperature)
            end_event.record()
            torch.npu.synchronize()
            host_samples.append(
                (time.perf_counter_ns() - host_started) / 1_000_000
            )
            device_samples.append(start_event.elapsed_time(end_event))

    distribution = (torch.bincount(output.flatten()) / samples).cpu().tolist()
    denominator = categories * (categories + 1) / 2
    expected = [index / denominator for index in range(1, categories + 1)]
    ratios = [actual / target for target, actual in zip(expected, distribution)]
    correctness = all(abs(ratio - 1) < 0.1 for ratio in ratios)
    target_count = counters["inductor"]["apply_gumbel_max_trick"]
    expected_count = 1 if enabled else 0
    if not correctness:
        raise AssertionError(f"Gumbel-max 分布超出 10% 相对容差: {ratios}")
    if target_count != expected_count:
        raise AssertionError(
            f"{args.mode} 的目标 counter 应为 {expected_count}，实际为 {target_count}"
        )

    code_text = "\n\n".join(codes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    code_path = args.output.with_name("generated_code.py")
    code_path.write_text(code_text, encoding="utf-8")
    result = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_id": "T-077",
        "acceptance_unit_id": "AU-apply-gumbel-max-trick",
        "performance_case_source": {
            "kind": "minimal-wrapper-around-community-functional-test",
            "nodeid": COMMUNITY_SOURCE_NODEID,
            "reused": [
                "sample 计算图",
                "float32[1000000,10]",
                "temperature=0.8",
                "十分类理论分布",
                "10% 相对容差",
            ],
            "added": [
                "pass OFF/ON",
                "fresh-process 隔离",
                "host/NPU Event 计时",
                "峰值显存",
                "generated code",
            ],
        },
        "mode": args.mode,
        "round": args.round,
        "warmup": args.warmup,
        "runs": args.runs,
        "input": {
            "shape": list(logits.shape),
            "dtype": str(logits.dtype),
            "temperature": temperature,
        },
        "correctness": {
            "status": "passed",
            "relative_tolerance": 0.1,
            "expected_distribution": expected,
            "actual_distribution": distribution,
            "actual_over_expected": ratios,
        },
        "pattern": {
            "expected_count": expected_count,
            "actual_count": target_count,
            "generated_code": str(code_path),
            "triton_marker_count": code_text.count("triton_"),
            "extern_kernel_marker_count": code_text.count("extern_kernels."),
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
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "cann": "9.0.1",
            "npu": torch.npu.get_device_name(0),
            "python": sys.version,
            "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "cwd": str(Path.cwd()),
        },
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
