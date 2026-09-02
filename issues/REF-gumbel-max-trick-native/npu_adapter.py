#!/usr/bin/env python3
"""Gumbel-max community contract 的 NPU 最小适配器。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch_npu
from torch._dynamo.utils import counters
from torch._inductor.utils import run_and_get_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    if not torch.npu.is_available():
        raise RuntimeError("当前进程不可见 NPU")
    torch.npu.set_device("npu:0")
    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"
    from torch_npu.utils._dynamo import register_inductor_npu

    register_inductor_npu()
    counters.clear()

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
    with torch._inductor.config.patch(apply_gumbel_max_trick=True):
        compiled = torch.compile(
            sample,
            options={"npu_backend": "triton_experimental"},
        )
        output, codes = run_and_get_code(compiled, logits, temperature)

    distribution = (torch.bincount(output.flatten()) / samples).cpu().tolist()
    denominator = categories * (categories + 1) / 2
    expected = [index / denominator for index in range(1, categories + 1)]
    ratios = [actual / target for target, actual in zip(expected, distribution)]
    if not all(abs(ratio - 1) < 0.1 for ratio in ratios):
        raise AssertionError(f"Gumbel-max 分布超出 10% 相对容差: {ratios}")
    target_count = counters["inductor"]["apply_gumbel_max_trick"]
    if target_count != 1:
        raise AssertionError(f"apply_gumbel_max_trick 计数应为 1，实际为 {target_count}")

    code_path = args.artifact_dir / "generated_code.py"
    code_path.write_text("\n\n".join(codes), encoding="utf-8")
    summary = {
        "source_test": "test/inductor/test_pattern_matcher.py::TestPatternMatcherLogging.test_gumbel_max_trick",
        "adapter_deviation": [
            "GPU_TYPE=npu",
            "backend=triton_experimental",
            "artifact capture",
        ],
        "product_gate_bypassed": False,
        "tests_run": 1,
        "successful": True,
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "observation": {
            "shape": list(logits.shape),
            "dtype": str(logits.dtype),
            "target_count": target_count,
            "expected_distribution": expected,
            "actual_distribution": distribution,
            "actual_over_expected": ratios,
            "relative_tolerance": 0.1,
            "correctness": "passed",
            "generated_code": str(code_path),
        },
        "selected_environment": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "SET_NPU_DEVICE",
                "TORCH_COMPILE_DEBUG",
                "TORCH_COMPILE_DEBUG_DIR",
                "TORCH_TRACE",
                "TORCHINDUCTOR_CACHE_DIR",
            )
        },
    }
    (args.artifact_dir / "adapter_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("T077_NPU_RESULT=" + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
