#!/usr/bin/env python3
"""decompose-bmm 三个 community contracts 的 NPU 产品基线适配器。"""

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


def bmm_fn(left, right):
    return torch.bmm(left, right)


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
    cases = [
        ("large-batch-small-mnk-positive", 10240, 2, 2, 2, True),
        ("other-dims-threshold-negative", 10240, 2, 32, 32, False),
        ("batch-threshold-negative", 2000, 2, 2, 2, False),
    ]
    observations = []
    options = {"decompose_mm_pass": {}}
    with torch._inductor.config.patch(post_grad_fusion_options=options):
        for variant_id, batch, m, k, n, reference_expected_match in cases:
            torch._dynamo.reset()
            counters.clear()
            left = torch.randn(batch, m, k, device="npu")
            right = torch.randn(batch, k, n, device="npu")
            ref_left = left.detach().clone().requires_grad_(True)
            ref_right = right.detach().clone().requires_grad_(True)
            run_left = left.detach().clone().requires_grad_(True)
            run_right = right.detach().clone().requires_grad_(True)

            expected = bmm_fn(ref_left, ref_right)
            compiled = torch.compile(
                bmm_fn,
                options={"npu_backend": "triton_experimental"},
            )
            actual, codes = run_and_get_code(compiled, run_left, run_right)
            torch.testing.assert_close(actual, expected)
            forward_count = counters["inductor"]["decompose_bmm"]
            expected.sum().backward()
            actual.sum().backward()
            torch.testing.assert_close(run_left.grad, ref_left.grad)
            torch.testing.assert_close(run_right.grad, ref_right.grad)
            total_count = counters["inductor"]["decompose_bmm"]
            if forward_count != 0 or total_count != 0:
                raise AssertionError(
                    f"{variant_id} 在 NPU device guard 下不应分解: "
                    f"forward={forward_count}, total={total_count}"
                )

            code_path = args.artifact_dir / f"{variant_id}-generated_code.py"
            code_path.write_text("\n\n".join(codes), encoding="utf-8")
            observations.append(
                {
                    "variant_id": variant_id,
                    "input_shapes": [list(left.shape), list(right.shape)],
                    "input_strides": [list(left.stride()), list(right.stride())],
                    "dtype": str(left.dtype),
                    "reference_expected_match": reference_expected_match,
                    "npu_forward_count": forward_count,
                    "npu_total_count": total_count,
                    "forward_correctness": "passed",
                    "gradient_correctness": "passed",
                    "generated_code": str(code_path),
                }
            )

    summary = {
        "source_test": "test/inductor/test_decompose_mem_bound_mm.py::TestDecomposeMemMM.test_decompose_bmm",
        "adapter_deviation": [
            "GPU_TYPE=npu",
            "backend=triton_experimental",
            "artifact capture",
        ],
        "product_gate_bypassed": False,
        "tests_run": len(observations),
        "successful": True,
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "guard_observation": "should_decompose_bmm 只接受 cuda/xpu 正例或 cpu 特例，NPU 三例均保持 bmm。",
        "observations": observations,
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
