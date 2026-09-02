#!/usr/bin/env python3
"""dynamic decompose-addmm community contract 的 NPU 产品基线适配器。"""

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


def addmm_fn(bias, left, right):
    return torch.ops.aten.addmm.default(bias, left, right)


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
    m, k, n = 19_494_144, 8, 8
    left = torch.randn(m, k, device="npu")
    right = torch.randn(k, n, device="npu")
    bias = torch.randn(n, device="npu")
    counters.clear()
    options = {"decompose_mm_pass": {"skip_dynamic_shape_dim_check": True}}
    with torch._inductor.config.patch(post_grad_fusion_options=options):
        expected = addmm_fn(bias, left, right)
        compiled = torch.compile(
            addmm_fn,
            dynamic=True,
            options={"npu_backend": "triton_experimental"},
        )
        actual, codes = run_and_get_code(compiled, bias, left, right)
    torch.testing.assert_close(actual, expected)
    target_count = counters["inductor"]["decompose_addmm"]
    if target_count != 0:
        raise AssertionError(
            f"dynamic addmm 在 NPU device guard 下不应分解，实际计数 {target_count}"
        )

    code_path = args.artifact_dir / "generated_code.py"
    code_path.write_text("\n\n".join(codes), encoding="utf-8")
    summary = {
        "source_test": "test/inductor/test_decompose_mem_bound_mm.py::TestDecomposeMemMM.test_dynamic_shape_decompose_addmm",
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
            "input_shapes": [list(bias.shape), list(left.shape), list(right.shape)],
            "input_strides": [list(bias.stride()), list(left.stride()), list(right.stride())],
            "dtype": str(left.dtype),
            "dynamic": True,
            "skip_dynamic_shape_dim_check": True,
            "reference_expected_count": 1,
            "npu_target_count": target_count,
            "correctness": "passed",
            "generated_code": str(code_path),
        },
        "guard_observation": "should_decompose_mm 的 dynamic 分支仍只接受 cuda/xpu 或 cpu，NPU 保持 addmm。",
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
