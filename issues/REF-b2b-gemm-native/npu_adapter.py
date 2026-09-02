#!/usr/bin/env python3
"""B2B GEMM 六个 community contracts 的 NPU 产品基线适配器。"""

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


def left_gelu(a, b, c):
    return torch.mm(torch.nn.functional.gelu(torch.mm(a, b)), c)


def right_relu(a, b, c):
    return torch.mm(a, torch.nn.functional.relu(torch.mm(b, c)))


def trivial_left(a, b, c):
    return torch.mm(torch.mm(a, b), c)


def trivial_right(a, b, c):
    return torch.mm(a, torch.mm(b, c))


def bad_pattern(a, b, c):
    first = torch.mm(a, b)
    second = torch.mm(first, c)
    return torch.mm(first, second)


def make_inputs(left: bool, size: int = 256):
    if left:
        shapes = ((size, 32), (32, size), (size, 32))
    else:
        shapes = ((32, size), (size, 32), (32, size))
    return tuple(
        torch.randn(shape, device="npu", dtype=torch.float16) for shape in shapes
    )


def fp32_reference(fn, inputs):
    promoted = tuple(value.float() for value in inputs)
    return fn(*promoted).to(torch.float16)


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
        ("left-gelu-positive", left_gelu, make_inputs(True), True),
        ("right-relu-positive", right_relu, make_inputs(False), True),
        ("trivial-left-positive", trivial_left, make_inputs(True), True),
        ("trivial-right-positive", trivial_right, make_inputs(False), True),
        ("bad-pattern-negative", bad_pattern, make_inputs(True), False),
        (
            "bad-shape-negative",
            trivial_left,
            tuple(
                torch.randn((100, 100), device="npu", dtype=torch.float16)
                for _ in range(3)
            ),
            False,
        ),
    ]
    observations = []
    with torch._inductor.config.patch(b2b_gemm_pass=True):
        for variant_id, fn, inputs, use_fp32_reference in cases:
            torch._dynamo.reset()
            counters.clear()
            expected = fp32_reference(fn, inputs) if use_fp32_reference else fn(*inputs)
            compiled = torch.compile(
                fn,
                options={"npu_backend": "triton_experimental"},
            )
            actual, codes = run_and_get_code(compiled, *inputs)
            torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.01)
            target_count = counters["inductor"]["b2b_gemm"]
            if target_count != 0:
                raise AssertionError(
                    f"{variant_id} 在 NPU device guard 下不应命中，实际计数 {target_count}"
                )
            code_text = "\n\n".join(codes)
            markers = {
                "left": "B2B_GEMM_LEFT_TRITON_ENTRANCE" in code_text,
                "right": "B2B_GEMM_RIGHT_TRITON_ENTRANCE" in code_text,
            }
            if any(markers.values()):
                raise AssertionError(f"{variant_id} 出现非预期 B2B marker: {markers}")
            code_path = args.artifact_dir / f"{variant_id}-generated_code.py"
            code_path.write_text(code_text, encoding="utf-8")
            observations.append(
                {
                    "variant_id": variant_id,
                    "input_shapes": [list(value.shape) for value in inputs],
                    "input_strides": [list(value.stride()) for value in inputs],
                    "dtype": "float16",
                    "reference_expected_match": use_fp32_reference,
                    "npu_target_count": target_count,
                    "markers": markers,
                    "correctness": "passed",
                    "generated_code": str(code_path),
                }
            )

    summary = {
        "source_tests": [
            "test/inductor/test_b2b_gemm.py::B2BGEMMTest.test_b2b_gemm_left_assoc_good_shape",
            "test/inductor/test_b2b_gemm.py::B2BGEMMTest.test_b2b_gemm_right_assoc_good_shape",
            "test/inductor/test_b2b_gemm.py::B2BGEMMTest.test_b2b_gemm_trivial_left_assoc_good_shape",
            "test/inductor/test_b2b_gemm.py::B2BGEMMTest.test_b2b_gemm_trivial_right_assoc_good_shape",
            "test/inductor/test_b2b_gemm.py::B2BGEMMTest.test_b2b_gemm_bad_pattern_good_shape",
            "test/inductor/test_b2b_gemm.py::B2BGEMMTest.test_b2b_gemm_good_pattern_bad_shape",
        ],
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
        "guard_observation": "is_b2b_gemm_good_on 只接受 is_cuda 或 is_xpu，NPU 六例均保持原图。",
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
