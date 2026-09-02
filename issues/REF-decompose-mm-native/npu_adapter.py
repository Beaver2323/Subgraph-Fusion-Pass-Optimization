#!/usr/bin/env python3
"""decompose-mm fp32/mixed contracts 的 NPU 产品基线适配器。"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

import torch
import torch_npu
from torch._dynamo.utils import counters
from torch._inductor.utils import run_and_get_code


def mm_fn(left, right):
    return torch.mm(left, right)


def autocast_context(mixed: bool):
    if mixed:
        return torch.autocast(device_type="npu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    absolute = (actual - expected).abs()
    denominator = expected.abs().clamp_min(torch.finfo(expected.dtype).eps)
    relative = absolute / denominator
    return {
        "max_abs_error": float(absolute.max().item()),
        "max_rel_error": float(relative.max().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        help="只执行指定 variant；可重复传入。默认执行全部六个 variant。",
    )
    parser.add_argument(
        "--apply-candidate-small-mm-guard",
        action="store_true",
        help="修复验证专用：禁止 NPU 进入 upstream small-mm pointwise lowering。",
    )
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    if not torch.npu.is_available():
        raise RuntimeError("当前进程不可见 NPU")
    torch.npu.set_device("npu:0")
    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"
    from torch_npu.utils._dynamo import register_inductor_npu

    register_inductor_npu()
    from torch._inductor.lowering import lowerings
    from torch._inductor.kernel import mm as upstream_mm_kernel

    if args.apply_candidate_small_mm_guard:
        original_small_mm_guard = upstream_mm_kernel._use_small_mm_pointwise

        def npu_safe_small_mm_guard(m, k, n, layout):
            if layout.device.type == "npu":
                return False
            return original_small_mm_guard(m, k, n, layout)

        upstream_mm_kernel._use_small_mm_pointwise = npu_safe_small_mm_guard

    lowering_diagnostic = {}
    for target in (torch.ops.aten.mm, torch.ops.aten.mm.default):
        handler = lowerings.get(target)
        device_handler = getattr(handler, "_torch_npu_device_handler", None)
        lowering_diagnostic[str(target)] = {
            "handler_module": getattr(handler, "__module__", None),
            "handler_name": getattr(handler, "__name__", None),
            "has_device_handler": device_handler is not None,
        }
    print(
        "T077_NPU_LOWERING="
        + json.dumps(lowering_diagnostic, ensure_ascii=False)
    )
    shapes = [
        ("large-m-small-kn-positive", 20480, 5, 2, True),
        ("k-threshold-negative", 20480, 32, 2, False),
        ("m-threshold-negative", 2048, 2, 2, False),
    ]
    observations = []
    options = {"decompose_mm_pass": {}}
    with torch._inductor.config.patch(post_grad_fusion_options=options):
        for precision, mixed in (("fp32", False), ("mixed", True)):
            for suffix, m, k, n, reference_expected_match in shapes:
                variant_id = f"{precision}-{suffix}"
                if args.variant and variant_id not in args.variant:
                    continue
                torch._dynamo.reset()
                counters.clear()
                left = torch.randn(m, k, device="npu", dtype=torch.float32)
                right = torch.randn(k, n, device="npu", dtype=torch.float32)
                ref_left = left.detach().clone().requires_grad_(True)
                ref_right = right.detach().clone().requires_grad_(True)
                run_left = left.detach().clone().requires_grad_(True)
                run_right = right.detach().clone().requires_grad_(True)

                with autocast_context(mixed):
                    expected = mm_fn(ref_left, ref_right)
                    compiled = torch.compile(
                        mm_fn,
                        options={"npu_backend": "triton_experimental"},
                    )
                    actual, codes = run_and_get_code(compiled, run_left, run_right)
                atol = 8e-3 if mixed else 1e-3
                rtol = 8e-3 if mixed else 1e-3
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                forward_count = counters["inductor"]["decompose_mm"]
                expected.sum().backward()
                actual.sum().backward()
                diagnostic = {
                    "variant_id": variant_id,
                    "forward": error_metrics(actual, expected),
                    "left_gradient": error_metrics(run_left.grad, ref_left.grad),
                    "right_gradient": error_metrics(run_right.grad, ref_right.grad),
                }
                diagnostic_path = args.artifact_dir / f"{variant_id}-diagnostic.json"
                diagnostic_path.write_text(
                    json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print("T077_NPU_DIAGNOSTIC=" + json.dumps(diagnostic, ensure_ascii=False))
                torch.testing.assert_close(
                    run_left.grad,
                    ref_left.grad,
                    atol=atol,
                    rtol=rtol,
                )
                torch.testing.assert_close(
                    run_right.grad,
                    ref_right.grad,
                    atol=atol,
                    rtol=rtol,
                )
                total_count = counters["inductor"]["decompose_mm"]
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
                        "input_dtype": str(left.dtype),
                        "autocast_dtype": "bfloat16" if mixed else None,
                        "output_dtype": str(actual.dtype),
                        "reference_expected_match": reference_expected_match,
                        "npu_forward_count": forward_count,
                        "npu_total_count": total_count,
                        "forward_correctness": "passed",
                        "gradient_correctness": "passed",
                        "generated_code": str(code_path),
                        "diagnostic": str(diagnostic_path),
                        **diagnostic,
                    }
                )

    summary = {
        "source_tests": [
            "test/inductor/test_decompose_mem_bound_mm.py::TestDecomposeMemMM.test_decompose_mm",
            "test/inductor/test_decompose_mem_bound_mm.py::TestDecomposeMemMM.test_decompose_mm_mixed_precision",
        ],
        "adapter_deviation": [
            "GPU_TYPE=npu",
            "backend=triton_experimental",
            "artifact capture",
        ],
        "product_gate_bypassed": False,
        "tests_run": len(observations),
        "selected_variants": args.variant,
        "candidate_small_mm_guard_applied": args.apply_candidate_small_mm_guard,
        "lowering_diagnostic": lowering_diagnostic,
        "upstream_generated_cases_represented": 12,
        "upstream_has_bias_parameter_values": [False, True],
        "has_bias_note": "原 test_decompose_mm 的 has_bias 参数未参与图，按 manifest 不重复计为独立 contract。",
        "successful": True,
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "guard_observation": "should_decompose_mm 只接受 cuda/xpu 或 cpu，NPU 六个语义合同均保持 mm。",
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
