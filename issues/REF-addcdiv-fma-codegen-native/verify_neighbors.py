#!/usr/bin/env python3
"""逐个验证 T-078 addcdiv 正式修复的 dtype 与拒绝 guard 边界。"""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback


if os.environ.get("TORCHINDUCTOR_NPU_BACKEND") != "triton_experimental":
    raise RuntimeError("必须在导入 torch 前指定 triton_experimental")
if Path.cwd().resolve() != Path("/home/z50063656/tmp"):
    raise RuntimeError("必须从 /home/z50063656/tmp 启动")

import torch
import torch_npu
from torch._dynamo.utils import counters
from torch._inductor import config
from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu


EXPECTED_COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
EXPERIMENTAL_SOURCE = Path(
    "/home/z50063656/Pass/src/torch_npu/torch_npu/_inductor/triton_experimental"
)


def verify_source_modules():
    if not getattr(sys, "_t078_formal_source_overlay", None):
        raise RuntimeError("formal source sitecustomize overlay 未激活")
    actual = {}
    for short_name in ("overrides", "fx_passes", "lowering"):
        module = importlib.import_module(
            f"torch_npu._inductor.triton_experimental.{short_name}"
        )
        actual[short_name] = str(Path(module.__file__).resolve())
    expected = {
        name: str((EXPERIMENTAL_SOURCE / f"{name}.py").resolve())
        for name in actual
    }
    if actual != expected:
        raise RuntimeError(
            f"formal source overlay 路径不符: actual={actual}, expected={expected}"
        )
    return actual


def make_case(case_name):
    shape = (64, 64)
    if case_name in ("fp16", "bfloat16"):
        dtype = torch.float16 if case_name == "fp16" else torch.bfloat16
        self_tensor = torch.randn(shape, device="npu", dtype=dtype)
        tensor1 = torch.randn(shape, device="npu", dtype=dtype)
        tensor2 = (
            torch.randn(shape, device="npu", dtype=dtype).abs().clamp(min=0.1)
        )

        def fn(s, t1, t2):
            return torch.addcdiv(s, t1, t2, value=2.0)

        # BF16 走 FMA；FP16 走同一 re-fusion 入口下的显式低精度舍入
        # lowering。两者都必须命中并与各自 NPU eager 逐位一致。
        expected_fusion = 1
        return fn, (self_tensor, tensor1, tensor2), dtype, expected_fusion

    if case_name == "integer-self-guard":
        self_tensor = torch.randint(0, 8, shape, device="npu", dtype=torch.int32)
        tensor1 = torch.randn(shape, device="npu", dtype=torch.float32)
        tensor2 = torch.randn(shape, device="npu").abs().clamp(min=0.1)

        def fn(s, t1, t2):
            return s + (t1 / t2) * 2.0

        return fn, (self_tensor, tensor1, tensor2), torch.float32, 0

    if case_name == "tensor-value-guard":
        self_tensor = torch.randn(shape, device="npu", dtype=torch.float32)
        tensor1 = torch.randn(shape, device="npu", dtype=torch.float32)
        tensor2 = torch.randn(shape, device="npu").abs().clamp(min=0.1)
        value = torch.tensor(2.0, device="npu", dtype=torch.float32)

        def fn(s, t1, t2, v):
            return s + (t1 / t2) * v

        return fn, (self_tensor, tensor1, tensor2, value), torch.float32, 0

    raise AssertionError(f"unsupported case: {case_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        required=True,
        choices=("fp16", "bfloat16", "integer-self-guard", "tensor-value-guard"),
    )
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "tracking_mode": "source-fix-neighbor-verification",
        "case": args.case,
        "source": "tracker-derived-from-upstream-addcdiv-contract",
        "backend": None,
        "formal_source_modules": {},
        "cpu_fallback_added": False,
        "disabled_test_added": False,
        "status": "error",
    }
    try:
        register_inductor_npu()
        if _InductorNpuRegistry._loaded_backend != "triton_experimental":
            raise RuntimeError(
                f"unexpected backend: {_InductorNpuRegistry._loaded_backend}"
            )
        if torch.version.git_version != EXPECTED_COMMIT:
            raise RuntimeError(f"unexpected PyTorch commit: {torch.version.git_version}")
        summary["backend"] = _InductorNpuRegistry._loaded_backend
        torch.npu.set_device(0)
        summary["formal_source_modules"] = verify_source_modules()

        torch._dynamo.reset()
        counters.clear()
        torch.npu.manual_seed(20260906)
        fn, inputs, dtype, expected_fusion = make_case(args.case)
        eager = fn(*inputs)
        with config.patch(
            {
                "test_configs.runtime_triton_dtype_assert": False,
                "test_configs.runtime_triton_shape_assert": False,
            }
        ):
            compiled = torch.compile(fn, fullgraph=True)(*inputs)
        torch.npu.synchronize()

        bitwise_equal = bool(torch.equal(compiled, eager))
        max_abs_error = float((compiled.float() - eager.float()).abs().max().item())
        actual_fusion = counters["inductor"]["addcdiv_fma_fused"]
        bitwise_required = args.case in ("fp16", "bfloat16")
        valid = (
            actual_fusion == expected_fusion
            and compiled.shape == eager.shape
            and compiled.dtype == eager.dtype
            and (bitwise_equal or not bitwise_required)
        )
        summary.update(
            {
                "shape": list(compiled.shape),
                "dtype": str(dtype),
                "bitwise_equal": bitwise_equal,
                "max_abs_error": max_abs_error,
                "expected_addcdiv_fma_fused": expected_fusion,
                "actual_addcdiv_fma_fused": actual_fusion,
                "guard_preserved": actual_fusion == expected_fusion,
                "bitwise_required": bitwise_required,
                "status": "passed" if valid else "failed",
            }
        )
    except Exception as error:
        summary.update(
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )

    (args.artifact_dir / "neighbor_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
