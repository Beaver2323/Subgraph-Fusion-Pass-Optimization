#!/usr/bin/env python3
"""T-078 addcdiv 低精度单臂 worker；每次进程只激活一个后端/arm。"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback


WORK = Path("/home/z50063656/tmp")
EXPECTED_COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
EXPERIMENTAL_SOURCE = Path(
    "/home/z50063656/Pass/src/torch_npu/torch_npu/_inductor/triton_experimental"
)

if Path.cwd().resolve() != WORK:
    raise RuntimeError("必须从 /home/z50063656/tmp 启动")
if os.environ.get("TORCHINDUCTOR_NPU_BACKEND") != "triton_experimental":
    raise RuntimeError("必须在导入 torch 前指定 triton_experimental")

import torch
import torch_npu
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu


def tensor_sha256(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def verify_source_overlay():
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


def make_inputs(dtype):
    # 从 CPU 固定 generator 构造输入，保证各 fresh process 的三个 arm 完全相同。
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260908)
    tensors = [
        torch.randn((64, 64), dtype=dtype, generator=generator)
        for _ in range(3)
    ]
    tensors[2] = tensors[2].abs().clamp(min=0.1)
    hashes = [tensor_sha256(tensor) for tensor in tensors]
    return tuple(tensor.to("npu") for tensor in tensors), hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument(
        "--arm", choices=("off", "decomposed", "re-fused"), required=True
    )
    args = parser.parse_args()
    # TORCH_COMPILE_DEBUG_DIR 可能在 torch 导入阶段先创建父目录；允许该空目录，
    # 但绝不覆盖已有 arm 结果。
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if (args.artifact_dir / "arm_result.json").exists():
        raise RuntimeError(f"拒绝覆盖已有 arm 结果：{args.artifact_dir}")
    result = {
        "schema_version": "1.0",
        "task": "T-078",
        "acceptance_unit_id": "AU-post-grad-fuse-addcdiv-to-fma",
        "dtype": args.dtype,
        "arm": args.arm,
        "backend": None,
        "status": "error",
        "product_source_modified": False,
        "cpu_fallback_added": False,
        "performance_measured": False,
    }
    try:
        register_inductor_npu()
        if _InductorNpuRegistry._loaded_backend != "triton_experimental":
            raise RuntimeError(
                f"unexpected backend: {_InductorNpuRegistry._loaded_backend}"
            )
        if torch.version.git_version != EXPECTED_COMMIT:
            raise RuntimeError(f"unexpected PyTorch commit: {torch.version.git_version}")
        torch.npu.set_device(0)
        result["backend"] = _InductorNpuRegistry._loaded_backend
        result["torch_version"] = torch.__version__
        result["torch_commit"] = torch.version.git_version
        result["torch_npu_version"] = torch_npu.__version__
        result["source_modules"] = verify_source_overlay()

        if args.arm == "re-fused":
            # 只在本进程内放宽到浮点 dtype，探查候选能力；不修改产品源码。
            from candidate_adapter import (
                register_candidate_lowering,
                register_candidate_pattern,
            )

            register_candidate_lowering()
            register_candidate_pattern()

        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        inputs, input_hashes = make_inputs(dtype)

        def addcdiv_fn(s, t1, t2):
            return torch.addcdiv(s, t1, t2, value=2.0)

        def decomposed_fn(s, t1, t2):
            return s + (t1 / t2) * 2.0

        fn = decomposed_fn if args.arm == "decomposed" else addcdiv_fn
        eager = addcdiv_fn(*inputs)
        torch._dynamo.reset()
        counters.clear()
        compiled_fn = torch.compile(fn, fullgraph=True)
        with config.patch(
            {
                "test_configs.runtime_triton_dtype_assert": False,
                "test_configs.runtime_triton_shape_assert": False,
            }
        ):
            outputs, codes = run_and_get_code(compiled_fn, *inputs)
        torch.npu.synchronize()
        actual = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        code = "\n".join(codes)
        (args.artifact_dir / "generated_code.py").write_text(code, encoding="utf-8")
        delta = actual.float() - eager.float()
        bitwise_equal = bool(torch.equal(actual, eager))
        result.update(
            {
                "input_sha256": input_hashes,
                "eager_sha256": tensor_sha256(eager),
                "compiled_sha256": tensor_sha256(actual),
                "shape": list(actual.shape),
                "output_dtype": str(actual.dtype),
                "bitwise_equal_to_addcdiv_eager": bitwise_equal,
                "max_abs_error_to_addcdiv_eager": float(delta.abs().max().item()),
                "mean_abs_error_to_addcdiv_eager": float(delta.abs().mean().item()),
                "mismatch_count": int((actual != eager).sum().item()),
                "addcdiv_fma_fused": counters["inductor"].get(
                    "addcdiv_fma_fused", 0
                ),
                "generated_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "generated_code_contains_fma": "tl.fma" in code,
                "generated_code_contains_div_rn": "triton.language.div_rn" in code,
                "expected_fusion": 1 if args.arm == "re-fused" else 0,
                "status": "passed",
            }
        )
        if result["addcdiv_fma_fused"] != result["expected_fusion"]:
            raise AssertionError(
                "fusion counter 不符合 arm 合同: "
                f"actual={result['addcdiv_fma_fused']} "
                f"expected={result['expected_fusion']}"
            )
    except Exception as error:
        result.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    (args.artifact_dir / "arm_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
