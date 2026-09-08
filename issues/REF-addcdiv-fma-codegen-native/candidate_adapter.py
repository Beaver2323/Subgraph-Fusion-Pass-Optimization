#!/usr/bin/env python3
"""addcdiv NPU post-grad/FMA lowering 候选能力探针；不修改冻结 PyTorch。"""

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
import unittest


if os.environ.get("TORCHINDUCTOR_NPU_BACKEND") != "triton_experimental":
    raise RuntimeError("必须在导入 torch 前指定 triton_experimental")
if Path.cwd().resolve() != Path("/home/z50063656/tmp"):
    raise RuntimeError("必须从 /home/z50063656/tmp 启动")

import torch
import torch_npu
import torch_npu.testing
from torch._dynamo.utils import counters
from torch._inductor import lowering
from torch._inductor.fx_passes import post_grad
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
    register_graph_pattern,
)
from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu


ATEN = torch.ops.aten
EXPECTED_COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
TORCH_NPU_EXPERIMENTAL_SOURCE = Path(
    "/home/z50063656/Pass/src/torch_npu/torch_npu/_inductor/triton_experimental"
)


def register_candidate_lowering() -> None:
    """在 experimental fallback 注册之后恢复 addcdiv 的 NPU FMA lowering。"""

    @lowering.register_lowering(ATEN.addcdiv.default, broadcast=True)
    def npu_addcdiv(self, tensor1, tensor2, *, value=1):
        dtype = lowering.get_promoted_dtype(
            self,
            tensor1,
            tensor2,
            type_promotion_kind=(
                lowering.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
            ),
        )
        self_loader = self.make_loader()
        t1_loader = tensor1.make_loader()
        t2_loader = tensor2.make_loader()

        def inner_fn(idx):
            self_val = self_loader(idx)
            t1_val = t1_loader(idx)
            t2_val = t2_loader(idx)

            # NPU eager FP16 addcdiv follows the observable low-precision
            # arithmetic sequence rather than CUDA's FP32-accumulating FMA:
            # div -> FP16 round -> mul -> FP16 round -> add -> FP16 store.
            # Keep those two internal rounding boundaries inside one Triton
            # kernel; otherwise Inductor's compute-type promotion collapses
            # the whole expression to FP32 and only rounds the final store.
            if dtype == torch.float16:
                quotient = t1_val / t2_val
                quotient = lowering.ops.to_dtype(
                    quotient,
                    torch.float16,
                    src_dtype=torch.float32,
                    use_compute_types=False,
                )
                quotient = lowering.ops.to_dtype(
                    quotient,
                    torch.float16,
                    src_dtype=torch.float16,
                )
                if isinstance(value, lowering.sympy.Basic):
                    value_expr = lowering.ops.index_expr(value, dtype)
                else:
                    # Wrapped Python scalars are converted to the tensor dtype
                    # by eager FP16 arithmetic.  Materialize that conversion at
                    # lowering time so Triton's float32 compute literal does not
                    # silently retain extra scalar precision.
                    quantized_value = torch.tensor(
                        value, dtype=torch.float16
                    ).item()
                    value_expr = lowering.ops.constant(quantized_value, dtype)
                product = lowering.ops.mul(quotient, value_expr)
                product = lowering.ops.to_dtype(
                    product,
                    torch.float16,
                    src_dtype=torch.float32,
                    use_compute_types=False,
                )
                product = lowering.ops.to_dtype(
                    product,
                    torch.float16,
                    src_dtype=torch.float16,
                )
                return lowering.ops.add(self_val, product)

            quotient = lowering.ops.div_rn(t1_val, t2_val)
            if value == 1:
                return lowering.ops.add(self_val, quotient)
            if isinstance(value, lowering.sympy.Basic):
                value_expr = lowering.ops.index_expr(value, dtype)
            else:
                value_expr = lowering.ops.constant(value, dtype)
            return lowering.ops.fma(value_expr, quotient, self_val)

        return lowering.Pointwise.create(
            device=self.get_device(),
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=self.get_size(),
        )


def register_candidate_pattern() -> None:
    """注册只接受 NPU 浮点标量合同的 addcdiv 重融合 pattern。"""

    def eligible(match: Match) -> bool:
        inp_val = match.kwargs["inp"].meta.get("val")
        out_val = match.output_node().meta.get("val")
        supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        if not (
            isinstance(inp_val, torch.Tensor)
            and inp_val.dtype in supported_dtypes
            and isinstance(out_val, torch.Tensor)
            and out_val.device.type == "npu"
            and out_val.dtype in supported_dtypes
        ):
            return False
        for key in ("t1", "t2"):
            node = match.kwargs.get(key)
            val = node.meta.get("val") if isinstance(node, torch.fx.Node) else node
            if not (
                isinstance(val, torch.Tensor) and val.dtype in supported_dtypes
            ):
                return False
        value = match.kwargs.get("value", 1)
        if isinstance(value, torch.fx.Node):
            return False
        if out_val.dtype == torch.float16 and not isinstance(value, (int, float)):
            return False
        return True

    @register_graph_pattern(
        CallFunction(
            ATEN.add.Tensor,
            KeywordArg("inp"),
            CallFunction(
                ATEN.mul.Tensor,
                CallFunction(
                    ATEN.div.Tensor,
                    KeywordArg("t1"),
                    KeywordArg("t2"),
                ),
                KeywordArg("value"),
            ),
        ),
        pass_dict=post_grad.pass_patterns[2],
        extra_check=eligible,
    )
    def fuse_addcdiv(match: Match, inp, t1, t2, value) -> None:
        def repl(inp, t1, t2, value):
            return ATEN.addcdiv(inp, t1, t2, value=value)

        counters["inductor"]["addcdiv_fma_fused"] += 1
        match.replace_by_example(repl, [inp, t1, t2, value])

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--source-fix",
        action="store_true",
        help="让真实后端激活流程从 torch_npu 工作树加载本次正式修复",
    )
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    source_modules = {}
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == "triton_experimental"
    assert torch.version.git_version == EXPECTED_COMMIT
    torch.npu.set_device(0)
    if args.source_fix:
        if not getattr(sys, "_t078_formal_source_overlay", None):
            raise RuntimeError("formal source sitecustomize overlay 未激活")
        for short_name in ("overrides", "fx_passes", "lowering"):
            module = importlib.import_module(
                f"torch_npu._inductor.triton_experimental.{short_name}"
            )
            source_modules[short_name] = str(Path(module.__file__).resolve())
        expected_modules = {
            name: str((TORCH_NPU_EXPERIMENTAL_SOURCE / f"{name}.py").resolve())
            for name in source_modules
        }
        if source_modules != expected_modules:
            raise RuntimeError(
                "formal source overlay 未加载目标工作树: "
                f"actual={source_modules}, expected={expected_modules}"
            )
    else:
        register_candidate_lowering()
        register_candidate_pattern()

    source = Path(
        "/home/z50063656/Pass/src/pytorch/test/inductor/test_torchinductor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "t078_codegen_candidate_upstream", source
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream.GPU_TYPE = "npu"
    method = inspect.unwrap(
        upstream.CommonTemplate.test_addcdiv_fma_uses_fma_and_div_rn
    )
    observations = []

    class NpuCandidateCase(upstream.TestCase):
        device = "npu"
        test_addcdiv_codegen = method

        @classmethod
        def setUpClass(cls):
            super().setUpClass()
            cls._stack.enter_context(
                upstream.config.patch(
                    {
                        "test_configs.runtime_triton_dtype_assert": False,
                        "test_configs.runtime_triton_shape_assert": False,
                    }
                )
            )

        def assertEqual(self, actual, expected, *positional, **kwargs):
            if isinstance(actual, int) and isinstance(expected, int):
                observations.append(
                    {
                        "assertion": "counter-equal",
                        "actual": actual,
                        "expected": expected,
                    }
                )
            return super().assertEqual(actual, expected, *positional, **kwargs)

        def assertIn(self, member, container, *positional, **kwargs):
            if member in ("tl.fma", "triton.language.div_rn"):
                observations.append(
                    {
                        "assertion": "code-contains",
                        "needle": member,
                        "present": member in container,
                    }
                )
            return super().assertIn(member, container, *positional, **kwargs)

    counters.clear()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite([NpuCandidateCase("test_addcdiv_codegen")])
    )
    assertions_valid = (
        len(observations) == 3
        and observations[0] == {
            "assertion": "counter-equal",
            "actual": 1,
            "expected": 1,
        }
        and all(
            item.get("present") is True
            for item in observations[1:]
            if item.get("assertion") == "code-contains"
        )
    )
    correctness = {"status": "not-run"}
    if result.wasSuccessful() and assertions_valid:
        try:
            torch._dynamo.reset()
            counters.clear()
            torch.npu.manual_seed(20260906)
            self_tensor = torch.randn(64, 64, device="npu", dtype=torch.float32)
            tensor1 = torch.randn(64, 64, device="npu", dtype=torch.float32)
            tensor2 = (
                torch.randn(64, 64, device="npu", dtype=torch.float32)
                .abs()
                .clamp(min=0.1)
            )

            @torch.compile(fullgraph=True)
            def literal_value_two(s, t1, t2):
                return torch.addcdiv(s, t1, t2, value=2.0)

            eager = torch.addcdiv(
                self_tensor, tensor1, tensor2, value=2.0
            )
            compiled = literal_value_two(self_tensor, tensor1, tensor2)
            torch.npu.synchronize()
            correctness = {
                "status": "passed" if torch.equal(compiled, eager) else "failed",
                "source": "tracker-derived-literal-value-two",
                "shape": list(compiled.shape),
                "dtype": str(compiled.dtype),
                "device": str(compiled.device),
                "bitwise_equal": bool(torch.equal(compiled, eager)),
                "max_abs_error": float((compiled - eager).abs().max().item()),
                "addcdiv_fma_fused": counters["inductor"][
                    "addcdiv_fma_fused"
                ],
            }
        except Exception as error:
            correctness = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
    correctness_valid = (
        correctness.get("status") == "passed"
        and correctness.get("bitwise_equal") is True
        and correctness.get("addcdiv_fma_fused") == 1
    )
    summary = {
        "tracking_mode": (
            "source-fix-verification"
            if args.source_fix
            else "candidate-capability-probe"
        ),
        "source_test": (
            f"{source}::CommonTemplate.test_addcdiv_fma_uses_fma_and_div_rn"
        ),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "backend": _InductorNpuRegistry._loaded_backend,
        "candidate_only": not args.source_fix,
        "formal_source_overlay": args.source_fix,
        "formal_source_modules": source_modules,
        "frozen_source_modified": False,
        "cpu_fallback_added": False,
        "disabled_test_added": False,
        "candidate_npu_pattern_registered": not args.source_fix,
        "candidate_npu_fma_lowering_registered": not args.source_fix,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "community_assertions_passed": result.wasSuccessful() and not result.skipped,
        "assertions_valid": assertions_valid,
        "derived_correctness": correctness,
        "derived_correctness_valid": correctness_valid,
        "observations": observations,
        "compatibility_verdict": (
            (
                "source-fix-codegen-valid"
                if args.source_fix
                else "candidate-codegen-valid"
            )
            if result.wasSuccessful() and assertions_valid and correctness_valid
            else (
                "source-fix-codegen-invalid"
                if args.source_fix
                else "candidate-codegen-invalid"
            )
        ),
    }
    result_name = (
        "source_fix_result.json"
        if args.source_fix
        else "candidate_result.json"
    )
    (args.artifact_dir / result_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return (
        0
        if result.wasSuccessful() and assertions_valid and correctness_valid
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
