#!/usr/bin/env python3
"""T-078 addcdiv 社区合同的 FP16/BF16 GPU dtype 扩展测例。

本文件不是 PyTorch 社区原生测例。它固定复用社区测例的 64x64 shape、
value 子分支、eager/compiled 对照和对应命中判据，只改变 dtype。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import unittest


CODEGEN_SOURCE_TEST = (
    "test/inductor/test_torchinductor.py::"
    "GPUTests.test_addcdiv_fma_uses_fma_and_div_rn_cuda"
)
BITWISE_SOURCE_TEST = (
    "test/inductor/test_torchinductor.py::"
    "GPUTests.test_addcdiv_fma_bitwise_equal_cuda"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-test", required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--value", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.source_test not in {CODEGEN_SOURCE_TEST, BITWISE_SOURCE_TEST}:
        raise ValueError("source-test 与已审核的 addcdiv 社区合同不一致")
    if args.source_test == CODEGEN_SOURCE_TEST and args.value != 2.0:
        raise ValueError("codegen 社区合同只允许 value=2")
    if args.source_test == BITWISE_SOURCE_TEST and args.value not in {1.0, 2.0}:
        raise ValueError("bitwise 社区合同只包含 value=1/2")
    if "torch_npu" in Path.cwd().resolve().parts:
        raise RuntimeError("不得从 torch_npu 源码树内启动测试")

    import torch
    from torch._dynamo.utils import counters
    from torch._inductor.utils import run_and_get_code

    class AddcdivDtypeReference(unittest.TestCase):
        def test_dtype_extension(self) -> None:
            if not torch.cuda.is_available():
                self.fail("CUDA 不可用；派生 reference 不允许 SKIP")

            dtype = getattr(torch, args.dtype)
            torch.manual_seed(20260908)
            self_tensor = torch.randn(64, 64, device="cuda", dtype=dtype)
            tensor1 = torch.randn(64, 64, device="cuda", dtype=dtype)
            tensor2 = (
                torch.randn(64, 64, device="cuda", dtype=dtype)
                .abs()
                .clamp(min=0.1)
            )

            def fn(s, t1, t2):
                return torch.addcdiv(s, t1, t2, value=args.value)

            expected = fn(self_tensor, tensor1, tensor2)
            torch._dynamo.reset()
            counters.clear()
            actual, source_codes = run_and_get_code(
                torch.compile(fn, fullgraph=True),
                self_tensor,
                tensor1,
                tensor2,
            )
            code = "\n".join(source_codes)
            max_abs_error = (actual.float() - expected.float()).abs().max().item()
            print(
                f"dtype={args.dtype} value={args.value} "
                f"bitwise_equal={torch.equal(actual, expected)} "
                f"max_abs_error={max_abs_error}"
            )

            self.assertTrue(
                torch.equal(actual, expected),
                f"{args.dtype} compiled/eager 非位级一致；max_abs_error={max_abs_error}",
            )
            expected_fusion = 0 if args.value == 1.0 else 1
            self.assertEqual(
                counters["inductor"].get("addcdiv_fma_fused", 0),
                expected_fusion,
            )
            if expected_fusion:
                self.assertIn("tl.fma", code)
                self.assertIn("triton.language.div_rn", code)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AddcdivDtypeReference)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
