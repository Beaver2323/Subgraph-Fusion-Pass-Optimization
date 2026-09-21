# Owner(s): ["module: inductor"]

import os
import unittest

os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
import torch_npu  # noqa: F401


@unittest.skipIf(not torch.npu.is_available(), "requires NPU")
class TestTritonExperimentalAddcdivPrecision(TestCase):
    def test_fp16_value_one_preserves_div_rounding_without_refusion(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260908)
        inputs = [
            torch.randn((64, 64), dtype=torch.float16, generator=generator)
            for _ in range(3)
        ]
        inputs[2] = inputs[2].abs().clamp(min=0.1)
        inputs = [value.npu() for value in inputs]

        def fn(self, tensor1, tensor2):
            return torch.addcdiv(self, tensor1, tensor2, value=1.0)

        eager = fn(*inputs)
        torch._dynamo.reset()
        counters.clear()
        compiled = torch.compile(fn, fullgraph=True)
        with config.patch(
            {
                "force_disable_caches": True,
                "test_configs.runtime_triton_dtype_assert": False,
                "test_configs.runtime_triton_shape_assert": False,
            }
        ):
            actual, codes = run_and_get_code(compiled, *inputs)
        torch.npu.synchronize()
        code = "\n".join(codes)

        self.assertTrue(torch.equal(actual, eager))
        self.assertEqual(counters["inductor"]["addcdiv_fma_fused"], 0)
        self.assertIn(".to(tl.float16)", code)
        self.assertNotIn("tl.fma", code)
        self.assertNotIn("triton.language.div_rn", code)

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("value", [2.0, 0.1, -0.75, 0.3, 1.3])
    def test_scaled_addcdiv_matches_eager(self, dtype, value):
        generator = torch.Generator().manual_seed(20260908)
        inputs = [
            torch.randn((64, 64), dtype=dtype, generator=generator)
            for _ in range(3)
        ]
        inputs[2] = inputs[2].abs().clamp(min=0.1)
        inputs = [x.npu() for x in inputs]
        self.check_scaled_addcdiv(inputs, value)

    def test_fp16_strided_and_broadcast_inputs(self):
        generator = torch.Generator().manual_seed(20260908)
        inputs = [
            torch.randn(shape, dtype=torch.float16, generator=generator).npu()
            for shape in ((64, 128), (1, 64), (64, 1))
        ]
        inputs[0] = inputs[0][:, ::2]
        inputs[2] = inputs[2].abs().clamp(min=0.1)
        self.assertFalse(inputs[0].is_contiguous())
        self.check_scaled_addcdiv(inputs, 0.1)

    def check_scaled_addcdiv(self, inputs, value):
        def fn(inp, tensor1, tensor2):
            return torch.addcdiv(inp, tensor1, tensor2, value=value)

        expected = fn(*inputs)
        torch._dynamo.reset()
        counters.clear()
        with config.patch(
            {
                "force_disable_caches": True,
                "test_configs.runtime_triton_dtype_assert": False,
                "test_configs.runtime_triton_shape_assert": False,
            }
        ):
            actual, codes = run_and_get_code(torch.compile(fn, fullgraph=True), *inputs)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(counters["inductor"]["addcdiv_fma_fused"], 1)
        code = "\n".join(codes)
        if inputs[0].dtype == torch.float16:
            self.assertIn(".to(tl.float16)", code)
            self.assertNotIn("tl.fma", code)
            self.assertNotIn("triton.language.div_rn", code)
        else:
            self.assertIn("tl.fma", code)
            self.assertIn("triton.language.div_rn", code)


instantiate_parametrized_tests(TestTritonExperimentalAddcdivPrecision)


if __name__ == "__main__":
    run_tests()
