# Owner(s): ["module: inductor"]

import math
import os
import unittest
from unittest.mock import patch

os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
import torch_npu  # noqa: F401


@unittest.skipIf(not torch.npu.is_available(), "requires NPU")
class TestTritonExperimentalSfdpTraining(TestCase):
    @parametrize("pattern", [1, 2, 3, 4, 5])
    @parametrize("dtype", [torch.float32, torch.float16])
    def test_training_pattern(self, pattern, dtype):
        def attention(query, key, value):
            scores = query @ key.transpose(-2, -1)
            if pattern == 1:
                weights = (scores / math.sqrt(key.shape[-1])).softmax(dim=-1)
            elif pattern == 2:
                weights = (scores * (1.0 / math.sqrt(key.shape[-1]))).softmax(dim=-1)
            elif pattern == 3:
                weights = torch.nn.functional.dropout(
                    (scores / 3.0).softmax(dim=-1), p=0.4, training=True
                )
            elif pattern == 4:
                weights = torch.nn.functional.dropout(
                    (scores * 0.4).softmax(dim=-1), p=0.2, training=True
                )
            else:
                mask = torch.ones(
                    query.shape[-2], key.shape[-2], device=query.device,
                    dtype=torch.bool,
                ).tril()
                weights = (scores / math.sqrt(key.shape[-1]) + mask).softmax(dim=-1)
            return weights @ value

        generator = torch.Generator().manual_seed(1234)
        inputs = [
            torch.randn((4, 2, 16, 32), dtype=dtype, generator=generator)
            .npu().requires_grad_()
            for _ in range(3)
        ]
        reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
        torch._dynamo.reset()
        counters.clear()
        with (
            config.patch(force_disable_caches=True),
            patch.dict(os.environ, TORCHINDUCTOR_PATTERN_MATCH_DEBUG="__none__"),
        ):
            actual = torch.compile(attention, fullgraph=True)(*inputs)
            actual.sum().backward()

        matches = counters["inductor_pattern_matcher_per_pattern"]
        training_matches = sum(
            count for name, count in matches.items()
            if name.startswith(f"_sfdp_pattern_{pattern}_")
            and name.endswith("_training")
        )
        self.assertGreater(training_matches, 0, str(matches))
        self.assertEqual(actual.device, inputs[0].device)
        self.assertTrue(torch.isfinite(actual).all().item())
        for inp in inputs:
            self.assertIsNotNone(inp.grad)
            self.assertTrue(torch.isfinite(inp.grad).all().item())

        # Dropout fusion uses a different random mask from eager. Its cases
        # verify training pattern selection and backward execution, not
        # elementwise equality between independent random samples.
        if pattern not in (3, 4):
            expected = attention(*reference_inputs)
            expected.sum().backward()
            self.assertEqual(actual, expected, atol=1.5e-3, rtol=1e-2)
            for inp, reference in zip(inputs, reference_inputs):
                self.assertEqual(inp.grad, reference.grad, atol=1.5e-3, rtol=1e-2)


instantiate_parametrized_tests(TestTritonExperimentalSfdpTraining)

if __name__ == "__main__":
    run_tests()
