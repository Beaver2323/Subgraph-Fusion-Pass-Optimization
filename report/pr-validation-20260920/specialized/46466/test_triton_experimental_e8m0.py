# Owner(s): ["module: inductor"]

import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch.testing._internal.common_utils import TestCase, run_tests
import torch_npu  # noqa: F401
from torch_npu._inductor.triton_experimental import config as npu_config


def encode(x):
    return (torch.clamp(torch.ceil(torch.log2(x)), -127, 127) + 127).to(torch.uint8)


def exact_encoding(value):
    # frexp avoids the rounding of log2 just above a power of two. This oracle
    # does not reuse the integer bit-extraction implementation being tested.
    if math.isnan(value) or value <= 0:
        return 0
    if math.isinf(value):
        return 254
    mantissa, exponent = math.frexp(value)
    ceiling = exponent - 1 if mantissa == 0.5 else exponent
    return min(254, max(0, ceiling + 127))


@unittest.skipIf(not torch.npu.is_available(), "requires NPU")
class TestTritonExperimentalE8M0(TestCase):
    def check_encoding(self, inp, expected, should_rewrite):
        torch._dynamo.reset()
        counters.clear()
        with (
            config.patch(force_disable_caches=True),
            patch.dict(os.environ, TORCHINDUCTOR_PATTERN_MATCH_DEBUG="__none__"),
        ):
            actual = torch.compile(encode, fullgraph=True)(inp)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        count = counters["inductor_pattern_matcher_per_pattern"][
            "e8m0_rceil_log2_pattern"
        ]
        if should_rewrite:
            self.assertGreater(count, 0)
        else:
            self.assertEqual(count, 0)

    def test_exponent_boundaries_and_special_values(self):
        bits = [
            (exponent << 23) | mantissa
            for exponent in range(1, 255)
            for mantissa in (0, 1, 0x3FFFFF, 0x7FFFFF)
        ]
        bits += [
            0, 0x80000000, 1, 0x3FFFFF, 0x400000, 0x400001, 0x7FFFFF,
            0x7F800000, 0xFF800000, 0xBF800000, 0x80000001,
            0x7FC00000, 0xFFC00000,
        ]
        values = [struct.unpack("<f", struct.pack("<I", b))[0] for b in bits]
        inp = torch.tensor(values, device="npu", dtype=torch.float32)
        expected = torch.tensor(
            [exact_encoding(v) for v in values], device="npu", dtype=torch.uint8
        )
        self.check_encoding(inp, expected, True)

    def test_strided_input(self):
        inp = torch.tensor(
            [1.0, 42.0, 1.0000001192092896, 42.0, 0.5, 42.0, 3.0, 42.0],
            device="npu",
        )[::2]
        self.assertFalse(inp.is_contiguous())
        expected = torch.tensor([127, 128, 126, 129], device="npu", dtype=torch.uint8)
        self.check_encoding(inp, expected, True)

    def test_non_fp32_inputs_are_not_rewritten(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                inp = torch.tensor([0.25, 0.5, 1.0, 1.5, 2.0, 3.0], device="npu", dtype=dtype)
                self.check_encoding(inp, encode(inp), False)

    def test_cpu_input_is_not_rewritten(self):
        inp = torch.tensor([0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
        self.check_encoding(inp, encode(inp), False)

    def test_disabled_rewrite(self):
        # Start a fresh process so the disabled setting precedes registration
        # and cannot reuse compiled artifacts from the enabled cases.
        marker = "TORCH_NPU_TEST_E8M0_DISABLED"
        if os.environ.get(marker) != "1":
            env = dict(os.environ, **{marker: "1", "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1"})
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 f"{type(self).__name__}.test_disabled_rewrite", "-v"],
                env=env, capture_output=True, text=True, timeout=600,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return
        npu_config.enable_e8m0_rceil_log2 = False
        inp = torch.tensor([0.25, 0.5, 1.0, 1.5, 2.0, 3.0], device="npu")
        self.check_encoding(inp, encode(inp), False)


if __name__ == "__main__":
    run_tests()
