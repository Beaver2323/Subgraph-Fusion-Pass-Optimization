# Owner(s): ["module: inductor"]

import os
import re
import unittest

os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"

import torch
import torch.compiler.config as compiler_config
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import TestCase, run_tests
import torch_npu  # noqa: F401


@unittest.skipIf(not torch.npu.is_available(), "requires NPU")
class TestTritonExperimentalCompileOnOneRank(TestCase):
    @staticmethod
    def factory_and_reduction(x):
        return torch.zeros(4, x.shape[1], device=x.device, dtype=x.dtype) + x.sum()

    def compile_on_device(self, device, dtype=torch.float32):
        torch._dynamo.reset()
        with torch.npu.device(device), config.patch(force_disable_caches=True):
            x = torch.arange(16, device="npu", dtype=dtype).reshape(2, 8)
            actual, codes = run_and_get_code(
                torch.compile(self.factory_and_reduction, fullgraph=True), x
            )
            self.assertEqual(actual, self.factory_and_reduction(x))
            self.assertEqual(actual.device, x.device)
        return "\n".join(codes)

    def assert_runtime_device(self, code):
        self.assertIn("torch.npu.current_device()", code)
        self.assertNotRegex(code, r"npu:\d")
        self.assertNotRegex(code, r"device\(type=.npu., index=\d")
        self.assertNotRegex(code, r"DeviceProperties\([^)]*index=\d")

    @compiler_config.patch(compile_on_one_rank=True)
    def test_compiles_with_runtime_device(self):
        self.assert_runtime_device(self.compile_on_device(torch.npu.current_device()))

    @compiler_config.patch(compile_on_one_rank=False)
    def test_ordinary_compile_preserves_device_index(self):
        device = torch.npu.current_device()
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                code = self.compile_on_device(device, dtype)
                self.assertRegex(code, rf"DeviceProperties\([^)]*index={device}\b")
                self.assertNotIn("_coor_device_idx", code)

    @unittest.skipIf(torch.npu.device_count() < 2, "requires two NPUs")
    @compiler_config.patch(compile_on_one_rank=True)
    def test_generated_code_identical_across_devices(self):
        # Both harnesses can introduce device literals into generated kernels.
        for options in (
            {"benchmark_kernel": True},
            {"triton.autotune_at_compile_time": True},
        ):
            with self.subTest(options=options), config.patch(options):
                codes = [self.compile_on_device(device) for device in (0, 1)]
                for code in codes:
                    self.assert_runtime_device(code)
                # AOT counters differ between compilations in the same process.
                normalized = [
                    re.sub(r"AOT ID: \['\d+_", "AOT ID: ['N_", code)
                    for code in codes
                ]
                self.assertEqual(*normalized)


if __name__ == "__main__":
    run_tests()
