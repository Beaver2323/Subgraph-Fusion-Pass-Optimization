#!/usr/bin/env python3
"""pad-mm 输出 stride community case 的 NPU 产品基线适配器。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

import torch
import torch_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch-root",
        type=Path,
        default=Path("/home/z50063656/Pass/src/pytorch"),
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def load_upstream(pytorch_root: Path):
    source = pytorch_root / "test/inductor/test_pad_mm.py"
    spec = importlib.util.spec_from_file_location("t076_pad_mm_stride", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载上游测试：{source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if not torch.npu.is_available():
        raise RuntimeError("当前进程不可见 NPU")
    torch.npu.set_device("npu:0")
    upstream = load_upstream(args.pytorch_root)
    observation: dict[str, object] = {}

    class NpuStride(unittest.TestCase):
        @upstream.inductor_config.patch(
            force_shape_pad=True,
            strict_output_strides=True,
        )
        def test_product_baseline(self):
            def fn(x, y):
                return x @ y

            inputs = [
                torch.randn(3, 5, device="npu", dtype=torch.float32),
                torch.randn(5, 2, device="npu", dtype=torch.float32),
            ]
            expected = fn(*inputs)
            compiled = torch.compile(
                fn,
                options={"npu_backend": "triton_experimental"},
            )
            actual, codes = upstream.run_and_get_code(compiled, *inputs)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual.stride(), expected.stride())

            code_text = "\n\n# ---- graph boundary ----\n\n".join(codes)
            code_path = args.artifact_dir / "generated_code.py"
            code_path.write_text(code_text, encoding="utf-8")
            from torch_npu._inductor.triton_experimental import config as ncfg

            self.assertTrue(ncfg.disable_pad_mm)
            self.assertFalse(upstream.inductor_config.shape_padding)
            observation.update(
                {
                    "input_shapes": [list(value.shape) for value in inputs],
                    "input_strides": [list(value.stride()) for value in inputs],
                    "expected_shape": list(expected.shape),
                    "expected_stride": list(expected.stride()),
                    "actual_stride": list(actual.stride()),
                    "disable_pad_mm": ncfg.disable_pad_mm,
                    "shape_padding_after_backend_load": (
                        upstream.inductor_config.shape_padding
                    ),
                    "correctness": "passed",
                    "stride_preserved": True,
                    "generated_code": str(code_path),
                }
            )

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite([NpuStride("test_product_baseline")])
    )
    from torch_npu.utils._dynamo import _InductorNpuRegistry

    summary = {
        "source_test": "test/inductor/test_pad_mm.py::PadMMTest.test_pad_mm_output_strides_preserved",
        "adapter_deviation": [
            "GPU_TYPE=npu",
            'torch.compile options={"npu_backend":"triton_experimental"}',
            "bypass GPU-only is_big_gpu harness",
        ],
        "product_gate_bypassed": False,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
        "loaded_backend": _InductorNpuRegistry._loaded_backend,
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "observation": observation,
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
    print("T076_NPU_RESULT=" + json.dumps(summary, ensure_ascii=False))
    return 0 if result.wasSuccessful() and result.testsRun == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
