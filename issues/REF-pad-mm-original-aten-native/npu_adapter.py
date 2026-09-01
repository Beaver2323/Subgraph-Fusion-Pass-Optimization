#!/usr/bin/env python3
"""original-aten pad-mm community case 的 NPU 产品基线适配器。"""

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
    spec = importlib.util.spec_from_file_location("t076_pad_mm_original", source)
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

    class NpuOriginalAten(unittest.TestCase):
        @upstream.fresh_cache()
        @upstream.inductor_config.patch(
            {
                "triton.unique_kernel_names": "original_aten",
                "max_autotune_gemm_backends": "TRITON,ATEN",
                "force_shape_pad": True,
            }
        )
        def test_product_baseline(self):
            def fn(x, y):
                return x @ y

            inputs = [
                torch.randn(16, 255, device="npu", dtype=torch.float16),
                torch.randn(255, 16, device="npu", dtype=torch.float16),
            ]
            upstream.counters.clear()
            with unittest.mock.patch(
                "torch._inductor.fx_passes.pad_mm._skip_do_bench_times",
                True,
            ):
                expected = fn(*inputs)
                compiled = torch.compile(
                    fn,
                    options={"npu_backend": "triton_experimental"},
                )
                actual, codes = upstream.run_and_get_code(compiled, *inputs)
            torch.testing.assert_close(actual, expected)
            code_text = "\n\n# ---- graph boundary ----\n\n".join(codes)
            code_path = args.artifact_dir / "generated_code.py"
            code_path.write_text(code_text, encoding="utf-8")

            from torch_npu._inductor.triton_experimental import config as ncfg

            target_template_present = "triton_tem_fused_mm" in code_text
            self.assertTrue(ncfg.disable_pad_mm)
            self.assertFalse(upstream.inductor_config.shape_padding)
            self.assertFalse(target_template_present)
            observation.update(
                {
                    "input_shapes": [list(value.shape) for value in inputs],
                    "input_strides": [list(value.stride()) for value in inputs],
                    "dtype": "float16",
                    "disable_pad_mm": ncfg.disable_pad_mm,
                    "shape_padding_after_backend_load": (
                        upstream.inductor_config.shape_padding
                    ),
                    "pattern_matcher_count": upstream.counters["inductor"][
                        "pattern_matcher_count"
                    ],
                    "pattern_matcher_nodes": upstream.counters["inductor"][
                        "pattern_matcher_nodes"
                    ],
                    "target_template_present": target_template_present,
                    "correctness": "passed",
                    "generated_code": str(code_path),
                }
            )

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite([NpuOriginalAten("test_product_baseline")])
    )
    from torch_npu.utils._dynamo import _InductorNpuRegistry

    summary = {
        "source_test": "test/inductor/test_pad_mm.py::PadMMTest.test_original_aten_preserved_pad_mm",
        "adapter_deviation": [
            "GPU_TYPE=npu",
            'torch.compile options={"npu_backend":"triton_experimental"}',
            "bypass GPU-only is_big_gpu harness",
            "expand TRITON-only choices to TRITON,ATEN",
            "map GPU positive template assertion to product-gate negative assertion",
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
