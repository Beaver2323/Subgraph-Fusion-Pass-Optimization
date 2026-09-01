#!/usr/bin/env python3
"""pad-mm exclusion community case 的 NPU 产品基线适配器。"""

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
    spec = importlib.util.spec_from_file_location("t076_pad_mm_exclusion", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载上游测试：{source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_selector_signature_compat() -> bool:
    """兼容当前 torch 新参数与 torch_npu selector shim 的签名差异。"""
    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"
    from torch_npu.utils._dynamo import register_inductor_npu

    register_inductor_npu()
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    original = AlgorithmSelectorCache.__call__
    if "best_config_future" in __import__("inspect").signature(original).parameters:
        return False

    def compatible(self, *args, best_config_future=None, **kwargs):
        del best_config_future
        return original(self, *args, **kwargs)

    AlgorithmSelectorCache.__call__ = compatible
    return True


def main() -> int:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if not torch.npu.is_available():
        raise RuntimeError("当前进程不可见 NPU")
    torch.npu.set_device("npu:0")
    upstream = load_upstream(args.pytorch_root)
    selector_signature_compat = install_selector_signature_compat()
    observations: list[dict[str, object]] = []

    class NpuExclusion(unittest.TestCase):
        @upstream.fresh_cache()
        def test_product_baseline(self):
            def graph_plain(a, b):
                return a @ b

            def graph_with_add(a, b):
                return (a + 1) @ b

            size = [61, 61]
            from torch._inductor.fx_passes.pad_mm import get_pad_cache
            from torch_npu._inductor.triton_experimental import config as ncfg

            # 产品 gate 会在 profitability benchmark 前关闭 shape padding；这里沿用
            # 其他 pad-mm 产品基线的稳定短路点，避免把 upstream benchmark mock 传给
            # 当前 torch_npu algorithm-selector shim。
            with unittest.mock.patch(
                "torch._inductor.fx_passes.pad_mm._skip_do_bench_times",
                True,
            ):
                for index, fn in enumerate((graph_plain, graph_with_add), 1):
                    inputs = (
                        torch.rand(size, device="npu"),
                        torch.rand(size, device="npu"),
                    )
                    expected = fn(*inputs)
                    compiled = torch.compile(
                        fn,
                        options={"npu_backend": "triton_experimental"},
                    )
                    actual, codes = upstream.run_and_get_code(compiled, *inputs)
                    torch.testing.assert_close(actual, expected)
                    cache = get_pad_cache().get_local_cache()
                    self.assertEqual(len(cache), 0)
                    code_path = args.artifact_dir / f"generated_code_{index}.py"
                    code_path.write_text("\n\n".join(codes), encoding="utf-8")
                    observations.append(
                        {
                            "graph": fn.__name__,
                            "input_shapes": [list(value.shape) for value in inputs],
                            "pad_cache_entries": len(cache),
                            "correctness": "passed",
                            "generated_code": str(code_path),
                        }
                    )

            self.assertTrue(ncfg.disable_pad_mm)
            self.assertFalse(upstream.inductor_config.shape_padding)

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite([NpuExclusion("test_product_baseline")])
    )
    from torch_npu.utils._dynamo import _InductorNpuRegistry

    summary = {
        "source_test": "test/inductor/test_pad_mm.py::PadMMTest.test_exclude_padding",
        "adapter_deviation": [
            "GPU_TYPE=npu",
            'torch.compile options={"npu_backend":"triton_experimental"}',
            "bypass GPU-only is_big_gpu harness",
            "map GPU cache-positive assertions to NPU product-gate empty-cache assertion",
            "omit inactive profitability benchmark mock after product gate disables shape padding",
        ],
        "product_gate_bypassed": False,
        "selector_signature_compat": selector_signature_compat,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
        "loaded_backend": _InductorNpuRegistry._loaded_backend,
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "disable_pad_mm": True,
        "shape_padding_after_backend_load": False,
        "observations": observations,
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
