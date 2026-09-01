#!/usr/bin/env python3
"""pad-addmm bias broadcast community case 的 NPU 产品基线适配器。"""

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


def load_upstream(root: Path):
    source = root / "test/inductor/test_pad_mm.py"
    spec = importlib.util.spec_from_file_location("t076_pad_addmm_bias", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-root", type=Path, default=Path("/home/z50063656/Pass/src/pytorch"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if not torch.npu.is_available():
        raise RuntimeError("当前进程不可见 NPU")
    torch.npu.set_device("npu:0")
    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"
    from torch_npu.utils._dynamo import register_inductor_npu
    register_inductor_npu()
    upstream = load_upstream(args.pytorch_root)
    observations = []

    class Case(unittest.TestCase):
        @upstream.fresh_cache()
        @upstream.inductor_config.patch(force_shape_pad=True, max_autotune_gemm_backends="TRITON,ATEN")
        def test_product_baseline(self):
            def fn(bias, x, y):
                return torch.ops.aten.addmm(bias, x, y)
            compiled = torch.compile(fn, options={"npu_backend": "triton_experimental"})
            bias_shapes = [(a, b) for a in (1, 4) for b in (1, 6)] + [(1,), (6,)]
            for index, bias_shape in enumerate(bias_shapes, 1):
                inputs = (torch.rand(bias_shape, device="npu"), torch.rand((4, 5), device="npu"), torch.rand((5, 6), device="npu"))
                expected = fn(*inputs)
                actual, codes = upstream.run_and_get_code(compiled, *inputs)
                torch.testing.assert_close(actual, expected)
                code_path = args.artifact_dir / f"generated_code_{index}.py"
                code_path.write_text("\n\n".join(codes), encoding="utf-8")
                observations.append({"bias_shape": list(bias_shape), "input_shapes": [list(x.shape) for x in inputs], "correctness": "passed", "generated_code": str(code_path)})
            from torch_npu._inductor.triton_experimental import config as ncfg
            self.assertTrue(ncfg.disable_pad_mm)
            self.assertFalse(upstream.inductor_config.shape_padding)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([Case("test_product_baseline")]))
    summary = {"source_test": "test/inductor/test_pad_mm.py::PadMMTest.test_pad_addmm_2d_bias", "adapter_deviation": ["GPU_TYPE=npu", "backend=triton_experimental", "bypass GPU-only harness", "allow ATEN choice"], "product_gate_bypassed": False, "disable_pad_mm": True, "shape_padding_after_backend_load": False, "logical_cases": len(observations), "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "successful": result.wasSuccessful(), "torch_version": torch.__version__, "torch_npu_version": torch_npu.__version__, "device": torch.npu.get_device_name(0), "observations": observations, "selected_environment": {key: os.environ.get(key) for key in ("ASCEND_RT_VISIBLE_DEVICES", "SET_NPU_DEVICE", "TORCH_COMPILE_DEBUG", "TORCH_COMPILE_DEBUG_DIR", "TORCH_TRACE", "TORCHINDUCTOR_CACHE_DIR")}}
    (args.artifact_dir / "adapter_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("T076_NPU_RESULT=" + json.dumps(summary, ensure_ascii=False))
    return 0 if result.wasSuccessful() and result.testsRun == 1 and len(observations) == 6 else 1


if __name__ == "__main__":
    sys.exit(main())
