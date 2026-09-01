#!/usr/bin/env python3
"""dynamic-batch pad-bmm community case 的 NPU 产品基线适配器。"""

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
    spec = importlib.util.spec_from_file_location("t076_pad_bmm_dynamic", source)
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
    observation: dict[str, object] = {}

    class Case(unittest.TestCase):
        @upstream.inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON,ATEN", force_shape_pad=True)
        def test_product_baseline(self):
            class Model(torch.nn.Module):
                def forward(self, a, b):
                    return torch.bmm(a, b)

            b, m, k, n = 10, 128, 33, 40
            fn = Model().to("npu")
            inputs = (
                torch.randn(b, m, k, device="npu", dtype=torch.float32),
                torch.randn(b, k, n, device="npu", dtype=torch.float32),
            )
            aligned_k = upstream.get_padded_length(k, upstream.get_alignment_size(inputs[0])) + k
            torch._dynamo.mark_dynamic(inputs[0], 0)
            torch._dynamo.mark_dynamic(inputs[1], 0)
            with unittest.mock.patch("torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True):
                expected = fn(*inputs)
                compiled = torch.compile(fn, options={"npu_backend": "triton_experimental"})
                actual, codes = upstream.run_and_get_code(compiled, *inputs)
            torch.testing.assert_close(actual, expected)
            code_text = "\n\n".join(codes)
            code_path = args.artifact_dir / "generated_code.py"
            code_path.write_text(code_text, encoding="utf-8")
            from torch_npu._inductor.triton_experimental import config as ncfg
            marker = f"K = {aligned_k}"
            self.assertTrue(ncfg.disable_pad_mm)
            self.assertFalse(upstream.inductor_config.shape_padding)
            self.assertNotIn(marker, code_text)
            observation.update({
                "input_shapes": [list(x.shape) for x in inputs],
                "input_strides": [list(x.stride()) for x in inputs],
                "dynamic_dimensions": ["input0.dim0", "input1.dim0"],
                "reference_positive_marker": marker,
                "reference_positive_marker_present": False,
                "disable_pad_mm": ncfg.disable_pad_mm,
                "shape_padding_after_backend_load": upstream.inductor_config.shape_padding,
                "correctness": "passed",
                "generated_code": str(code_path),
            })

    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([Case("test_product_baseline")]))
    summary = {
        "source_test": "test/inductor/test_pad_mm.py::PadMMTest.test_pad_bmm_dyn_b",
        "adapter_deviation": ["GPU_TYPE=npu", "backend=triton_experimental", "bypass GPU-only big-GPU harness", "expand TRITON to TRITON,ATEN", "map padded-K positive to product-gate negative"],
        "product_gate_bypassed": False,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "observation": observation,
        "selected_environment": {key: os.environ.get(key) for key in ("ASCEND_RT_VISIBLE_DEVICES", "SET_NPU_DEVICE", "TORCH_COMPILE_DEBUG", "TORCH_COMPILE_DEBUG_DIR", "TORCH_TRACE", "TORCHINDUCTOR_CACHE_DIR")},
    }
    (args.artifact_dir / "adapter_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("T076_NPU_RESULT=" + json.dumps(summary, ensure_ascii=False))
    return 0 if result.wasSuccessful() and result.testsRun == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
