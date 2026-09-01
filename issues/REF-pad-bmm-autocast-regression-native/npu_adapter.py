#!/usr/bin/env python3
"""直接复用上游 autocast/bmm joint-graph regression test body。"""

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
    spec = importlib.util.spec_from_file_location("t076_pad_bmm_autocast", source)
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
    upstream.GPU_TYPE = "npu"

    class Case(unittest.TestCase):
        def test_exact_upstream_body(self):
            upstream.PadMMTest.test_no_autocast_in_pad_bmm_joint_graph_pass(self)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([Case("test_exact_upstream_body")]))
    from torch_npu._inductor.triton_experimental import config as ncfg
    summary = {
        "source_test": "test/inductor/test_pad_mm.py::PadMMTest.test_no_autocast_in_pad_bmm_joint_graph_pass",
        "adapter_deviation": ["module GPU_TYPE=npu", "preselect backend=triton_experimental", "bypass GPU-only big-GPU harness"],
        "preserved_contract": ["exact upstream MaskedMHA body", "B=2,H=32,S=549,D=128", "bfloat16 autocast", "joint pre/post bmm dtype equality", "x1/x2 backward gradients contain no NaN"],
        "product_gate_bypassed": False,
        "disable_pad_mm": ncfg.disable_pad_mm,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "selected_environment": {key: os.environ.get(key) for key in ("ASCEND_RT_VISIBLE_DEVICES", "SET_NPU_DEVICE", "TORCH_COMPILE_DEBUG", "TORCH_COMPILE_DEBUG_DIR", "TORCH_TRACE", "TORCHINDUCTOR_CACHE_DIR")},
    }
    (args.artifact_dir / "adapter_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("T076_NPU_RESULT=" + json.dumps(summary, ensure_ascii=False))
    return 0 if result.wasSuccessful() and result.testsRun == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
