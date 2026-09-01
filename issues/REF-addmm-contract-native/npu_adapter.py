#!/usr/bin/env python3
"""直接复用上游 post-grad addmm contract test body。"""

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
    source = root / "test/inductor/test_pattern_matcher.py"
    spec = importlib.util.spec_from_file_location("t076_post_grad_addmm", source)
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
    counter_observations = []

    class Case(unittest.TestCase):
        def assertEqual(self, first, second, msg=None):
            if isinstance(first, int) and isinstance(second, int):
                counter_observations.append({"actual": first, "expected": second})
                return
            super().assertEqual(first, second, msg)

        def test_exact_upstream_body(self):
            upstream.TestPatternMatcher.test_addmm(self)
            mismatches = [item for item in counter_observations if item["actual"] != item["expected"]]
            if mismatches:
                self.fail(f"目标 counter 合同不一致: {mismatches}")

    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([Case("test_exact_upstream_body")]))
    summary = {"source_test": "test/inductor/test_pattern_matcher.py::TestPatternMatcher.test_addmm", "adapter_deviation": ["module GPU_TYPE=npu", "preselect backend=triton_experimental", "bypass GPU-only suite entry", "collect all upstream counter assertions before reporting mismatch"], "preserved_contract": ["five exact upstream logical cases", "two output operand orders", "target counter 2/4 for matrix/vector bias", "target counter 0/0 for non-expandable/batched/Python scalar", "eager/compiled correctness"], "product_gate_bypassed": False, "logical_cases": len(counter_observations) // 2, "counter_observations": counter_observations, "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "successful": result.wasSuccessful(), "torch_version": torch.__version__, "torch_npu_version": torch_npu.__version__, "device": torch.npu.get_device_name(0), "selected_environment": {key: os.environ.get(key) for key in ("ASCEND_RT_VISIBLE_DEVICES", "SET_NPU_DEVICE", "TORCH_COMPILE_DEBUG", "TORCH_COMPILE_DEBUG_DIR", "TORCH_TRACE", "TORCHINDUCTOR_CACHE_DIR")}}
    (args.artifact_dir / "adapter_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("T076_NPU_RESULT=" + json.dumps(summary, ensure_ascii=False))
    return 0 if result.wasSuccessful() and result.testsRun == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
