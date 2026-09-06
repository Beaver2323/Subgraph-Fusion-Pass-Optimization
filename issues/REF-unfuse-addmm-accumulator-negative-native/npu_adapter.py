#!/usr/bin/env python3
"""复用上游 computed-accumulator addmm 负例；只适配设备与入口。"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest


if os.environ.get("TORCHINDUCTOR_NPU_BACKEND") != "triton_experimental":
    raise RuntimeError("必须在导入 torch 前指定 triton_experimental")
if Path.cwd().resolve() != Path("/home/z50063656/tmp"):
    raise RuntimeError("必须从 /home/z50063656/tmp 启动")

import torch
import torch_npu
from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu


SOURCE = Path(
    "/home/z50063656/Pass/src/pytorch/test/inductor/test_pattern_matcher.py"
)
METHOD = "test_preserve_accumulator_addmm_with_pointwise"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == "triton_experimental"
    assert torch.version.git_version == "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
    torch.npu.set_device(0)

    spec = importlib.util.spec_from_file_location(
        "t078_unfuse_addmm_accumulator_upstream", SOURCE
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream.GPU_TYPE = "npu"

    class NpuCase(upstream.TestPatternMatcher):
        device_type = "npu"

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite([NpuCase(METHOD)])
    )
    summary = {
        "tracking_mode": "adapter",
        "source_test": f"{SOURCE}::TestPatternMatcher.{METHOD}",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "adapter_deviation": [
            "GPU_TYPE=npu",
            "显式构造原方法，绕过不识别 NPU 的 HAS_GPU 启动门",
            "不改变 input sin、shape、数值断言或 extern addmm FileCheck",
        ],
        "backend": _InductorNpuRegistry._loaded_backend,
        "product_gate_bypassed": False,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "community_assertions_passed": result.wasSuccessful() and not result.skipped,
        "compatibility_verdict": (
            "community-negative-pass"
            if result.wasSuccessful() and result.testsRun == 1
            else "community-negative-fail"
        ),
    }
    (args.artifact_dir / "adapter_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if result.wasSuccessful() and result.testsRun == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
