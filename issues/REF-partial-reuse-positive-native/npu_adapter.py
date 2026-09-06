#!/usr/bin/env python3
"""复用上游 partial reduction reuse 三个正例；只适配设备与测试入口。"""

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
from torch._dynamo.utils import counters
from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu


SOURCE = Path(
    "/home/z50063656/Pass/src/pytorch/test/inductor/test_pattern_matcher.py"
)
METHODS = [
    "test_successful_partial_reuse_case0",
    "test_successful_partial_reuse_case1",
    "test_successful_partial_reuse_case2",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source-fix", action="store_true")
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == "triton_experimental"
    assert torch.version.git_version == "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
    torch.npu.set_device(0)
    from torch_npu._inductor.triton_experimental import npu_triton_helpers

    helper_path = Path(npu_triton_helpers.__file__).resolve()
    formal_helper_path = Path(
        "/home/z50063656/Pass/src/torch_npu/torch_npu/_inductor/"
        "triton_experimental/npu_triton_helpers.py"
    ).resolve()
    if args.source_fix and helper_path != formal_helper_path:
        raise RuntimeError(
            f"正式修复未从 torch_npu 工作树加载: {helper_path} != {formal_helper_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "t078_partial_reuse_positive_upstream", SOURCE
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream.GPU_TYPE = "npu"

    observations = []

    class NpuCase(upstream.TestPatternMatcher):
        device_type = "npu"

        def assertEqual(self, actual, expected, *positional, **kwargs):
            if isinstance(actual, int) and isinstance(expected, int):
                observations.append(
                    {
                        "test": self._testMethodName,
                        "assertion": "integer-equal",
                        "actual": actual,
                        "expected": expected,
                    }
                )
            return super().assertEqual(actual, expected, *positional, **kwargs)

    suite = unittest.TestSuite([NpuCase(method) for method in METHODS])
    counters.clear()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    counter_assertions = [
        item
        for item in observations
        if item["actual"] == 1 and item["expected"] == 1
    ]
    summary = {
        "tracking_mode": (
            "source-fix-verification" if args.source_fix else "adapter"
        ),
        "source_test": (
            f"{SOURCE}::TestPatternMatcher.test_successful_partial_reuse"
        ),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "direct_blocker": (
            "模块 __main__ 只在 HAS_GPU=true 时调用 run_tests；上游 HAS_GPU "
            "仅覆盖 CUDA/XPU/MTIA，NPU 上退出 0 但执行 0 个测例"
        ),
        "adapter_deviation": [
            "GPU_TYPE=npu",
            "显式构造原类生成的 case0/case1/case2 unittest，绕过 HAS_GPU 启动门",
            "只旁路记录原整数断言，不改变测试体、shape、threshold 或判据",
        ],
        "backend": _InductorNpuRegistry._loaded_backend,
        "npu_triton_helpers_file": str(helper_path),
        "product_gate_bypassed": False,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "community_assertions_passed": result.wasSuccessful() and not result.skipped,
        "expected_counter_assertions_seen": len(counter_assertions),
        "observations": observations,
        "compatibility_verdict": (
            "community-positive-pass"
            if result.wasSuccessful()
            and result.testsRun == 3
            and len(counter_assertions) == 3
            else "community-positive-fail"
        ),
    }
    (args.artifact_dir / "adapter_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return (
        0
        if result.wasSuccessful()
        and result.testsRun == 3
        and len(counter_assertions) == 3
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
