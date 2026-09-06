#!/usr/bin/env python3
"""复用上游 partial reduction reuse 两个负例；只适配设备与测试入口。"""

import argparse
import hashlib
import importlib.util
import inspect
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
        "t078_partial_reuse_negative_upstream", SOURCE
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream.GPU_TYPE = "npu"
    method = inspect.unwrap(
        upstream.TestPatternMatcher.test_unsuccessful_partial_reuse_case0
    )
    observations = []

    class NpuCase(upstream.TestPatternMatcher):
        device_type = "npu"

        def test_small_static(self):
            return method(self, ((4, 8), "npu"))

        def test_dynamic(self):
            return method(self, ("dynamic", "npu"))

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

    counters.clear()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite(
            [NpuCase("test_small_static"), NpuCase("test_dynamic")]
        )
    )
    counter_assertions = [
        item
        for item in observations
        if item["actual"] == 0 and item["expected"] == 0
    ]
    summary = {
        "tracking_mode": "adapter",
        "source_test": (
            f"{SOURCE}::TestPatternMatcher.test_unsuccessful_partial_reuse"
        ),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "direct_blocker": (
            "模块 __main__ 只在 HAS_GPU=true 时调用 run_tests；上游 HAS_GPU "
            "仅覆盖 CUDA/XPU/MTIA，NPU 上退出 0 但执行 0 个测例"
        ),
        "adapter_deviation": [
            "GPU_TYPE=npu",
            "将 case0 decorator 捕获的 cuda device 参数等价替换为 npu",
            "显式运行原始方法的 small-static 与 dynamic 两组参数，绕过 HAS_GPU 启动门",
            "只旁路记录原 counter 断言，不改变测试体、shape、dynamic 标记或判据",
        ],
        "backend": _InductorNpuRegistry._loaded_backend,
        "product_gate_bypassed": False,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "community_assertions_passed": result.wasSuccessful() and not result.skipped,
        "expected_counter_assertions_seen": len(counter_assertions),
        "observations": observations,
        "compatibility_verdict": (
            "community-negative-pass"
            if result.wasSuccessful()
            and result.testsRun == 2
            and len(counter_assertions) == 2
            else "community-negative-fail"
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
        and result.testsRun == 2
        and len(counter_assertions) == 2
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
