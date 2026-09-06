#!/usr/bin/env python3
"""复用上游 addcdiv codegen 方法；只适配设备/入口和测试诊断插桩。"""

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
import torch_npu.testing
from torch._dynamo.utils import counters
from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == "triton_experimental"
    assert torch.version.git_version == "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
    torch.npu.set_device(0)
    source = Path(
        "/home/z50063656/Pass/src/pytorch/test/inductor/test_torchinductor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "t078_codegen_upstream_torchinductor", source
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream.GPU_TYPE = "npu"
    method = inspect.unwrap(
        upstream.CommonTemplate.test_addcdiv_fma_uses_fma_and_div_rn
    )
    observations = []

    class NpuCase(upstream.TestCase):
        device = "npu"
        test_addcdiv_codegen = method

        @classmethod
        def setUpClass(cls):
            super().setUpClass()
            cls._stack.enter_context(
                upstream.config.patch(
                    {
                        "test_configs.runtime_triton_dtype_assert": False,
                        "test_configs.runtime_triton_shape_assert": False,
                    }
                )
            )

        def assertEqual(self, actual, expected, *positional, **kwargs):
            if isinstance(actual, int) and isinstance(expected, int):
                observations.append(
                    {
                        "assertion": "counter-equal",
                        "actual": actual,
                        "expected": expected,
                    }
                )
            return super().assertEqual(actual, expected, *positional, **kwargs)

        def assertIn(self, member, container, *positional, **kwargs):
            if member in ("tl.fma", "triton.language.div_rn"):
                observations.append(
                    {
                        "assertion": "code-contains",
                        "needle": member,
                        "present": member in container,
                    }
                )
            return super().assertIn(member, container, *positional, **kwargs)

    counters.clear()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite([NpuCase("test_addcdiv_codegen")])
    )
    summary = {
        "tracking_mode": "adapter",
        "source_test": (
            f"{source}::CommonTemplate.test_addcdiv_fma_uses_fma_and_div_rn"
        ),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "adapter_deviation": [
            "GPU_TYPE=npu",
            "缺少 GPUTests 时直接绑定 CommonTemplate 原方法，去除 CUDA-only 测试包装",
            "关闭 CUDA 测试基类额外开启的 runtime Triton dtype/shape 诊断插桩",
            "只记录原 counter/code contains 断言，不改变断言判据",
        ],
        "backend": _InductorNpuRegistry._loaded_backend,
        "product_gate_bypassed": False,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "community_assertions_passed": result.wasSuccessful() and not result.skipped,
        "addcdiv_fma_fused": counters["inductor"]["addcdiv_fma_fused"],
        "observations": observations,
        "compatibility_verdict": (
            "community-codegen-pass"
            if result.wasSuccessful()
            else "community-codegen-fail"
        ),
    }
    (args.artifact_dir / "adapter_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if result.wasSuccessful() and result.testsRun == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
