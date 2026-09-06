#!/usr/bin/env python3
"""复用上游 baddbmm bias unfuse 核心方法；只适配设备与入口。"""

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
METHOD = "test_unfuse_broadcast_bias_baddbmm"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--normalize-codegen-symbols", action="store_true")
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == "triton_experimental"
    assert torch.version.git_version == "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
    torch.npu.set_device(0)
    spec = importlib.util.spec_from_file_location(
        "t078_unfuse_baddbmm_core_upstream", SOURCE
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream.GPU_TYPE = "npu"
    codegen_observations = []
    if args.normalize_codegen_symbols:
        original_filecheck = upstream.FileCheck

        class NpuCodegenFileCheck:
            """Keep original checks after canonicalizing NPU direct-op spellings."""

            def __init__(self):
                self._inner = original_filecheck()

            def check(self, text):
                self._inner.check(text)
                return self

            def check_not(self, text):
                self._inner.check_not(text)
                return self

            def run(self, code):
                codegen_observations.append(
                    {
                        "direct_baddbmm": "torch.ops.aten.baddbmm.default(" in code,
                        "extern_baddbmm": "extern_kernels.baddbmm(" in code,
                        "direct_bmm": "torch.ops.aten.bmm.default(" in code,
                        "extern_bmm": "extern_kernels.bmm(" in code,
                    }
                )
                normalized = code.replace(
                    "torch.ops.aten.baddbmm.default(", "extern_kernels.baddbmm("
                ).replace("torch.ops.aten.bmm.default(", "extern_kernels.bmm(")
                return self._inner.run(normalized)

        upstream.FileCheck = NpuCodegenFileCheck

    class NpuCase(upstream.TestPatternMatcher):
        device_type = "npu"

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite([NpuCase(METHOD)])
    )
    summary = {
        "tracking_mode": (
            "backend-codegen-adapter"
            if args.normalize_codegen_symbols
            else "adapter"
        ),
        "source_test": f"{SOURCE}::TestPatternMatcher.{METHOD}",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "adapter_deviation": [
            "GPU_TYPE=npu",
            "显式构造原方法，绕过不识别 NPU 的 HAS_GPU 启动门",
            "不改变 broadcast/expand、shape、数值断言或 baddbmm/bmm FileCheck",
            *(
                [
                    "仅在 FileCheck 输入副本中把 torch.ops.aten.{baddbmm,bmm}.default "
                    "规范化为等价 extern_kernels needle；原生成代码原样保留",
                ]
                if args.normalize_codegen_symbols
                else []
            ),
        ],
        "backend": _InductorNpuRegistry._loaded_backend,
        "product_gate_bypassed": False,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "community_assertions_passed": result.wasSuccessful() and not result.skipped,
        "codegen_observations": codegen_observations,
        "compatibility_verdict": (
            "community-core-pass"
            if result.wasSuccessful() and result.testsRun == 1
            else "community-core-fail"
        ),
    }
    (args.artifact_dir / "adapter_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if result.wasSuccessful() and result.testsRun == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
