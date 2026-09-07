#!/usr/bin/env python3
"""T-079 社区方法的最小 NPU 入口适配与结构化证据采集。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import unittest


WORK = Path("/home/z50063656/tmp")
SOURCE = Path(
    "/home/z50063656/Pass/src/pytorch/test/inductor/test_pattern_matcher.py"
)
PYTORCH_COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
CASES = {
    "REF-bmm-to-mm-native": {
        "method": "test_bmm_to_mm",
        "unit": "AU-joint-graph-bmm-to-mm",
        "variants": ["batch-one-positive", "multi-batch-negative"],
    },
    "REF-cat-slice-cat-native": {
        "method": "test_cat_slice_cat_cuda",
        "unit": "AU-post-grad-cat-slice-cat",
        "variants": [
            "slice-within-first-input-positive",
            "slice-end-exceeds-first-input-fallback",
            "negative-slice-end-fallback",
        ],
    },
    "REF-splitwithsizes-cat-native": {
        "method": "test_splitwithsizes_cat",
        "unit": "AU-post-grad-splitwithsizes-cat-replace",
        "variants": [
            "complete-same-order-positive",
            "missing-getitem-negative",
            "different-dim-negative",
            "reordered-getitems-negative",
        ],
    },
    "REF-cat-splitwithsizes-native": {
        "method": "test_cat_splitwithsizes",
        "unit": "AU-post-grad-cat-splitwithsizes-replace",
        "variants": [
            "same-boundaries-positive",
            "cat-multi-user-negative",
            "different-dim-negative",
            "different-count-negative",
            "different-boundaries-negative",
        ],
    },
}


def copy_debug_artifacts(debug_root: Path, artifact_dir: Path) -> list[str]:
    """复制最靠后的可读 FX/IR/code；原始 debug 树仍完整保留。"""
    copied = []
    candidates = {
        "fx_before.txt": ("fx_graph_readable.py", "fx_graph_runnable.py"),
        "fx_after.txt": ("fx_graph_transformed.py",),
        "ir_pre_fusion.txt": ("ir_pre_fusion.txt",),
        "ir_post_fusion.txt": ("ir_post_fusion.txt",),
        "generated_code.py": ("output_code.py",),
    }
    all_files = [path for path in debug_root.rglob("*") if path.is_file()]
    for destination, names in candidates.items():
        matches = [path for path in all_files if path.name in names]
        if not matches:
            continue
        source = max(matches, key=lambda path: path.stat().st_mtime_ns)
        shutil.copyfile(source, artifact_dir / destination)
        copied.append(destination)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", choices=sorted(CASES), required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    if Path.cwd().resolve() != WORK:
        raise RuntimeError("必须从 /home/z50063656/tmp 启动")
    if os.environ.get("TORCHINDUCTOR_NPU_BACKEND") != "triton_experimental":
        raise RuntimeError("必须在导入 torch 前指定 triton_experimental")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    # 后端注册具有进程级生命周期，故 import 必须留在环境检查之后。
    import torch
    import torch_npu
    from torch_npu.utils._dynamo import (
        _InductorNpuRegistry,
        register_inductor_npu,
    )

    register_inductor_npu()
    if _InductorNpuRegistry._loaded_backend != "triton_experimental":
        raise RuntimeError("实际加载的 NPU backend 不是 triton_experimental")
    if torch.version.git_version != PYTORCH_COMMIT:
        raise RuntimeError(
            f"PyTorch commit 不匹配: {torch.version.git_version} != {PYTORCH_COMMIT}"
        )
    torch.npu.set_device(0)

    spec = importlib.util.spec_from_file_location("t079_upstream_pattern_matcher", SOURCE)
    upstream = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(upstream)
    upstream.GPU_TYPE = "npu"
    # 只恢复 GPU 类测试中的设备能力分支；bmm 方法原 FileCheck 由此继续要求
    # batch=1 为 mm、batch>1 为 bmm，不删除或替换断言。
    upstream.HAS_GPU = True

    case = CASES[args.case_id]
    integer_assertions = []

    class NpuCase(upstream.TestPatternMatcher):
        device_type = "npu"

        def assertEqual(self, actual, expected, *positional, **kwargs):
            if isinstance(actual, int) and isinstance(expected, int):
                integer_assertions.append(
                    {"actual": actual, "expected": expected, "equal": actual == expected}
                )
            return super().assertEqual(actual, expected, *positional, **kwargs)

    with upstream.inductor_config.patch(
        {
            "test_configs.runtime_triton_dtype_assert": False,
            "test_configs.runtime_triton_shape_assert": False,
        }
    ):
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.TestSuite([NpuCase(case["method"])])
        )
    copied = copy_debug_artifacts(
        Path(os.environ.get("TORCH_COMPILE_DEBUG_DIR", args.artifact_dir / "debug")),
        args.artifact_dir,
    )
    summary = {
        "schema_version": "1.0",
        "task_id": "T-079",
        "case_id": args.case_id,
        "acceptance_unit_id": case["unit"],
        "variant_ids": case["variants"],
        "source_test": f"{SOURCE}::TestPatternMatcher.{case['method']}",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "torch_commit": torch.version.git_version,
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "backend": _InductorNpuRegistry._loaded_backend,
        "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "direct_blocker": (
            "上游 __main__ 仅在 HAS_GPU=true 时调用 run_tests；NPU 原生执行返回 0，"
            "但实际执行 0 tests"
        ),
        "adapter_deviation": [
            "GPU_TYPE=npu",
            "显式实例化原 TestPatternMatcher 方法，绕过 __main__ 的 HAS_GPU 启动门",
            "令原方法中的 GPU 能力分支在 NPU 上执行，保留原 FileCheck/counter 判据",
            "关闭仅用于 CUDA 测试诊断且 triton-ascend 不兼容的 runtime dtype/shape 插桩",
        ],
        "preserved_contract": [
            "原社区方法体、输入 shape/dtype、正负分支和 torch.compile 调用不变",
            "原 torch.testing.assert_close、counter 与 FileCheck 断言不变",
        ],
        "product_gate_bypassed": False,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "community_assertions_passed": result.wasSuccessful() and not result.skipped,
        "integer_assertions": integer_assertions,
        "copied_debug_artifacts": copied,
        "raw_debug_dir": os.environ.get("TORCH_COMPILE_DEBUG_DIR"),
        "compatibility_verdict": (
            "community-functional-pass"
            if result.wasSuccessful() and result.testsRun == 1 and not result.skipped
            else "community-functional-fail"
        ),
    }
    (args.artifact_dir / "adapter_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("T079_NPU_RESULT=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["compatibility_verdict"] == "community-functional-pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
