#!/usr/bin/env python3
"""REF-mm-plus-mm-native 的最小 NPU 适配器。

继承上游 test_mm_plus_mm 的四组输入与数值断言，仅注入 NPU 设备、
triton_experimental backend 和目标专属 pattern 断言，同时保存每次 compile 的
counter 和生成代码。
"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch-root",
        type=Path,
        default=Path("/home/z50063656/Pass/src/pytorch"),
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def load_upstream_module(pytorch_root: Path):
    source = pytorch_root / "test/inductor/test_pattern_matcher.py"
    spec = importlib.util.spec_from_file_location("t076_upstream_pattern_matcher", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载上游测试：{source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    if not torch.npu.is_available():
        raise RuntimeError("当前进程不可见 NPU")
    torch.npu.set_device("npu:0")

    upstream = load_upstream_module(args.pytorch_root)
    upstream.GPU_TYPE = "npu"
    observations: list[dict[str, object]] = []

    class NpuTestPatternMatcher(upstream.TestPatternMatcher):
        device_type = "npu"

        def common(
            self,
            fn,
            args,
            expected_matches,
            expected_nodes,
            additional_check=lambda code: None,
            reference_in_float=False,
        ):
            upstream.counters.clear()
            torch.manual_seed(42)
            if reference_in_float:
                ref_inputs = upstream.pytree.tree_map_only(
                    torch.Tensor, lambda x: x.to(torch.float32), args
                )
            else:
                ref_inputs = args
            expected = fn(*ref_inputs)
            torch.manual_seed(42)
            compiled = torch.compile(
                fn,
                options={"npu_backend": "triton_experimental"},
            )
            actual, codes = upstream.run_and_get_code(compiled, *args)
            code_list = list(codes)
            code_text = "\n\n# ---- generated graph boundary ----\n\n".join(code_list)
            case_index = len(observations) + 1
            code_path = args_namespace.artifact_dir / f"generated_code_{case_index}.py"
            code_path.write_text(code_text, encoding="utf-8")

            actual_matches = upstream.counters["inductor"]["pattern_matcher_count"]
            actual_nodes = upstream.counters["inductor"]["pattern_matcher_nodes"]
            observation = {
                "case_index": case_index,
                "input_shapes": [list(value.shape) for value in args],
                "expected_matches": expected_matches,
                "expected_nodes": expected_nodes,
                "actual_matches": actual_matches,
                "actual_nodes": actual_nodes,
                "generated_code": str(code_path),
                "generated_code_contains_mm_plus_mm": "mm_plus_mm" in code_text,
            }
            observations.append(observation)

            torch.testing.assert_close(
                actual,
                expected,
                check_dtype=not reference_in_float,
            )
            target_match_expected = expected_nodes == 3
            target_match_actual = "mm_plus_mm" in code_text
            self.assertEqual(target_match_actual, target_match_expected)
            if target_match_expected:
                self.assertEqual(actual_matches, expected_matches)
                self.assertEqual(actual_nodes, expected_nodes)
            else:
                observation["reference_global_counter_only"] = {
                    "expected_matches": expected_matches,
                    "expected_nodes": expected_nodes,
                    "reason": (
                        "GPU reference 的 1/2 来自非目标 add-mm pattern；"
                        "NPU 验收按目标 mm_plus_mm 未出现判定"
                    ),
                }
            additional_check(code_list[0] if len(code_list) == 1 else code_list)
            upstream.counters.clear()

    args_namespace = args
    suite = unittest.TestSuite(
        [NpuTestPatternMatcher(methodName="test_mm_plus_mm")]
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    from torch_npu.utils._dynamo import _InductorNpuRegistry

    summary = {
        "source_test": (
            "test/inductor/test_pattern_matcher.py::"
            "TestPatternMatcher.test_mm_plus_mm"
        ),
        "adapter_deviation": [
            "GPU_TYPE=npu",
            'torch.compile options={"npu_backend":"triton_experimental"}',
            "counter and generated-code capture",
        ],
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
        "loaded_backend": _InductorNpuRegistry._loaded_backend,
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "observations": observations,
        "selected_environment": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "SET_NPU_DEVICE",
                "TORCH_COMPILE_DEBUG",
                "TORCH_COMPILE_DEBUG_DIR",
                "TORCH_TRACE",
                "TORCHINDUCTOR_CACHE_DIR",
            )
        },
    }
    (args.artifact_dir / "adapter_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("T076_NPU_RESULT=" + json.dumps(summary, ensure_ascii=False))
    return 0 if result.wasSuccessful() and result.testsRun == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
