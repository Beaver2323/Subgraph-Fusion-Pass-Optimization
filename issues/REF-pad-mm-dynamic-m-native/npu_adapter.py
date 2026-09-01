#!/usr/bin/env python3
"""REF-pad-mm-dynamic-m-native 的产品基线 NPU 适配器。

复用上游 test_pad_mm_dyn_m 的 Model、shape、stride、dynamic-M 和 correctness，
注入 NPU device/backend，绕过 GPU-only big-GPU harness，把 GPU-only 的 TRITON 候选限制
扩为 TRITON,ATEN，并把 GPU 正向 padded-K 断言映射为 NPU 产品默认 disable_pad_mm
gate 的负向断言。不得在本适配器中关闭该 gate。
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
    source = pytorch_root / "test/inductor/test_pad_mm.py"
    spec = importlib.util.spec_from_file_location("t076_upstream_pad_mm", source)
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
    observations: list[dict[str, object]] = []

    class NpuPadMMDynamicM(unittest.TestCase):
        @upstream.inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON,ATEN",
            force_shape_pad=True,
        )
        def test_product_baseline(self):
            m = 40
            k1 = 581
            k2 = 49
            n = 30

            class Model(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.w = upstream.rand_strided(
                        (k2, n),
                        (1, k2),
                        device="npu",
                        dtype=torch.float32,
                    )

                def forward(self, a):
                    a1 = torch.narrow(a, 1, 0, k2)
                    return torch.mm(a1, self.w)

            fn = Model().to("npu")
            a = upstream.rand_strided(
                (m, k1),
                (k1, 1),
                device="npu",
                dtype=torch.float32,
            )
            aligned_k = (
                upstream.get_padded_length(
                    k2,
                    upstream.get_alignment_size(a),
                )
                + k2
            )
            torch._dynamo.mark_dynamic(a, 0)
            upstream.counters.clear()
            with unittest.mock.patch(
                "torch._inductor.fx_passes.pad_mm._skip_do_bench_times",
                True,
            ):
                expected = fn(a)
                compiled_fn = torch.compile(
                    fn,
                    options={"npu_backend": "triton_experimental"},
                )
                actual, codes = upstream.run_and_get_code(compiled_fn, a)

            torch.testing.assert_close(actual, expected)
            code_text = "\n\n# ---- generated graph boundary ----\n\n".join(codes)
            code_path = args.artifact_dir / "generated_code.py"
            code_path.write_text(code_text, encoding="utf-8")

            from torch_npu._inductor.triton_experimental import config as npu_config

            product_gate_enabled = npu_config.disable_pad_mm
            shape_padding_after_backend_load = upstream.inductor_config.shape_padding
            gpu_positive_marker = f"K = {aligned_k}"
            marker_present = gpu_positive_marker in code_text

            self.assertTrue(product_gate_enabled)
            self.assertFalse(shape_padding_after_backend_load)
            self.assertFalse(marker_present)
            observations.append(
                {
                    "input_shape": list(a.shape),
                    "input_stride": list(a.stride()),
                    "weight_shape": list(fn.w.shape),
                    "weight_stride": list(fn.w.stride()),
                    "dynamic_dimension": 0,
                    "reference_padded_k": aligned_k,
                    "reference_positive_marker": gpu_positive_marker,
                    "reference_positive_marker_present": marker_present,
                    "disable_pad_mm": product_gate_enabled,
                    "shape_padding_after_backend_load": (
                        shape_padding_after_backend_load
                    ),
                    "pattern_matcher_count": upstream.counters["inductor"][
                        "pattern_matcher_count"
                    ],
                    "pattern_matcher_nodes": upstream.counters["inductor"][
                        "pattern_matcher_nodes"
                    ],
                    "generated_code": str(code_path),
                    "correctness": "passed",
                }
            )

    suite = unittest.TestSuite([NpuPadMMDynamicM("test_product_baseline")])
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    from torch_npu.utils._dynamo import _InductorNpuRegistry

    summary = {
        "source_test": "test/inductor/test_pad_mm.py::PadMMTest.test_pad_mm_dyn_m",
        "adapter_deviation": [
            "GPU_TYPE=npu",
            'torch.compile options={"npu_backend":"triton_experimental"}',
            "bypass GPU-only is_big_gpu harness",
            "expand GPU-only max_autotune_gemm_backends=TRITON to TRITON,ATEN",
            "map GPU padded-K positive assertion to NPU product-gate negative assertion",
        ],
        "product_gate_bypassed": False,
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
