#!/usr/bin/env python3
"""验证 T-078 min2 修复的 NaN 传播近邻语义。"""

import argparse
import json
import os
from pathlib import Path


if os.environ.get("TORCHINDUCTOR_NPU_BACKEND") != "triton_experimental":
    raise RuntimeError("必须在导入 torch 前指定 triton_experimental")
if Path.cwd().resolve() != Path("/home/z50063656/tmp"):
    raise RuntimeError("必须从 /home/z50063656/tmp 启动")

import torch
import torch_npu
from torch._inductor.utils import run_and_get_code
from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("nan-propagation",), required=True)
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
    if helper_path != formal_helper_path:
        raise RuntimeError(
            f"正式修复未从 torch_npu 工作树加载: {helper_path} != {formal_helper_path}"
        )

    def reduction(x):
        return torch.amin(x, dim=0)

    torch.manual_seed(42)
    x = torch.randn(64, 64, device="npu", dtype=torch.float32)
    x[3, 7] = float("nan")
    x[12, 31] = float("nan")
    expected = reduction(x)
    actual, codes = run_and_get_code(torch.compile(reduction), x)
    code = codes[0] if len(codes) == 1 else "\n".join(codes)

    expected_nan = torch.isnan(expected)
    actual_nan = torch.isnan(actual)
    nan_masks_equal = torch.equal(expected_nan, actual_nan)
    finite_mask = ~expected_nan
    finite_equal = torch.equal(expected[finite_mask], actual[finite_mask])
    helper_used = "triton_helpers.min2" in code
    passed = nan_masks_equal and finite_equal and helper_used
    summary = {
        "tracking_mode": "source-fix-neighbor-verification",
        "case": args.case,
        "backend": _InductorNpuRegistry._loaded_backend,
        "npu_triton_helpers_file": str(helper_path),
        "shape": [64, 64],
        "dtype": "float32",
        "nan_columns": [7, 31],
        "expected_nan_count": int(expected_nan.sum().item()),
        "actual_nan_count": int(actual_nan.sum().item()),
        "nan_masks_equal": nan_masks_equal,
        "finite_values_bitwise_equal": finite_equal,
        "generated_code_uses_triton_helpers_min2": helper_used,
        "passed": passed,
    }
    (args.artifact_dir / "neighbor_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.artifact_dir / "generated_code.py").write_text(code, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
