#!/usr/bin/env python3
"""验证 T-078 addmm/baddbmm 性能处置后的正式 NPU product gate。"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


# NPU backend 注册是进程级状态，必须先于 torch / torch_npu 导入。
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"

WORK_DIR = Path("/home/z50063656/tmp")
TORCH_NPU_ROOT = Path("/home/z50063656/Pass/src/torch_npu")
EXPECTED_PYTORCH_COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
if Path.cwd().resolve() != WORK_DIR:
    raise RuntimeError("T-078 product gate 验证必须从 /home/z50063656/tmp 启动")

import torch
import torch_npu
from torch._inductor.utils import run_and_get_code


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(TORCH_NPU_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def observe_handlers(hits: dict[str, int]) -> list[dict[str, Any]]:
    from torch._inductor.fx_passes import post_grad

    names = {
        "unfuse_bias_add_to_pointwise",
        "unfuse_bias_baddbmm_to_pointwise",
    }
    selected = []
    seen = set()
    for key, entries in post_grad.pass_patterns[2].patterns.items():
        for entry in entries:
            if id(entry) in seen:
                continue
            seen.add(id(entry))
            handler = getattr(entry, "handler", None)
            name = getattr(handler, "__name__", None)
            if name not in names:
                continue
            gate = entry.extra_check
            if not getattr(gate, "_torch_npu_unfuse_bias_gate_installed", False):
                raise AssertionError(f"{name} 未安装正式 NPU profitability gate")
            if getattr(gate, "_torch_npu_unfuse_bias_handler", None) != name:
                raise AssertionError(f"{name} gate 元数据不一致")

            @functools.wraps(handler)
            def observed(*args, _handler=handler, _name=name, **kwargs):
                hits[_name] = hits.get(_name, 0) + 1
                return _handler(*args, **kwargs)

            entry.handler = observed
            selected.append(
                {
                    "handler": name,
                    "registry_key": [str(key[0]), str(key[1])],
                    "gate_installed": True,
                }
            )
    if {item["handler"] for item in selected} != names:
        raise AssertionError("addmm/baddbmm 目标 entry 不完整")
    return selected


def contains(code: str, operator: str) -> bool:
    return any(
        token in code
        for token in (
            f"torch.ops.aten.{operator}.default(",
            f"extern_kernels.{operator}(",
        )
    )


def execute_case(
    case_id: str,
    function,
    inputs: tuple[torch.Tensor, ...],
    hits: dict[str, int],
    handler: str,
    expected_hit_delta: int,
    required_operator: str,
    forbidden_operator: str,
) -> dict[str, Any]:
    eager = function(*inputs)
    before = hits.get(handler, 0)
    compiled = torch.compile(
        function,
        fullgraph=True,
        options={"npu_backend": "triton_experimental"},
    )
    actual, codes = run_and_get_code(compiled, *inputs)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, eager, atol=1e-4, rtol=1e-4)
    code = "\n\n".join(codes)
    hit_delta = hits.get(handler, 0) - before
    if hit_delta != expected_hit_delta:
        raise AssertionError(
            f"{case_id} handler 命中应为 {expected_hit_delta}，实际为 {hit_delta}"
        )
    if not contains(code, required_operator):
        raise AssertionError(f"{case_id} 未保留所需 {required_operator}")
    if contains(code, forbidden_operator):
        raise AssertionError(f"{case_id} 意外包含 {forbidden_operator}")
    return {
        "case_id": case_id,
        "status": "passed",
        "handler": handler,
        "target_hits": hit_delta,
        "required_operator": required_operator,
        "forbidden_operator": forbidden_operator,
        "max_abs_error": float((actual - eager).abs().max().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if torch.version.git_version != EXPECTED_PYTORCH_COMMIT:
        raise RuntimeError("PyTorch commit 与 T-078 冻结版本不一致")
    if not torch.npu.is_available():
        raise RuntimeError("NPU 不可见")
    torch.npu.set_device("npu:0")

    from torch_npu._inductor.triton_experimental import config as npu_config
    from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu

    register_inductor_npu()
    if _InductorNpuRegistry._loaded_backend != "triton_experimental":
        raise RuntimeError("实际 backend 不是 triton_experimental")
    if not npu_config.disable_unfuse_bias_addmm:
        raise AssertionError("addmm unfuse 产品 disable 未开启")
    if not npu_config.disable_unfuse_baddbmm_non_default_scalars:
        raise AssertionError("baddbmm 非默认标量产品 disable 未开启")

    expected_source = (
        TORCH_NPU_ROOT / "torch_npu/_inductor/triton_experimental"
    ).resolve()
    loaded_modules = {}
    for name in ("config", "fx_passes", "overrides"):
        module = __import__(
            f"torch_npu._inductor.triton_experimental.{name}",
            fromlist=[name],
        )
        module_path = Path(module.__file__).resolve()
        if module_path != expected_source / f"{name}.py":
            raise RuntimeError(f"未加载正式工作树模块：{name}: {module_path}")
        loaded_modules[name] = str(module_path)

    hits: dict[str, int] = {}
    selected = observe_handlers(hits)
    torch.manual_seed(20260906)
    torch.npu.manual_seed_all(20260906)

    add_bias = torch.randn(20, device="npu")
    add_left = torch.randn(10, 15, device="npu")
    add_right = torch.randn(15, 20, device="npu")

    def addmm_gelu(inp, left, right):
        return torch.nn.functional.gelu(torch.ops.aten.addmm(inp, left, right))

    batch_bias = torch.randn(4, 1, 8, device="npu")
    batch_left = torch.randn(4, 6, 5, device="npu")
    batch_right = torch.randn(4, 5, 8, device="npu")

    def baddbmm_gelu(inp, left, right):
        return torch.nn.functional.gelu(
            torch.ops.aten.baddbmm(inp, left, right)
        )

    def baddbmm_alpha_beta(inp, left, right):
        return torch.nn.functional.relu(
            torch.ops.aten.baddbmm(
                inp,
                left,
                right,
                alpha=0.8,
                beta=0.2,
            )
        )

    cases = [
        execute_case(
            "addmm-default-disabled",
            addmm_gelu,
            (add_bias, add_left, add_right),
            hits,
            "unfuse_bias_add_to_pointwise",
            0,
            "addmm",
            "mm",
        ),
        execute_case(
            "baddbmm-default-enabled",
            baddbmm_gelu,
            (batch_bias, batch_left, batch_right),
            hits,
            "unfuse_bias_baddbmm_to_pointwise",
            1,
            "bmm",
            "baddbmm",
        ),
        execute_case(
            "baddbmm-alpha-beta-disabled",
            baddbmm_alpha_beta,
            (batch_bias, batch_left, batch_right),
            hits,
            "unfuse_bias_baddbmm_to_pointwise",
            0,
            "baddbmm",
            "bmm",
        ),
    ]

    source_files = [
        expected_source / "config.py",
        expected_source / "fx_passes.py",
        expected_source / "overrides.py",
    ]
    result = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_id": "T-078",
        "status": "passed",
        "environment": {
            "backend": _InductorNpuRegistry._loaded_backend,
            "device": torch.npu.get_device_name(0),
            "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_commit": torch.version.git_version,
            "torch_npu": torch_npu.__version__,
            "torch_npu_commit": git_output("rev-parse", "HEAD"),
            "cwd": str(Path.cwd()),
            "python": sys.version,
        },
        "product_config": {
            "disable_unfuse_bias_addmm": npu_config.disable_unfuse_bias_addmm,
            "disable_unfuse_baddbmm_non_default_scalars": (
                npu_config.disable_unfuse_baddbmm_non_default_scalars
            ),
        },
        "selected_entries": selected,
        "cases": cases,
        "source_sha256": {
            str(path.relative_to(TORCH_NPU_ROOT)): sha256(path)
            for path in source_files
        },
        "loaded_formal_modules": loaded_modules,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("T078_PRODUCT_GATE=" + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
