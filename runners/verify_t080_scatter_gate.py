#!/usr/bin/env python3
"""验证 T-080 Scatter 产品门禁的默认关闭与可逆重开路径。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import functools
import json
import os
from pathlib import Path


WORK = Path("/home/z50063656/tmp")
SOURCE_ROOT = Path("/home/z50063656/Pass/src/torch_npu")
PYTORCH_COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"


def count_codegen(root: Path) -> dict[str, object]:
    files = sorted(root.rglob("output_code.py"))
    text = "\n".join(path.read_text(errors="replace") for path in files)
    return {
        "files": [str(path) for path in files],
        "triton_kernels": text.count("async_compile.triton("),
        "where_tokens": text.count("where_self"),
        "scatter_tokens": text.count("scatter"),
    }


def find_entry():
    from torch._inductor.fx_passes import joint_graph

    seen = set()
    for entries in joint_graph.patterns.patterns.values():
        for entry in entries:
            if id(entry) in seen:
                continue
            seen.add(id(entry))
            handler = getattr(entry, "handler", None)
            if getattr(handler, "__name__", None) == "scatter_upon_const_tensor":
                return entry
    raise RuntimeError("未找到 scatter_upon_const_tensor pattern entry")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("default-disabled", "gate-disabled"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if Path.cwd().resolve() != WORK:
        raise RuntimeError("必须从 /home/z50063656/tmp 启动")
    if os.environ.get("TORCHINDUCTOR_NPU_BACKEND") != "triton_experimental":
        raise RuntimeError("必须在导入 torch 前指定 triton_experimental")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    debug_root = args.output.parent / "debug"
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(debug_root)

    import torch
    import torch_npu
    from torch._inductor import metrics
    from torch_npu._inductor.triton_experimental import config as npu_config
    from torch_npu.utils._dynamo import (
        _InductorNpuRegistry,
        register_inductor_npu,
    )

    register_inductor_npu()
    if _InductorNpuRegistry._loaded_backend != "triton_experimental":
        raise RuntimeError("实际 backend 不是 triton_experimental")
    if torch.version.git_version != PYTORCH_COMMIT:
        raise RuntimeError("PyTorch commit 不匹配")
    if SOURCE_ROOT not in Path(npu_config.__file__).resolve().parents:
        raise RuntimeError(f"未加载源码修复模块：{npu_config.__file__}")

    torch.npu.set_device(0)
    entry = find_entry()
    if not getattr(
        entry.extra_check,
        "_torch_npu_scatter_upon_const_gate_installed",
        False,
    ):
        raise RuntimeError("scatter_upon_const_tensor 产品门禁未安装")

    hits = {"total": 0}
    original_handler = entry.handler

    @functools.wraps(original_handler)
    def observed(*positional, **kwargs):
        hits["total"] += 1
        return original_handler(*positional, **kwargs)

    entry.handler = observed

    def fn(index):
        output = torch.full((2, 1024, 2048), 3.14, device="npu")
        return output.scatter(2, index, 2.718)

    torch.manual_seed(20260907)
    torch.npu.manual_seed_all(20260907)
    index = torch.randint(
        0, 2048, (2, 1024, 1), dtype=torch.int64, device="npu"
    )
    expected = fn(index)
    gate_context = (
        npu_config.patch(disable_scatter_upon_const_tensor=False)
        if args.mode == "gate-disabled"
        else nullcontext()
    )
    metrics.reset()
    with gate_context:
        compiled = torch.compile(
            fn,
            fullgraph=True,
            options={"npu_backend": "triton_experimental"},
        )
        actual = compiled(index)
        torch.npu.synchronize()
    torch.testing.assert_close(actual, expected)

    metric_hits = metrics.num_matches_for_scatter_upon_const_tensor
    codegen = count_codegen(debug_root)
    if args.mode == "default-disabled":
        verdict = hits["total"] == 0 and metric_hits == 0
    else:
        verdict = hits["total"] >= 1 and metric_hits >= 1

    result = {
        "schema_version": "1.0",
        "task_id": "T-080",
        "acceptance_unit_id": "AU-joint-graph-scatter-upon-const-tensor",
        "mode": args.mode,
        "backend": _InductorNpuRegistry._loaded_backend,
        "torch_commit": torch.version.git_version,
        "torch_npu_version": torch_npu.__version__,
        "config_source": npu_config.__file__,
        "disable_scatter_upon_const_tensor_default": (
            npu_config.disable_scatter_upon_const_tensor
        ),
        "disable_scatter_upon_const_tensor_during_compile": (
            args.mode == "default-disabled"
        ),
        "gate_installed": True,
        "handler_hits": hits["total"],
        "metric_hits": metric_hits,
        "correctness": "passed",
        "codegen": codegen,
        "verdict": "PASS" if verdict else "FAIL",
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("T080_SCATTER_GATE=" + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
