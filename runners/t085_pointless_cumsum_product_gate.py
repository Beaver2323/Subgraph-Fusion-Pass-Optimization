#!/usr/bin/env python3
"""真机验证T-085 pointless-cumsum NPU产品默认门禁。"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys


COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = Path(os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp"))
    if Path.cwd().resolve() != work.resolve():
        parser.error(f"必须从{work.resolve()}启动")
    if os.environ.get("TORCHINDUCTOR_NPU_BACKEND") != "triton_experimental":
        parser.error("必须在导入torch前选择triton_experimental")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(output / "debug")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(output / "inductor-cache")
    os.environ["TRITON_CACHE_DIR"] = str(output / "triton-cache")
    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

    import torch
    import torch_npu  # noqa: F401
    from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu

    register_inductor_npu()
    from torch._inductor.fx_passes import post_grad
    from torch_npu._inductor.triton_experimental import config as npu_config

    if _InductorNpuRegistry._loaded_backend != "triton_experimental":
        raise RuntimeError("实际后端不是triton_experimental")
    if torch.version.git_version != COMMIT:
        raise RuntimeError("PyTorch不是冻结commit")
    if npu_config.disable_pointless_cumsum is not True:
        raise RuntimeError("产品默认门禁未关闭pointless-cumsum")

    entries = []
    seen = set()
    for patterns in post_grad.pass_patterns[1].patterns.values():
        for entry in patterns:
            if id(entry) in seen:
                continue
            seen.add(id(entry))
            if (
                getattr(getattr(entry, "handler", None), "__name__", None)
                == "pointless_cumsum_replacement"
            ):
                entries.append(entry)
    if len(entries) != 1:
        raise RuntimeError(f"目标pattern注册数量异常：{len(entries)}")
    entry = entries[0]
    if not getattr(
        entry.extra_check, "_torch_npu_pointless_cumsum_gate_installed", False
    ):
        raise RuntimeError("pointless-cumsum NPU live gate未安装")
    state = {"handler_calls": 0}
    original_handler = entry.handler

    def observed_handler(*handler_args, **handler_kwargs):
        state["handler_calls"] += 1
        return original_handler(*handler_args, **handler_kwargs)

    entry.handler = observed_handler

    def function():
        value = torch.full([5000], 1.0, dtype=torch.float16, device="npu")
        return torch.cumsum(value, 0, dtype=torch.float32)

    expected = function()
    torch._dynamo.reset()
    compiled = torch.compile(function, backend="inductor", fullgraph=True)
    actual = compiled()
    torch.npu.synchronize()
    if not torch.equal(actual, expected):
        raise AssertionError("产品门禁下compiled与eager不一致")
    if state["handler_calls"] != 0:
        raise AssertionError("产品默认门禁仍进入pointless-cumsum handler")

    output_codes = sorted((output / "debug").rglob("output_code.py"))
    if not output_codes:
        raise RuntimeError("缺少output_code.py")
    code = output_codes[0].read_text(encoding="utf-8", errors="replace")
    native_calls = code.count("buf1 = torch.ops.aten.cumsum.default(")
    if native_calls != 1:
        raise AssertionError("产品门禁后没有保留唯一NPU原生cumsum调用")
    sources = {}
    for module in list(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if not source:
            continue
        path = Path(source)
        if path.is_file() and path.name in {"config.py", "fx_passes.py"} and (
            "triton_experimental" in str(path)
            or str(path).endswith("torch/_inductor/fx_passes/post_grad.py")
        ):
            sources[str(path.resolve())] = sha256(path)
    result = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "task_id": "T-085",
        "acceptance_unit_id": "AU-post-grad-pointless-cumsum",
        "backend": "triton_experimental",
        "pytorch_commit": torch.version.git_version,
        "device": "npu",
        "device_execution": True,
        "correctness": "passed",
        "product_gate": "disable_pointless_cumsum=True",
        "handler_calls": state["handler_calls"],
        "native_cumsum_calls": native_calls,
        "output_code": {
            "path": str(output_codes[0]),
            "sha256": sha256(output_codes[0]),
        },
        "source_files": sources,
    }
    path = output / "product_gate_result.json"
    write_json(path, result)
    print(f"product_gate_result={path}")


if __name__ == "__main__":
    main()
