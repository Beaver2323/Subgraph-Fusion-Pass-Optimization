#!/usr/bin/env python3
"""原生 unittest 的只读 FX 观察器（2026-09-07）。

保留原生参数、test body、设备、断言和 unittest 主入口。只包装 joint_graph
入口保存前后图，补齐 make_fx+直接 pass 测例不会生成 compile-debug 的证据。
这份图只证明 joint pass 结构；不冒充 lower/codegen/数值执行证据。
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("methods", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    methods = args.methods[1:] if args.methods[:1] == ["--"] else args.methods
    if not methods:
        parser.error("必须显式选择 unittest 方法")
    import torch
    from torch._inductor.fx_passes import joint_graph

    original = joint_graph.joint_graph_passes
    root = Path(os.environ["TORCH_COMPILE_DEBUG_DIR"]) / "native_joint_observer"
    index = 0

    @functools.wraps(original)
    def observed(gm, *call_args, **kwargs):
        nonlocal index
        index += 1
        destination = root / f"pid-{os.getpid()}-graph-{index:04d}"
        destination.mkdir(parents=True)
        (destination / "fx_graph_readable.py").write_text(gm.code, encoding="utf-8")
        result = original(gm, *call_args, **kwargs)
        output = result if isinstance(result, torch.fx.GraphModule) else gm
        (destination / "fx_graph_transformed.py").write_text(output.code, encoding="utf-8")
        metadata = {
            "capture_scope": "joint_graph_passes-entry-exit",
            "test_body_modified": False,
            "device_modified": False,
            "assertions_modified": False,
            "numerical_execution_proven_by_observer": False,
            "tensor_devices": sorted({str(v.device) for n in gm.graph.nodes
                                      if isinstance((v := n.meta.get("val")), torch.Tensor)}),
        }
        (destination / "observer.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result

    joint_graph.joint_graph_passes = observed
    # compile_fx在某些测试的导入链中已缓存from-import别名，同步观察同一入口。
    compile_fx = sys.modules.get("torch._inductor.compile_fx")
    if compile_fx is not None and getattr(compile_fx, "joint_graph_passes", None) is original:
        compile_fx.joint_graph_passes = observed
    sys.path.insert(0, str(args.source.resolve().parent))
    sys.argv = [str(args.source), "-v", *methods]
    runpy.run_path(str(args.source), run_name="__main__")


if __name__ == "__main__":
    main()
