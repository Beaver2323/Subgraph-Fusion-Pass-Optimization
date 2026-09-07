#!/usr/bin/env python3
"""T-079 单臂目标级 NPU 性能 worker；每个进程只运行一个 OFF/ON arm。"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any


os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"
WORK = Path("/home/z50063656/tmp")
if Path.cwd().resolve() != WORK:
    raise RuntimeError("必须从 /home/z50063656/tmp 启动")

import torch
import torch_npu
from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu


PYTORCH_COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
TARGETS = {
    "bmm-to-mm": {
        "handler": "bmm_to_mm",
        "unit": "AU-joint-graph-bmm-to-mm",
        "registry": "joint",
    },
    "cat-slice-cat": {
        "handler": "cat_slice_cat",
        "unit": "AU-post-grad-cat-slice-cat",
        "registry": "post",
    },
    "split-cat": {
        "handler": "splitwithsizes_cat_replace",
        "unit": "AU-post-grad-splitwithsizes-cat-replace",
        "registry": "post",
    },
    "cat-split": {
        "handler": "cat_splitwithsizes_replace",
        "unit": "AU-post-grad-cat-splitwithsizes-replace",
        "registry": "post",
    },
}


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50_ms": percentile(values, 0.50),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def install_target_control(unit: str, mode: str, hits: dict[str, int]) -> list[dict[str, Any]]:
    from torch._inductor.fx_passes import joint_graph, post_grad

    target = TARGETS[unit]
    registries = [joint_graph.patterns] if target["registry"] == "joint" else post_grad.pass_patterns
    selected = []
    seen: set[int] = set()
    for registry in registries:
        for entries in registry.patterns.values():
            for entry in entries:
                if id(entry) in seen:
                    continue
                seen.add(id(entry))
                handler = getattr(entry, "handler", None)
                if getattr(handler, "__name__", None) != target["handler"]:
                    continue
                selected.append(
                    {
                        "handler": target["handler"],
                        "entry_type": type(entry).__name__,
                    }
                )
                original = handler

                @functools.wraps(original)
                def observed(*args, _handler=original, **kwargs):
                    hits["total"] = hits.get("total", 0) + 1
                    return _handler(*args, **kwargs)

                entry.handler = observed
                if mode == "off":
                    entry.extra_check = lambda match: False
    if not selected:
        raise RuntimeError(f"未找到目标 handler: {target['handler']}")
    return selected


def make_workloads(unit: str) -> list[dict[str, Any]]:
    torch.manual_seed(20260907)
    torch.npu.manual_seed_all(20260907)
    if unit == "bmm-to-mm":
        def fn(a, b):
            return torch.bmm(a, b)

        shapes = [
            ("community-shape", 16, 8, 32),
            ("sensitivity-64", 64, 64, 64),
            ("sensitivity-256", 256, 64, 256),
            ("sensitivity-1024", 1024, 128, 512),
        ]
        return [
            {
                "id": workload_id,
                "fn": fn,
                "inputs": (
                    torch.randn(1, m, k, device="npu"),
                    torch.randn(1, k, n, device="npu"),
                ),
                "shape": [1, m, k, n],
                "community_primary": workload_id == "community-shape",
            }
            for workload_id, m, k, n in shapes
        ]
    if unit == "cat-slice-cat":
        def fn(a, b):
            cat1 = torch.ops.aten.cat.default([a, b], 1)
            sliced = torch.ops.aten.slice.Tensor(cat1, 1, 0, 19)
            return torch.ops.aten.cat.default([cat1, sliced], 1)

        return [{
            "id": "community-shape",
            "fn": fn,
            "inputs": (
                torch.randn(2, 32, device="npu"),
                torch.randn(2, 16, device="npu"),
            ),
            "shape": [[2, 32], [2, 16]],
            "community_primary": True,
        }]
    if unit == "split-cat":
        def fn(a):
            parts = torch.ops.aten.split_with_sizes.default(a, [8, 24], 1)
            return torch.ops.aten.cat.default([parts[0], parts[1]], 1) ** 2

        return [{
            "id": "community-shape",
            "fn": fn,
            "inputs": (torch.randn(2, 32, device="npu"),),
            "shape": [2, 32],
            "community_primary": True,
        }]
    if unit == "cat-split":
        def fn(a, b, c):
            joined = torch.ops.aten.cat.default([a, b, c], 1)
            parts = torch.ops.aten.split_with_sizes.default(joined, [2, 3, 5], 1)
            return tuple(part**2 for part in parts)

        return [{
            "id": "community-shape",
            "fn": fn,
            "inputs": (
                torch.randn(2, 2, device="npu"),
                torch.randn(2, 3, device="npu"),
                torch.randn(2, 5, device="npu"),
            ),
            "shape": [[2, 2], [2, 3], [2, 5]],
            "community_primary": True,
        }]
    raise AssertionError(unit)


def count_codegen(debug_root: Path) -> dict[str, Any]:
    files = sorted(debug_root.rglob("output_code.py"))
    texts = [path.read_text(errors="replace") for path in files]
    joined = "\n".join(texts)
    return {
        "output_code_files": len(files),
        "extern_mm": joined.count("extern_kernels.mm("),
        "extern_bmm": joined.count("extern_kernels.bmm("),
        "triton_kernels": joined.count(".triton("),
        "sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in files],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", choices=sorted(TARGETS), required=True)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    register_inductor_npu()
    if _InductorNpuRegistry._loaded_backend != "triton_experimental":
        raise RuntimeError("实际 backend 不是 triton_experimental")
    if torch.version.git_version != PYTORCH_COMMIT:
        raise RuntimeError("PyTorch commit 不匹配")
    torch.npu.set_device(0)
    hits: dict[str, int] = {}
    selected = install_target_control(args.unit, args.mode, hits)
    workloads = []
    for item in make_workloads(args.unit):
        torch._dynamo.reset()
        eager = item["fn"](*item["inputs"])
        torch.npu.synchronize()
        started = time.perf_counter()
        config = {"npu_backend": "triton_experimental"}
        if args.unit == "bmm-to-mm":
            config["max_autotune_gemm_backends"] = "ATEN"
        compiled = torch.compile(item["fn"], options=config)
        actual = compiled(*item["inputs"])
        torch.npu.synchronize()
        compile_ms = (time.perf_counter() - started) * 1000
        torch.testing.assert_close(actual, eager)
        for _ in range(args.warmup):
            compiled(*item["inputs"])
        torch.npu.synchronize()

        torch.npu.reset_peak_memory_stats()
        event_ms = []
        for _ in range(args.runs):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            compiled(*item["inputs"])
            end.record()
            torch.npu.synchronize()
            event_ms.append(start.elapsed_time(end))
        host_ms = []
        for _ in range(args.runs):
            before = time.perf_counter()
            compiled(*item["inputs"])
            torch.npu.synchronize()
            host_ms.append((time.perf_counter() - before) * 1000)
        workloads.append(
            {
                "workload_id": item["id"],
                "shape": item["shape"],
                "community_primary": item["community_primary"],
                "correctness": "passed",
                "compile_ms": compile_ms,
                "event": summarize(event_ms),
                "host": summarize(host_ms),
                "max_memory_allocated": torch.npu.max_memory_allocated(),
                "max_memory_reserved": torch.npu.max_memory_reserved(),
            }
        )

    if args.mode == "on" and hits.get("total", 0) < len(workloads):
        raise RuntimeError(f"ON target hits 不足: {hits}")
    if args.mode == "off" and hits.get("total", 0) != 0:
        raise RuntimeError(f"OFF 仍命中 target: {hits}")
    debug_root = Path(os.environ.get("TORCH_COMPILE_DEBUG_DIR", args.output.parent / "debug"))
    result = {
        "schema_version": "1.0",
        "task_id": "T-079",
        "acceptance_unit_id": TARGETS[args.unit]["unit"],
        "unit": args.unit,
        "mode": args.mode,
        "round": args.round,
        "backend": _InductorNpuRegistry._loaded_backend,
        "torch_commit": torch.version.git_version,
        "torch_npu_commit": git_head(Path("/home/z50063656/Pass/src/torch_npu")),
        "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "target_entries": selected,
        "target_hits": hits.get("total", 0),
        "product_gate_bypassed": False,
        "warmup": args.warmup,
        "runs": args.runs,
        "workloads": workloads,
        "codegen": count_codegen(debug_root),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("T079_PERFORMANCE=" + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
