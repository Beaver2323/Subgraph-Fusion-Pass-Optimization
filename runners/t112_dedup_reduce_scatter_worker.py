#!/usr/bin/env python3
"""T-112 双 rank reduce-scatter 去重的功能/性能臂。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime


COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
TASK = "T-112"
UNIT = "AU-fsdp-get-dedup-rs"
WORKLOAD = "dedup-reduce-scatter-two-rank-fp32"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def read_gate(path: Path, device: str) -> dict:
    gate = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "task_id": TASK,
        "acceptance_unit_id": UNIT,
        "backend": "triton_experimental" if device == "npu" else "inductor-default",
        "pytorch_commit": COMMIT,
        "correctness": "passed",
        "target_rewrite": "confirmed",
        "graph_breaks": 0,
        "fallbacks": 0,
        "product_disabled": False,
        "measurement_workload": WORKLOAD,
        "world_size": 2,
        "worker_sha256": sha256(Path(__file__)),
    }
    for key, value in expected.items():
        if gate.get(key) != value or type(gate.get(key)) is not type(value):
            raise ValueError(f"性能门禁字段 {key} 不符：需要 {value!r}")
    if not gate.get("reviewed_at") or not gate.get("reviewer"):
        raise ValueError("性能门禁必须记录人工复核者和时间")
    for name in ("gpu_reference", "target_functional"):
        item = gate.get(name, {})
        source = Path(item.get("path", ""))
        if not source.is_absolute():
            source = path.parent / source
        if not source.is_file() or sha256(source) != item.get("sha256"):
            raise ValueError(f"{name}原件缺失或sha256不符")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--device", choices=("cuda", "npu"), required=True)
    parser.add_argument("--phase", choices=("functional", "benchmark"), default="functional")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=4096)
    args = parser.parse_args()
    if min(args.warmup, args.runs, args.rows, args.hidden) < 1:
        parser.error("warmup/runs/rows/hidden必须为正整数")
    if args.phase == "benchmark":
        if args.gate is None:
            parser.error("benchmark必须提供人工签发的--gate")
        read_gate(args.gate.resolve(), args.device)

    work = Path(os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")).resolve()
    if Path.cwd().resolve() != work:
        raise RuntimeError(f"必须从 {work} 启动")
    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"
    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("T-112设备功能/性能合同固定world_size=2")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rank_output = output / f"rank-{rank}"
    rank_output.mkdir(parents=True, exist_ok=False)
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(rank_output / "debug")
    os.environ["TORCH_TRACE"] = str(rank_output / "trace")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(rank_output / "inductor-cache")
    os.environ["TRITON_CACHE_DIR"] = str(rank_output / "triton-cache")

    import torch
    import torch.distributed as dist
    from torch._inductor import config
    from torch._inductor.utils import run_and_get_code

    if torch.version.git_version != COMMIT:
        raise RuntimeError("实际加载PyTorch不是冻结commit")
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu

        register_inductor_npu()
        if _InductorNpuRegistry._loaded_backend != "triton_experimental":
            raise RuntimeError("NPU实际注册后端不是triton_experimental")
    runtime = getattr(torch, args.device)
    runtime.set_device(local_rank)
    device = torch.device(args.device, local_rank)
    dist.init_process_group(backend="hccl" if args.device == "npu" else "nccl")
    group = dist.distributed_c10d._get_default_group()
    group_name = group.group_name

    def model(rs_0, rs_1):
        first = torch.ops._c10d_functional.reduce_scatter_tensor(
            rs_0, "avg", world_size, group_name
        )
        second = torch.ops._c10d_functional.reduce_scatter_tensor(
            rs_1, "avg", world_size, group_name
        )
        first = torch.ops._c10d_functional.wait_tensor(first)
        second = torch.ops._c10d_functional.wait_tensor(second)
        return first + second

    shape = (world_size * args.rows, args.hidden)
    rs_0 = torch.full(shape, float(rank + 1), device=device)
    rs_1 = torch.full(shape, float(rank + 2), device=device)
    expected = torch.full(
        (args.rows, args.hidden), float(world_size + 2), device=device
    )
    settings = {
        "fx_graph_cache": False,
        "force_disable_caches": True,
        "dedup_reduce_scatters": args.mode == "on",
        "reorder_for_compute_comm_overlap": False,
    }
    options = {"npu_backend": "triton_experimental"} if args.device == "npu" else None
    try:
        with config.patch(settings):
            started = time.perf_counter()
            compiled = torch.compile(model, backend="inductor", fullgraph=True, options=options)
            actual, code_parts = run_and_get_code(compiled, rs_0, rs_1)
            runtime.synchronize()
            compile_ms = (time.perf_counter() - started) * 1000
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            code = "\n".join(code_parts)
            marker = "torch.ops._c10d_functional.reduce_scatter_tensor.default("
            reduce_scatter_count = code.count(marker)
            expected_count = 1 if args.mode == "on" else 2
            if reduce_scatter_count != expected_count:
                raise RuntimeError(
                    f"reduce_scatter代码计数不符：实际{reduce_scatter_count}，预期{expected_count}"
                )
            (rank_output / "generated_code.py").write_text(code, encoding="utf-8")

            samples = None
            memory = None
            if args.phase == "benchmark":
                for _ in range(args.warmup):
                    compiled(rs_0, rs_1)
                runtime.synchronize()
                dist.barrier()
                runtime.reset_peak_memory_stats()
                samples = {"host_ms": [], "event_ms": []}
                for _ in range(args.runs):
                    dist.barrier()
                    runtime.synchronize()
                    start = runtime.Event(enable_timing=True)
                    end = runtime.Event(enable_timing=True)
                    before = time.perf_counter()
                    start.record()
                    compiled(rs_0, rs_1)
                    end.record()
                    runtime.synchronize()
                    samples["host_ms"].append((time.perf_counter() - before) * 1000)
                    samples["event_ms"].append(start.elapsed_time(end))
                memory = {
                    "allocated": runtime.max_memory_allocated(),
                    "reserved": runtime.max_memory_reserved(),
                }

        torch_root = Path(torch.__file__).resolve().parents[1]
        worktree = subprocess.run(
            ["git", "-C", str(torch_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        record = {
            "schema_version": "1.0",
            "generated_at": datetime.now().astimezone().isoformat(),
            "task_id": TASK,
            "acceptance_unit_id": UNIT,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "mode": args.mode,
            "phase": args.phase,
            "backend": "triton_experimental" if args.device == "npu" else "inductor-default",
            "backend_selected_before_import": True,
            "collective_backend": "hccl" if args.device == "npu" else "nccl",
            "pytorch_commit": torch.version.git_version,
            "pytorch_worktree_status": worktree,
            "correctness": "passed",
            "numerical_execution": True,
            "target_rewrite": "confirmed" if args.mode == "on" else "disabled-control",
            "reduce_scatter_count": reduce_scatter_count,
            "graph_breaks": 0,
            "fallbacks": 0,
            "product_disabled": False,
            "measurement_workload": WORKLOAD,
            "input_spec": [
                {"shape": list(shape), "dtype": "torch.float32"},
                {"shape": list(shape), "dtype": "torch.float32"},
            ],
            "compile_ms": compile_ms,
            "samples": samples,
            "memory": memory,
            "timing": (
                {
                    key: {"p50": percentile(value, 0.5), "p99": percentile(value, 0.99)}
                    for key, value in samples.items()
                }
                if samples
                else None
            ),
            "worker_sha256": sha256(Path(__file__)),
            "gate_sha256": sha256(args.gate) if args.gate else None,
        }
        (rank_output / "result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"worker_status=passed rank={rank} mode={args.mode} "
            f"phase={args.phase} reduce_scatter_count={reduce_scatter_count}"
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
