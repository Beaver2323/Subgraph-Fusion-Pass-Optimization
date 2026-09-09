#!/usr/bin/env python3
"""T-084 dedup reduce-scatter 功能与目标级性能 worker。"""

from __future__ import annotations

import argparse
from datetime import datetime
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
TASK_ID = "T-084"
UNIT = "dedup-reduce-scatter"
UNIT_ID = "AU-post-grad-dedup-reduce-scatters"
WORKLOAD = "dedup-reduce-scatter-community-shape"


def input_spec() -> list[dict]:
    return [
        {"shape": [4, 128], "dtype": "torch.float32"},
        {"shape": [4, 128], "dtype": "torch.float32"},
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_gate(path: Path, device: str) -> dict:
    gate = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "task_id": TASK_ID,
        "acceptance_unit_id": UNIT_ID,
        "backend": "triton_experimental" if device == "npu" else "inductor-default",
        "pytorch_commit": COMMIT,
        "correctness": "passed",
        "target_rewrite": "confirmed",
        "graph_breaks": 0,
        "fallbacks": 0,
        "product_disabled": False,
        "measurement_workload": WORKLOAD,
        "world_size": 2,
        "input_spec": input_spec(),
        "worker_sha256": sha256(Path(__file__)),
    }
    for key, value in expected.items():
        if gate.get(key) != value or type(gate.get(key)) is not type(value):
            raise ValueError(f"性能门禁 {key} 不符合要求：需要 {value!r}")
    if not gate.get("reviewed_at") or not gate.get("reviewer"):
        raise ValueError("性能门禁缺少复核时间或复核者")
    for key in ("gpu_reference", "target_functional"):
        item = gate.get(key, {})
        evidence = Path(item.get("path", ""))
        if not evidence.is_absolute():
            evidence = path.parent / evidence
        if not evidence.is_file() or sha256(evidence) != item.get("sha256"):
            raise ValueError(f"{key} 原件不存在或sha256不符")
        parsed = json.loads(evidence.read_text(encoding="utf-8"))
        if key == "gpu_reference":
            raw_valid = (
                parsed.get("status") == "valid-reference-suite"
                and parsed.get("suite_valid") is True
                and any(
                    case.get("acceptance_unit_id") == UNIT_ID
                    and case.get("reference_valid") is True
                    for case in parsed.get("cases", [])
                )
            )
            reviewed_valid = (
                str(parsed.get("review_status", "")).startswith("accepted")
                and parsed.get("task_id") == TASK_ID
                and parsed.get("pytorch_commit") == COMMIT
                and UNIT_ID in parsed.get("acceptance_units", [])
                and parsed.get("suite", {}).get("valid_cases")
                == parsed.get("suite", {}).get("cases")
                and parsed.get("suite", {}).get("tests_skipped") == 0
                and parsed.get("suite", {}).get("adapters_used") == 0
            )
            if not (raw_valid or reviewed_valid):
                raise ValueError("GPU reference原件无效或不包含T-084单元")
            continue
        for field, value in expected.items():
            if field in {"reviewed_at", "reviewer"}:
                continue
            if parsed.get(field) != value:
                raise ValueError(f"NPU功能原件 {field} 与性能门禁不一致")
        if parsed.get("numerical_execution") is not True:
            raise ValueError("FakeTensor或结构-only证据不能签发性能门禁")
        if parsed.get("process_group_backend") not in {"nccl", "hccl"}:
            raise ValueError("功能原件必须来自真实两rank NCCL/HCCL")
        sources = parsed.get("source_files", {})
        if not isinstance(sources, dict) or not sources:
            raise ValueError("功能原件缺少实际加载源码sha256")
        for source, digest in sources.items():
            actual = Path(source)
            if not actual.is_file() or sha256(actual) != digest:
                raise ValueError("功能原件绑定源码已变化")
    return gate


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    index = int(position)
    upper = min(index + 1, len(ordered) - 1)
    return ordered[index] + (ordered[upper] - ordered[index]) * (position - index)


def loaded_source_hashes() -> dict[str, str]:
    result = {}
    for name, module in list(sys.modules.items()):
        path_value = getattr(module, "__file__", None)
        if not name.startswith(("torch._inductor", "torch_npu._inductor", "triton")) or not path_value:
            continue
        path = Path(path_value)
        if path.is_file():
            result[str(path.resolve())] = sha256(path)
    return result


def main() -> None:
    process_started_at = datetime.now().astimezone().isoformat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", choices=(UNIT,), default=UNIT)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--device", choices=("cuda", "npu"), required=True)
    parser.add_argument("--phase", choices=("functional", "benchmark"), default="benchmark")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    if args.phase == "benchmark":
        if args.gate is None:
            parser.error("benchmark阶段必须提供--gate")
        gate = read_gate(args.gate.resolve(), args.device)
    else:
        gate = None
    if args.warmup < 1 or args.runs < 1:
        parser.error("warmup/runs必须为正整数")

    work_dir = Path(
        os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")
    ).resolve()
    if Path.cwd().resolve() != work_dir:
        raise RuntimeError(f"测试必须从 {work_dir} 启动")
    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"
    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank_dir = output / f"rank-{rank}"
    rank_dir.mkdir(exist_ok=True)
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(rank_dir / "debug")
    os.environ["TORCH_TRACE"] = str(rank_dir / "trace")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(rank_dir / "inductor-cache")
    os.environ["TRITON_CACHE_DIR"] = str(rank_dir / "triton-cache")

    import torch
    import torch.distributed as dist
    from torch._inductor import config
    from torch._inductor.fx_passes import fsdp, post_grad

    if torch.version.git_version != COMMIT:
        raise RuntimeError("实际加载PyTorch不是T-084冻结commit")
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu

        register_inductor_npu()
        if _InductorNpuRegistry._loaded_backend != "triton_experimental":
            raise RuntimeError("实际NPU后端不是triton_experimental")
    runtime = getattr(torch, args.device)
    if world_size != 2 or runtime.device_count() < 2:
        raise RuntimeError("T-084功能/性能要求torchrun两rank和至少两张可见设备")
    runtime.set_device(local_rank)
    device = f"{args.device}:{local_rank}"
    dist.init_process_group("hccl" if args.device == "npu" else "nccl")
    if dist.get_world_size() != 2 or dist.get_backend() not in {"hccl", "nccl"}:
        raise RuntimeError("拒绝单rank或fake process group")

    torch.manual_seed(20260909)
    state = {"handler_calls": 0, "graph_changes": 0, "collective_counts": [], "post_grad_counts": []}

    def collective_count(graph) -> int:
        return sum(
            node.op == "call_function"
            and "reduce_scatter_tensor" in str(node.target)
            for node in graph.nodes
        )

    original_helper = fsdp.dedup_fsdp_reduce_scatter

    @functools.wraps(original_helper)
    def observed_helper(gm, *extra, **kwargs):
        before = str(gm.graph)
        before_count = collective_count(gm.graph)
        state["handler_calls"] += 1
        result = original_helper(gm, *extra, **kwargs)
        after = str(gm.graph)
        after_count = collective_count(gm.graph)
        state["graph_changes"] += int(before != after)
        state["collective_counts"].append(
            {"before": before_count, "after": after_count}
        )
        (rank_dir / "target-before.txt").write_text(before, encoding="utf-8")
        (rank_dir / "target-after.txt").write_text(after, encoding="utf-8")
        return result

    fsdp.dedup_fsdp_reduce_scatter = observed_helper
    original_post_grad = post_grad.post_grad_passes

    @functools.wraps(original_post_grad)
    def observed_post_grad(gm, *extra, **kwargs):
        result = original_post_grad(gm, *extra, **kwargs)
        state["post_grad_counts"].append(collective_count(gm.graph))
        (rank_dir / "post-grad-final.txt").write_text(gm.code, encoding="utf-8")
        return result

    post_grad.post_grad_passes = observed_post_grad
    compile_fx = sys.modules.get("torch._inductor.compile_fx")
    if compile_fx is not None and getattr(compile_fx, "post_grad_passes", None) is original_post_grad:
        compile_fx.post_grad_passes = observed_post_grad

    group = dist.distributed_c10d._get_default_group().group_name
    reduce_scatter = torch.ops._c10d_functional.reduce_scatter_tensor
    wait_tensor = torch.ops.c10d_functional.wait_tensor

    def fn(first, second):
        first_out = reduce_scatter(first, "avg", world_size, group)
        second_out = reduce_scatter(second, "avg", world_size, group)
        return wait_tensor(first_out) + wait_tensor(second_out)

    inputs = (
        torch.ones(4, 128, device=device) * (rank + 1),
        torch.ones(4, 128, device=device) * (rank + 1),
    )
    actual_spec = [
        {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for tensor in inputs
    ]
    if actual_spec != input_spec() or not all(tensor.is_contiguous() for tensor in inputs):
        raise RuntimeError("实际输入不符合冻结社区shape/dtype/stride合同")
    expected = fn(*inputs)
    settings = {
        "dedup_reduce_scatters": args.mode == "on",
        "reorder_for_compute_comm_overlap": False,
        "fx_graph_cache": False,
        "force_disable_caches": True,
    }
    options = {"npu_backend": "triton_experimental"} if args.device == "npu" else None
    with config.patch(settings):
        started = time.perf_counter()
        compiled = torch.compile(fn, fullgraph=True, options=options)
        actual = compiled(*inputs)
        runtime.synchronize()
        compile_ms = (time.perf_counter() - started) * 1000
        torch.testing.assert_close(actual, expected)
        expected_count = 1 if args.mode == "on" else 2
        if state["post_grad_counts"] != [expected_count]:
            raise RuntimeError("post-grad最终collective数不符合目标OFF/ON合同")
        if args.mode == "on":
            if state["graph_changes"] < 1 or {"before": 2, "after": 1} not in state["collective_counts"]:
                raise RuntimeError("ON未捕获reduce-scatter 2→1，拒绝继续")
        elif state["handler_calls"] != 0:
            raise RuntimeError("OFF仍调用目标helper，拒绝继续")

        source_files = loaded_source_hashes()
        torch_root = Path(torch.__file__).resolve().parents[1]
        git_state = subprocess.run(
            ["git", "-C", str(torch_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        backend = "triton_experimental" if args.device == "npu" else "inductor-default"
        input_contract = [
            {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }
            for tensor in inputs
        ]
        common = {
            "schema_version": "1.0",
            "task_id": TASK_ID,
            "acceptance_unit_id": UNIT_ID,
            "unit": UNIT,
            "mode": args.mode,
            "backend": backend,
            "backend_selected_before_import": True,
            "pytorch_commit": torch.version.git_version,
            "pytorch_worktree_status": git_state,
            "correctness": "passed",
            "numerical_execution": True,
            "target_rewrite": "confirmed" if args.mode == "on" else "disabled-control",
            "graph_breaks": 0,
            "fallbacks": 0,
            "product_disabled": False,
            "measurement_workload": WORKLOAD,
            "world_size": world_size,
            "process_group_backend": dist.get_backend(),
            "input_spec": input_spec(),
            "input_contract": input_contract,
            "state": state,
            "compile_ms": compile_ms,
            "worker_sha256": sha256(Path(__file__)),
            "source_files": source_files,
            "rank": rank,
            "pid": os.getpid(),
            "created_at": datetime.now().astimezone().isoformat(),
        }
        if args.phase == "functional":
            (rank_dir / "functional_result.json").write_text(
                json.dumps(common, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            rank_results = [None] * world_size
            dist.all_gather_object(rank_results, common)
            if rank == 0:
                summary = dict(common)
                summary["rank_results"] = rank_results
                (output / "functional_summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            dist.destroy_process_group()
            return

        for _ in range(args.warmup):
            compiled(*inputs)
        runtime.synchronize()
        runtime.reset_peak_memory_stats()
        samples = {"host_ms": [], "event_ms": []}
        for _ in range(args.runs):
            dist.barrier()
            runtime.synchronize()
            start = runtime.Event(enable_timing=True)
            end = runtime.Event(enable_timing=True)
            before = time.perf_counter()
            start.record()
            compiled(*inputs)
            end.record()
            runtime.synchronize()
            samples["host_ms"].append((time.perf_counter() - before) * 1000)
            samples["event_ms"].append(start.elapsed_time(end))
        memory = {
            "allocated": runtime.max_memory_allocated(),
            "reserved": runtime.max_memory_reserved(),
        }

    record = dict(
        common,
        process_started_at=process_started_at,
        samples=samples,
        memory=memory,
        loaded_source_sha256=source_files,
        gate_sha256=sha256(args.gate.resolve()),
    )
    (rank_dir / "worker_result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rank_results = [None] * world_size
    dist.all_gather_object(rank_results, record)
    if rank == 0:
        maxima = {
            metric: [
                max(item["samples"][metric][index] for item in rank_results)
                for index in range(args.runs)
            ]
            for metric in samples
        }
        summary = {
            "acceptance_unit_id": UNIT_ID,
            "mode": args.mode,
            "world_size": world_size,
            "rank_max_samples": maxima,
            "timing": {
                metric: {"p50": percentile(values, 0.5), "p99": percentile(values, 0.99)}
                for metric, values in maxima.items()
            },
            "rank_peak_memory": [item["memory"] for item in rank_results],
            "compile_ms_per_rank": [item["compile_ms"] for item in rank_results],
        }
        (output / "arm_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
