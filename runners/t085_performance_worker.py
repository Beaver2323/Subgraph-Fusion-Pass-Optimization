#!/usr/bin/env python3
"""T-085目标级功能/性能单臂worker；每次调用只运行一个OFF或ON新进程。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from datetime import datetime


COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
TASK_ID = "T-085"
TARGETS = {
    "overlap-device-put": {
        "acceptance_unit_id": "AU-post-grad-overlap-scheduling-device-put-sync",
        "world_size": 2,
        "workload": "overlap-device-put-community-shape",
    },
    "pointless-cumsum": {
        "acceptance_unit_id": "AU-post-grad-pointless-cumsum",
        "world_size": 1,
        "workload": "pointless-cumsum-fn10-community-shape",
    },
    "partitioned-scatter": {
        "acceptance_unit_id": "AU-post-grad-partitioned-scatter-optimization",
        "world_size": 1,
        "workload": "partitioned-scatter-community-benchmark-shape",
    },
}


def input_spec(unit: str) -> list[dict[str, object]]:
    if unit == "pointless-cumsum":
        return [
            {
                "role": "constructed-constant",
                "shape": [5000],
                "dtype": "torch.float16",
                "output_dtype": "torch.float32",
            }
        ]
    if unit == "overlap-device-put":
        return [
            {
                "role": "num_tokens_per_expert",
                "shape": [8],
                "dtype": "torch.int64",
            },
            {
                "role": "routed_input",
                "shape": [1024, 128],
                "dtype": "torch.float32",
            },
        ]
    return [
        {"role": "out0", "shape": [501, 100], "dtype": "torch.float32"},
        {"role": "out1", "shape": [501, 100], "dtype": "torch.float32"},
        {"role": "out2", "shape": [501, 100], "dtype": "torch.float32"},
        {"role": "index", "shape": [1000000], "dtype": "torch.int64"},
        {"role": "values", "shape": [1000000, 100], "dtype": "torch.float32"},
    ]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def summarize(values: list[float]) -> dict[str, object]:
    return {
        "p50": statistics.median(values),
        "p99": percentile(values, 0.99),
        "samples": values,
    }


def read_gate(path: Path, unit: str, device: str) -> dict:
    gate = json.loads(path.read_text(encoding="utf-8"))
    target = TARGETS[unit]
    expected = {
        "task_id": TASK_ID,
        "acceptance_unit_id": target["acceptance_unit_id"],
        "backend": "triton_experimental" if device == "npu" else "inductor-default",
        "pytorch_commit": COMMIT,
        "correctness": "passed",
        "target_rewrite": "confirmed",
        "graph_breaks": 0,
        "fallbacks": 0,
        "product_disabled": False,
        "measurement_workload": target["workload"],
        "world_size": target["world_size"],
        "input_spec": input_spec(unit),
        "worker_sha256": sha256(Path(__file__)),
    }
    for key, value in expected.items():
        if gate.get(key) != value or type(gate.get(key)) is not type(value):
            raise ValueError(f"性能gate字段{key}不符：需要{value!r}")
    if not gate.get("reviewed_at") or not gate.get("reviewer"):
        raise ValueError("性能gate缺少人工复核者或时间")
    for evidence_name in ("gpu_reference", "target_functional"):
        evidence = gate.get(evidence_name, {})
        evidence_path = Path(evidence.get("path", ""))
        if not evidence_path.is_absolute():
            evidence_path = path.parent / evidence_path
        if (
            not evidence_path.is_file()
            or sha256(evidence_path) != evidence.get("sha256")
        ):
            raise ValueError(f"性能gate绑定的{evidence_name}原件不存在或sha256不符")
        original = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence_name == "gpu_reference":
            reviewed_valid = (
                str(original.get("review_status", "")).startswith("accepted")
                and original.get("task_id") == TASK_ID
                and original.get("pytorch_commit") == COMMIT
                and target["acceptance_unit_id"]
                in original.get("acceptance_units", [])
                and original.get("suite", {}).get("valid_cases")
                == original.get("suite", {}).get("cases")
                and original.get("suite", {}).get("tests_skipped") == 0
                and original.get("suite", {}).get("adapters_used") == 0
            )
            if not reviewed_valid:
                raise ValueError("GPU reference复核原件无效或未包含当前单元")
            continue
        for key in expected:
            if original.get(key) != expected[key]:
                raise ValueError(f"功能原件字段{key}与gate不一致")
        if original.get("numerical_execution") is not True:
            raise ValueError("结构-only证据不能签发性能gate")
        sources = original.get("source_files", {})
        if not isinstance(sources, dict) or not sources:
            raise ValueError("功能原件缺少实际源码路径与sha256")
        for name, digest in sources.items():
            source = Path(name)
            if not source.is_file() or sha256(source) != digest:
                raise ValueError("功能原件绑定的当前源码不存在或已变化")
        if unit == "overlap-device-put" and original.get(
            "process_group_backend"
        ) not in {"nccl", "hccl"}:
            raise ValueError("overlap性能必须绑定真实NCCL/HCCL 2-rank功能证据")
    return gate


def source_hashes() -> dict[str, str]:
    result = {}
    for module in list(sys.modules.values()):
        file_name = getattr(module, "__file__", None)
        if not file_name:
            continue
        path = Path(file_name)
        if path.is_file() and any(
            token in str(path)
            for token in (
                "torch/_inductor/fx_passes/post_grad.py",
                "torch/_inductor/fx_passes/overlap_scheduling.py",
                "torch/_inductor/fx_passes/reduced_atomic_contention.py",
                "torch/_inductor/pattern_matcher.py",
                "torch_npu/_inductor",
            )
        ):
            result[str(path.resolve())] = sha256(path)
    return result


def event_sample(runtime, function) -> tuple[float, object]:
    start = runtime.Event(enable_timing=True)
    end = runtime.Event(enable_timing=True)
    start.record()
    value = function()
    end.record()
    runtime.synchronize()
    return float(start.elapsed_time(end)), value


def assert_outputs_close(torch, actual, expected, *, atol=1e-3, rtol=1e-3):
    actual_leaves = torch.utils._pytree.tree_leaves(actual)
    expected_leaves = torch.utils._pytree.tree_leaves(expected)
    if len(actual_leaves) != len(expected_leaves):
        raise AssertionError("输出树结构不一致")
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        if not torch.allclose(actual_leaf, expected_leaf, atol=atol, rtol=rtol):
            raise AssertionError("输出数值与eager不一致")


def run_pointless_cumsum(torch, config, runtime, device: str, mode: str, rank_dir: Path):
    from torch._dynamo.utils import counters
    from torch._inductor.fx_passes import post_grad

    matches = []
    for registry in post_grad.pass_patterns:
        for entries in registry.patterns.values():
            for entry in entries:
                handler = getattr(entry, "handler", None)
                if getattr(handler, "__name__", None) == "pointless_cumsum_replacement":
                    matches.append(entry)
    if len(matches) != 1:
        raise RuntimeError(f"目标pointless_cumsum注册数量异常：{len(matches)}")

    entry = matches[0]
    state = {"handler_calls": 0}
    original_handler = entry.handler

    def observed_handler(*args, **kwargs):
        state["handler_calls"] += 1
        return original_handler(*args, **kwargs)

    entry.handler = observed_handler
    if mode == "off":
        entry.extra_check = lambda match: False

    def function():
        value = torch.full([5000], 1.0, dtype=torch.float16, device=device)
        return torch.cumsum(value, 0, dtype=torch.float32)

    expected = function()
    settings = {
        "fx_graph_cache": False,
        "force_disable_caches": True,
        "joint_graph_constant_folding": True,
    }
    counters.clear()
    torch._dynamo.reset()
    started = time.perf_counter()
    with config.patch(settings):
        compiled = torch.compile(function, backend="inductor", fullgraph=True)
        actual = compiled()
    runtime.synchronize()
    compile_ms = (time.perf_counter() - started) * 1000
    if not torch.equal(actual, expected) or actual.dtype != torch.float32:
        raise AssertionError("pointless-cumsum输出或dtype与eager不一致")
    if mode == "on" and state["handler_calls"] < 1:
        raise AssertionError("ON未进入pointless_cumsum_replacement")
    if mode == "off" and state["handler_calls"] != 0:
        raise AssertionError("OFF仍进入pointless_cumsum_replacement")
    write_json(rank_dir / "target_state.json", state)
    return compiled, expected, state, compile_ms


def run_overlap(torch, config, runtime, device: str, mode: str, rank_dir: Path):
    import torch.distributed as dist
    from torch._inductor.fx_passes import overlap_scheduling

    rank = dist.get_rank()
    world = dist.get_world_size()
    group_name = dist.distributed_c10d._get_default_group().group_name
    torch._C._distributed_c10d._register_process_group(
        "t085-default", dist.distributed_c10d._get_default_group()
    )
    group_name = "t085-default"

    state = {"scheduler_calls": 0, "converted_device_puts": 0}
    original_schedule = overlap_scheduling.schedule_overlap_bucketing_from_inductor_configs
    original_convert = overlap_scheduling.make_all_device_put_sync

    def observed_convert(gm):
        count = original_convert(gm)
        state["converted_device_puts"] += count
        return count

    def observed_schedule(gm):
        index = state["scheduler_calls"]
        state["scheduler_calls"] += 1
        (rank_dir / f"target-{index}-before.txt").write_text(gm.code, encoding="utf-8")
        result = original_schedule(gm)
        (rank_dir / f"target-{index}-after.txt").write_text(result.code, encoding="utf-8")
        return result

    overlap_scheduling.make_all_device_put_sync = observed_convert
    overlap_scheduling.schedule_overlap_bucketing_from_inductor_configs = observed_schedule

    def function(num_tokens_per_expert, routed_input):
        exchanged = torch.ops._c10d_functional.all_to_all_single(
            num_tokens_per_expert,
            [num_tokens_per_expert.size(0) // world] * world,
            [num_tokens_per_expert.size(0) // world] * world,
            group_name,
        )
        exchanged = torch.ops._c10d_functional.wait_tensor(exchanged)
        input_splits = num_tokens_per_expert.view(world, -1).sum(dim=1)
        output_splits = exchanged.view(world, -1).sum(dim=1)
        cpu_input = input_splits.to("cpu", non_blocking=True)
        cpu_output = output_splits.to("cpu", non_blocking=False)
        routed = torch.ops._c10d_functional.all_to_all_single(
            routed_input,
            cpu_output.tolist(),
            cpu_input.tolist(),
            group_name,
        )
        return torch.ops._c10d_functional.wait_tensor(routed)

    torch.manual_seed(20260909)
    scores = torch.randn(512, 8, device=device, dtype=torch.float32)
    selected = torch.topk(scores, k=2, dim=1).indices.reshape(-1)
    counts = torch.bincount(selected, minlength=8).to(torch.int64)
    routed = torch.randn(int(counts.sum().item()), 128, device=device)
    expected = function(counts, routed)
    dist.barrier()
    settings = {
        "fx_graph_cache": False,
        "force_disable_caches": True,
        "aten_distributed_optimizations.enable_overlap_scheduling": mode == "on",
        "aten_distributed_optimizations.collective_bucketing": False,
        "aten_distributed_optimizations.insert_overlap_deps": True,
        "aten_distributed_optimizations.collective_estimator": "analytical",
        "aten_distributed_optimizations.compute_estimator": "analytical",
        "aten_distributed_optimizations.max_memory_increase_ratio": 0.05,
    }
    torch._dynamo.reset()
    started = time.perf_counter()
    with config.patch(settings):
        compiled = torch.compile(function, backend="inductor", fullgraph=True)
        actual = compiled(counts, routed)
    runtime.synchronize()
    dist.barrier()
    compile_ms = (time.perf_counter() - started) * 1000
    if not torch.allclose(actual, expected, rtol=1e-3, atol=1e-3):
        raise AssertionError("overlap device-put输出与eager不一致")
    if mode == "on" and (
        state["scheduler_calls"] < 1 or state["converted_device_puts"] < 1
    ):
        raise AssertionError("ON未进入目标scheduler或未转换async device_put")
    if mode == "off" and state["scheduler_calls"] != 0:
        raise AssertionError("OFF仍进入目标scheduler")
    write_json(rank_dir / "target_state.json", state)
    return lambda: compiled(counts, routed), expected, state, compile_ms


def run_partitioned_scatter(
    torch,
    config,
    runtime,
    device: str,
    mode: str,
    phase: str,
    rank_dir: Path,
):
    """复用社区性能图，并补跑社区accuracy/negative合同。"""
    from torch._dynamo.utils import counters
    from torch._inductor.fx_passes import post_grad
    from torch._inductor.fx_passes import reduced_atomic_contention as scatter

    memory_builder = scatter._build_scatter_memory_state
    marker = "_torch_npu_triton_experimental_scatter_memory"
    if device.startswith("npu") and not getattr(memory_builder, marker, False):
        raise RuntimeError("NPU partitioned-scatter显存门禁适配未激活")

    state = {
        "pass_calls": 0,
        "memory_probe_calls": 0,
        "memory_state": [],
        "partitioned_scatter_applied": 0,
        "negative_accumulate_false_applied": None,
    }
    original_pass = post_grad.partitioned_scatter_optimization_pass

    def observed_memory(graph):
        state["memory_probe_calls"] += 1
        result = memory_builder(graph)
        state["memory_state"].append(
            None
            if result is None
            else {
                "total_device_bytes": result.total_gpu_bytes,
                "non_model_floor_bytes": result.non_model_floor_bytes,
                "allowed_peak_bytes": result.allowed_peak_bytes,
                "profiled_nodes": len(result.peak_mem_by_node),
            }
        )
        return result

    def observed_pass(graph):
        index = state["pass_calls"]
        state["pass_calls"] += 1
        (rank_dir / f"target-{index}-before.txt").write_text(
            graph.python_code("self").src, encoding="utf-8"
        )
        result = original_pass(graph)
        (rank_dir / f"target-{index}-after.txt").write_text(
            result.python_code("self").src, encoding="utf-8"
        )
        return result

    scatter._build_scatter_memory_state = observed_memory
    post_grad.partitioned_scatter_optimization_pass = observed_pass

    def scatter_fn(out0, out1, out2, index, values):
        out0 = out0.index_put([index], values, accumulate=True)
        out1 = out1.index_put([index], values, accumulate=True)
        out2 = out2.index_put([index], values, accumulate=True)
        return out0, out1, out2

    torch.manual_seed(42)
    n_rows, width, output_rows = 1_000_000, 100, 501
    values = torch.randn(n_rows, width, dtype=torch.float32, device=device)
    outputs = tuple(
        torch.zeros(output_rows, width, dtype=torch.float32, device=device)
        for _ in range(3)
    )
    index = torch.randint(0, 8, (n_rows,), dtype=torch.int64, device=device)
    args = (*outputs, index, values)
    expected = scatter_fn(*args)
    settings = {
        "fx_graph_cache": False,
        "force_disable_caches": True,
        "partitioned_scatter_enabled": mode == "on",
        "partitioned_scatter_force": False,
    }
    counters.clear()
    torch._dynamo.reset()
    started = time.perf_counter()
    with config.patch(settings):
        compiled = torch.compile(scatter_fn, backend="inductor", fullgraph=True)
        actual = compiled(*args)
    runtime.synchronize()
    compile_ms = (time.perf_counter() - started) * 1000
    assert_outputs_close(torch, actual, expected, atol=1.0, rtol=1e-2)
    state["partitioned_scatter_applied"] = counters["inductor"][
        "partitioned_scatter_applied"
    ]
    if mode == "on":
        if state["partitioned_scatter_applied"] < 3:
            raise AssertionError("ON未分区化三个高争用scatter")
        if state["memory_probe_calls"] < 1 or not any(state["memory_state"]):
            raise AssertionError("ON未使用真实NPU显存预算")
    elif state["partitioned_scatter_applied"] != 0 or state["pass_calls"] != 0:
        raise AssertionError("OFF仍进入partitioned-scatter改写")

    if phase == "functional":

        def negative_fn(out, negative_index, negative_values):
            return out.index_put(
                [negative_index], negative_values, accumulate=False
            )

        negative_out = torch.zeros(256, dtype=torch.float32, device=device)
        negative_index = torch.randperm(256, dtype=torch.int64, device=device)
        negative_values = torch.randn(256, dtype=torch.float32, device=device)
        negative_expected = negative_fn(
            negative_out, negative_index, negative_values
        )
        before_negative = counters["inductor"]["partitioned_scatter_applied"]
        torch._dynamo.reset()
        with config.patch(settings):
            negative_compiled = torch.compile(
                negative_fn, backend="inductor", fullgraph=True
            )
            negative_actual = negative_compiled(
                negative_out, negative_index, negative_values
            )
        runtime.synchronize()
        if not torch.equal(negative_actual, negative_expected):
            raise AssertionError("accumulate=False负例与eager不一致")
        negative_delta = (
            counters["inductor"]["partitioned_scatter_applied"]
            - before_negative
        )
        state["negative_accumulate_false_applied"] = negative_delta
        if negative_delta != 0:
            raise AssertionError("accumulate=False负例被错误分区化")
    write_json(rank_dir / "target_state.json", state)
    return lambda: compiled(*args), expected, state, compile_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", choices=TARGETS, required=True)
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
            parser.error("benchmark阶段必须提供人工签署gate")
        gate = read_gate(args.gate.resolve(), args.unit, args.device)
    else:
        gate = None
    if args.warmup < 1 or args.runs < 1:
        parser.error("warmup/runs必须为正整数")

    work_dir = Path(os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")).resolve()
    if Path.cwd().resolve() != work_dir:
        raise RuntimeError(f"必须从{work_dir}启动")
    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"
    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

    output = args.output.resolve()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    expected_world = TARGETS[args.unit]["world_size"]
    if world != expected_world:
        raise RuntimeError(f"{args.unit}要求world_size={expected_world}，实际{world}")
    rank_dir = output / f"rank-{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(rank_dir / "debug")
    os.environ["TORCH_TRACE"] = str(rank_dir / "trace")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(rank_dir / "inductor-cache")
    os.environ["TRITON_CACHE_DIR"] = str(rank_dir / "triton-cache")

    import torch
    from torch._inductor import config

    if torch.version.git_version != COMMIT:
        raise RuntimeError("实际PyTorch不是冻结commit")
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu

        register_inductor_npu()
        if _InductorNpuRegistry._loaded_backend != "triton_experimental":
            raise RuntimeError("实际NPU后端不是triton_experimental")
    runtime = getattr(torch, args.device)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    runtime.set_device(local_rank)
    device = f"{args.device}:{local_rank}"

    dist = None
    if expected_world == 2:
        import torch.distributed as dist

        dist.init_process_group("hccl" if args.device == "npu" else "nccl")
        if dist.get_world_size() != 2 or dist.get_backend() not in {"nccl", "hccl"}:
            raise RuntimeError("拒绝单rank或非真实NCCL/HCCL")

    runtime.reset_peak_memory_stats()
    if args.unit == "pointless-cumsum":
        function, expected, state, compile_ms = run_pointless_cumsum(
            torch, config, runtime, device, args.mode, rank_dir
        )
    elif args.unit == "overlap-device-put":
        function, expected, state, compile_ms = run_overlap(
            torch, config, runtime, device, args.mode, rank_dir
        )
    else:
        function, expected, state, compile_ms = run_partitioned_scatter(
            torch, config, runtime, device, args.mode, args.phase, rank_dir
        )

    for _ in range(args.warmup):
        function()
    runtime.synchronize()
    if dist is not None:
        dist.barrier()

    host_ms = []
    event_ms = []
    for _ in range(args.runs):
        if dist is not None:
            dist.barrier()
        runtime.synchronize()
        started = time.perf_counter_ns()
        actual = function()
        runtime.synchronize()
        host_ms.append((time.perf_counter_ns() - started) / 1e6)
        value, event_actual = event_sample(runtime, function)
        event_ms.append(value)
        atol = 1.0 if args.unit == "partitioned-scatter" else 1e-3
        rtol = 1e-2 if args.unit == "partitioned-scatter" else 1e-3
        assert_outputs_close(torch, actual, expected, atol=atol, rtol=rtol)
        assert_outputs_close(torch, event_actual, expected, atol=atol, rtol=rtol)

    result = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "task_id": TASK_ID,
        "acceptance_unit_id": TARGETS[args.unit]["acceptance_unit_id"],
        "unit": args.unit,
        "mode": args.mode,
        "phase": args.phase,
        "backend": "triton_experimental" if args.device == "npu" else "inductor-default",
        "device": args.device,
        "rank": rank,
        "world_size": world,
        "process_group_backend": dist.get_backend() if dist is not None else None,
        "pid": os.getpid(),
        "pytorch_commit": torch.version.git_version,
        "measurement_workload": TARGETS[args.unit]["workload"],
        "input_spec": input_spec(args.unit),
        "target_state": state,
        "correctness": "passed",
        "compile_ms": compile_ms,
        "timing": {"host_ms": summarize(host_ms), "event_ms": summarize(event_ms)},
        "memory": {
            "peak_allocated": runtime.max_memory_allocated(),
            "peak_reserved": runtime.max_memory_reserved(),
        },
        "worker_sha256": sha256(Path(__file__)),
        "gate_sha256": sha256(args.gate.resolve()) if gate is not None else None,
        "loaded_source_sha256": source_hashes(),
    }
    write_json(rank_dir / "worker_result.json", result)
    if dist is not None:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(f"worker_result={rank_dir / 'worker_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
