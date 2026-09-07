#!/usr/bin/env python3
"""T-081～T-083 目标级性能 worker；准备实现，未经门禁不得执行。"""

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
from datetime import datetime


COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
TARGETS = {
    "constant-fold": ("T-081", "AU-joint-graph-constant-fold-uniform-value", "constant_fold_uniform_value"),
    "convert": ("T-081", "AU-joint-graph-pointless-convert", "pointless_convert"),
    "permute": ("T-082", "AU-joint-graph-pointless-permute-pair", "pointless_permute_pair"),
    "view": ("T-082", "AU-joint-graph-pointless-view-pair", "pointless_view_pair"),
    "all-gather": ("T-083", "AU-post-grad-bucket-all-gathers", "bucket_all_gather"),
    "all-reduce": ("T-083", "AU-post-grad-bucket-all-reduce", "bucket_all_reduce"),
    "reduce-scatter": ("T-083", "AU-post-grad-bucket-reduce-scatters", "bucket_reduce_scatter"),
}


def input_spec(unit: str) -> list[dict]:
    shapes = {
        "constant-fold": [(2, 4)], "convert": [(8,)],
        "permute": [(15, 7)], "view": [(15, 7)],
        "all-gather": [(64,), (64,), (64,)],
        "all-reduce": [(4, 384), (384, 512), (384, 512), (384, 256)],
        "reduce-scatter": [(4, 384), (384, 512), (384, 512), (384, 256)],
    }[unit]
    return [{"shape": list(shape), "dtype": "torch.float16" if unit == "convert" else "torch.float32"}
            for shape in shapes]


def read_gate(path: Path, unit: str, device: str) -> dict:
    """标准库门禁：复核记录必须绑定原始功能证据，不接受裸通过开关。"""
    gate = json.loads(path.read_text(encoding="utf-8"))
    task_id, unit_id, _ = TARGETS[unit]
    expected = {
        "task_id": task_id, "acceptance_unit_id": unit_id,
        "backend": "triton_experimental" if device == "npu" else "inductor-default",
        "pytorch_commit": COMMIT,
        "correctness": "passed", "target_rewrite": "confirmed",
        "graph_breaks": 0, "fallbacks": 0, "product_disabled": False,
        "measurement_workload": unit + "-community-shape",
        "world_size": 2 if task_id == "T-083" else 1,
        "input_spec": input_spec(unit),
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for key, value in expected.items():
        if gate.get(key) != value or type(gate.get(key)) is not type(value):
            raise ValueError(f"性能门禁 {key} 不符合要求：需要 {value!r}")
    if not gate.get("reviewed_at") or not gate.get("reviewer"):
        raise ValueError("门禁缺少复核时间/复核者")
    for key in ("gpu_reference", "target_functional"):
        record = gate.get(key, {})
        evidence = Path(record.get("path", ""))
        if not evidence.is_absolute():
            evidence = path.parent / evidence
        if not evidence.is_file() or hashlib.sha256(evidence.read_bytes()).hexdigest() != record.get("sha256"):
            raise ValueError(f"{key} 原件不存在或sha256不符")
        parsed = json.loads(evidence.read_text(encoding="utf-8"))
        if key == "gpu_reference":
            if parsed.get("status") != "valid-reference-suite" or parsed.get("suite_valid") is not True:
                raise ValueError("GPU reference suite无效")
            if not any(item.get("acceptance_unit_id") == unit_id and item.get("reference_valid") is True
                       for item in parsed.get("cases", [])):
                raise ValueError("GPU reference未包含本单元")
        else:
            # 目标功能补证必须覆盖实际待测输入，结构-only case不能签发此记录。
            for field in ("acceptance_unit_id", "backend", "pytorch_commit", "correctness",
                          "target_rewrite", "world_size", "measurement_workload", "graph_breaks",
                          "fallbacks", "product_disabled", "input_spec", "worker_sha256"):
                if parsed.get(field) != expected[field]:
                    raise ValueError(f"目标功能原件 {field} 与门禁不一致")
            if parsed.get("numerical_execution") is not True:
                raise ValueError("结构-only/FakeTensor不能作为性能数值门禁")
            sources = parsed.get("source_files", {})
            if not isinstance(sources, dict) or not sources:
                raise ValueError("目标功能原件缺少实际源码路径/sha256绑定")
            for source, digest in sources.items():
                actual_source = Path(source)
                if not actual_source.is_file() or hashlib.sha256(actual_source.read_bytes()).hexdigest() != digest:
                    raise ValueError("目标功能原件与当前源码不一致")
            if task_id == "T-083" and parsed.get("process_group_backend") not in {"nccl", "hccl"}:
                raise ValueError("通信功能补证必须是真实2rank NCCL/HCCL")
    return gate


def percentile(values, quantile):
    values = sorted(values)
    pos = (len(values) - 1) * quantile
    index = int(pos)
    return values[index] + (values[min(index + 1, len(values) - 1)] - values[index]) * (pos - index)


def main() -> None:
    process_started_at = datetime.now().astimezone().isoformat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", choices=TARGETS, required=True)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--device", choices=("cuda", "npu"), required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    gate = read_gate(args.gate, args.unit, args.device)
    if args.warmup < 1 or args.runs < 1:
        parser.error("warmup/runs必须为正整数")
    work_dir = Path(os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")).resolve()
    if Path.cwd().resolve() != work_dir:
        raise RuntimeError(f"测试必须从 {work_dir} 启动")
    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"
    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank_dir = output / f"rank-{rank}"
    rank_dir.mkdir(exist_ok=True)
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(rank_dir / "debug")
    os.environ["TORCH_TRACE"] = str(rank_dir / "trace")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(rank_dir / "inductor-cache")
    os.environ["TRITON_CACHE_DIR"] = str(rank_dir / "triton-cache")

    import torch
    from torch._inductor import config
    from torch._inductor.fx_passes import joint_graph

    if torch.version.git_version != COMMIT:
        raise RuntimeError("实际加载的PyTorch不是冻结commit")
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu
        register_inductor_npu()
        if _InductorNpuRegistry._loaded_backend != "triton_experimental":
            raise RuntimeError("实际NPU注册后端不符")
    runtime = getattr(torch, args.device)
    distributed = TARGETS[args.unit][0] == "T-083"
    if distributed and (world != 2 or runtime.device_count() < 2):
        raise RuntimeError("T083性能要求torchrun --nproc-per-node=2，至少2张可见设备；拒绝单rank/fake PG")
    if not distributed and world != 1:
        raise RuntimeError("非通信单元只允许单进程")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    runtime.set_device(local_rank)
    device = f"{args.device}:{local_rank}"
    import torch.distributed as dist
    if distributed:
        dist.init_process_group("hccl" if args.device == "npu" else "nccl")
        if dist.get_backend() not in {"nccl", "hccl"} or dist.get_world_size() != 2:
            raise RuntimeError("拒绝非真实2rank通信后端")
    torch.manual_seed(20260907)
    state = {"handler_calls": 0, "graph_changes": 0, "collective_counts": [], "post_grad_counts": []}
    _, unit_id, symbol = TARGETS[args.unit]

    collective_token = {"all-gather": "all_gather_into_tensor", "all-reduce": "all_reduce",
                        "reduce-scatter": "reduce_scatter_tensor"}.get(args.unit)

    def collective_count(graph):
        return sum(n.op == "call_function" and collective_token in str(n.target) for n in graph.nodes)

    def observed(original):
        @functools.wraps(original)
        def wrapper(first, *rest, **kwargs):
            graph = getattr(first, "graph", first)
            before = str(graph)
            before_count = collective_count(graph) if distributed else None
            index = state["handler_calls"]
            state["handler_calls"] += 1
            result = original(first, *rest, **kwargs)
            after = str(graph)
            state["graph_changes"] += int(before != after)
            if distributed:
                state["collective_counts"].append({"before": before_count, "after": collective_count(graph)})
            (rank_dir / f"target-{index}-before.txt").write_text(before, encoding="utf-8")
            (rank_dir / f"target-{index}-after.txt").write_text(after, encoding="utf-8")
            return result
        return wrapper

    settings = {"fx_graph_cache": False, "force_disable_caches": True}
    if args.unit == "constant-fold":
        joint_graph.constant_fold_uniform_value = observed(joint_graph.constant_fold_uniform_value)
        settings["joint_graph_constant_folding"] = args.mode == "on"
    elif not distributed:
        seen = set()
        for entries in joint_graph.patterns.patterns.values():
            for entry in entries:
                if id(entry) in seen or getattr(getattr(entry, "handler", None), "__name__", None) != symbol:
                    continue
                seen.add(id(entry))
                entry.handler = observed(entry.handler)
                if args.mode == "off":
                    entry.extra_check = lambda match: False
        if not seen:
            raise RuntimeError("未找到精确目标注册")
        settings["emulate_precision_casts"] = True
    else:
        from torch._inductor.fx_passes import bucketing, post_grad
        setattr(bucketing, symbol, observed(getattr(bucketing, symbol)))
        original_post_grad = post_grad.post_grad_passes

        @functools.wraps(original_post_grad)
        def observed_post_grad(gm, *rest, **kwargs):
            result = original_post_grad(gm, *rest, **kwargs)
            state["post_grad_counts"].append(collective_count(gm.graph))
            (rank_dir / "post-grad-final.txt").write_text(gm.code, encoding="utf-8")
            return result

        post_grad.post_grad_passes = observed_post_grad
        compile_fx = sys.modules.get("torch._inductor.compile_fx")
        if compile_fx is not None and getattr(compile_fx, "post_grad_passes", None) is original_post_grad:
            compile_fx.post_grad_passes = observed_post_grad
        settings.update(bucket_all_gathers_fx="none", bucket_all_reduces_fx="none",
                        bucket_reduce_scatters_fx="none", reorder_for_compute_comm_overlap=False,
                        bucket_all_gathers_bucket_mode="default", bucket_reduce_scatters_bucket_mode="default")
        control = {"all-gather": "bucket_all_gathers_fx", "all-reduce": "bucket_all_reduces_fx",
                   "reduce-scatter": "bucket_reduce_scatters_fx"}[args.unit]
        settings[control] = "all" if args.mode == "on" else "none"

    if args.unit == "constant-fold":
        def fn(x):
            a = torch.full(x.shape, 1, dtype=x.dtype, device=x.device)
            return x + (a - 1)
        inputs = (torch.randn(2, 4, device=device),)
    elif args.unit == "convert":
        def fn(x):
            y = torch.ops.prims.convert_element_type.default(x, torch.float32)
            return torch.ops.prims.convert_element_type.default(y, torch.float16)
        inputs = (torch.randn(8, device=device, dtype=torch.float16),)
    elif args.unit == "permute":
        def fn(x):
            return torch.ops.aten.permute.default(torch.ops.aten.permute.default(x, [1, 0]), [1, 0])
        inputs = (torch.randn(15, 7, device=device),)
    elif args.unit == "view":
        def fn(x):
            return torch.ops.aten.view.default(torch.ops.aten.view.default(x, [3, 5, 7]), [15, 7])
        inputs = (torch.randn(15, 7, device=device),)
    else:
        group = dist.distributed_c10d._get_default_group().group_name
        ops = torch.ops._c10d_functional
        if args.unit == "all-gather":
            def fn(a, b, c):
                return tuple(ops.wait_tensor(ops.all_gather_into_tensor(x, world, group)) for x in (a, b, c))
            inputs = tuple(torch.ones(64, device=device) * (rank + i + 1) for i in range(3))
        else:
            def fn(x, w, a, b):
                if args.unit == "reduce-scatter":
                    aa = ops.reduce_scatter_tensor(a.to(torch.bfloat16), "sum", world, group)
                    bb = ops.reduce_scatter_tensor(b.to(torch.bfloat16), "sum", world, group)
                else:
                    aa, bb = ops.all_reduce(a, "sum", group), ops.all_reduce(b, "sum", group)
                return torch.mm(x, w), ops.wait_tensor(aa), ops.wait_tensor(bb)
            inputs = tuple(torch.ones(*shape, device=device) * (rank + 1) for shape in
                           ((4, 384), (384, 512), (384, 512), (384, 256)))

    # 数值参照在目标修改前图语义上执行；实际算子/进程组与待测输入相同。
    expected = fn(*inputs)
    actual_spec = [{"shape": list(x.shape), "dtype": str(x.dtype)} for x in inputs]
    if actual_spec != gate["input_spec"] or not all(x.is_contiguous() for x in inputs):
        raise RuntimeError("运行输入与已复核的连续输入合同不一致")
    with config.patch(settings):
        if args.unit in {"convert", "permute", "view"}:
            from torch.fx.experimental.proxy_tensor import make_fx
            gm = make_fx(fn)(*inputs)
            (rank_dir / "community-before.txt").write_text(gm.code, encoding="utf-8")
            joint_graph.joint_graph_passes(gm)
            (rank_dir / "community-after.txt").write_text(gm.code, encoding="utf-8")
            torch.testing.assert_close(gm(*inputs), expected)
            fn = gm
        options = {"npu_backend": "triton_experimental"} if args.device == "npu" else None
        started = time.perf_counter()
        compiled = torch.compile(fn, fullgraph=True, dynamic=args.unit == "constant-fold", options=options)
        actual = compiled(*inputs)
        runtime.synchronize()
        compile_ms = (time.perf_counter() - started) * 1000
        torch.testing.assert_close(actual, expected)
        if args.unit in {"view", "permute"}:
            if actual.stride() != expected.stride() or actual.untyped_storage().data_ptr() != inputs[0].untyped_storage().data_ptr():
                raise RuntimeError("view/permute stride或alias合同失败")
        if args.mode == "on" and state["graph_changes"] < 1:
            raise RuntimeError("目标ON没有实际图改写，不计时")
        if args.mode == "off" and state["handler_calls"] != 0:
            raise RuntimeError("目标OFF仍进入handler，不计时")
        if distributed:
            expected_before = 3 if args.unit == "all-gather" else 2
            expected_after = 1 if args.mode == "on" else expected_before
            if state["post_grad_counts"] != [expected_after]:
                raise RuntimeError("post-grad实际collective数不满足目标OFF/ON合同")
            if args.mode == "on" and {"before": expected_before, "after": 1} not in state["collective_counts"]:
                raise RuntimeError("未捕获精确3→1/2→1目标分桶，拒绝计时")
        for _ in range(args.warmup):
            compiled(*inputs)
        runtime.synchronize()
        runtime.reset_peak_memory_stats()
        samples = {"host_ms": [], "event_ms": []}
        for _ in range(args.runs):
            if distributed:
                dist.barrier()
            runtime.synchronize()
            start, end = runtime.Event(enable_timing=True), runtime.Event(enable_timing=True)
            before = time.perf_counter()
            start.record()
            compiled(*inputs)
            end.record()
            runtime.synchronize()
            host = (time.perf_counter() - before) * 1000
            event = start.elapsed_time(end)
            samples["host_ms"].append(host)
            samples["event_ms"].append(event)
        memory = {"allocated": runtime.max_memory_allocated(), "reserved": runtime.max_memory_reserved()}
    source_files = {}
    for name, module in list(sys.modules.items()):
        if name.startswith(("torch._inductor", "torch_npu._inductor", "triton")) and getattr(module, "__file__", None):
            path = Path(module.__file__)
            if path.is_file():
                source_files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    torch_root = Path(torch.__file__).resolve().parents[1]
    git_state = subprocess.run(["git", "-C", str(torch_root), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    record = {
        "acceptance_unit_id": unit_id, "mode": args.mode, "correctness": "passed",
        "pytorch_commit": torch.version.git_version, "pytorch_worktree_status": git_state,
        "backend": gate["backend"], "selected_before_import": True,
        "pid": os.getpid(), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "process_started_at": process_started_at,
        "torch_version": torch.__version__,
        "torch_npu_version": getattr(sys.modules.get("torch_npu"), "__version__", None),
        "torch_npu_git_version": getattr(getattr(sys.modules.get("torch_npu"), "version", None), "git_version", None),
        "triton_version": getattr(__import__("triton"), "__version__", None),
        "rank": rank, "world_size": world, "process_group_backend": dist.get_backend() if distributed else None,
        "input_contract": [{"shape": list(x.shape), "stride": list(x.stride()), "dtype": str(x.dtype), "device": str(x.device)} for x in inputs],
        "state": state, "compile_ms": compile_ms, "samples": samples, "memory": memory,
        "loaded_source_sha256": source_files,
        "gate_sha256": hashlib.sha256(args.gate.read_bytes()).hexdigest(),
    }
    (rank_dir / "worker_result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ranks = [None] * world
    if distributed:
        dist.all_gather_object(ranks, record)
    else:
        ranks[0] = record
    if rank == 0:
        maxima = {metric: [max(r["samples"][metric][i] for r in ranks) for i in range(args.runs)]
                  for metric in samples}
        summary = {"acceptance_unit_id": unit_id, "mode": args.mode, "world_size": world,
                   "rank_max_samples": maxima, "timing": {k: {"p50": percentile(v, .5), "p99": percentile(v, .99)} for k, v in maxima.items()},
                   "rank_peak_memory": [r["memory"] for r in ranks], "compile_ms_per_rank": [r["compile_ms"] for r in ranks]}
        (output / "arm_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
