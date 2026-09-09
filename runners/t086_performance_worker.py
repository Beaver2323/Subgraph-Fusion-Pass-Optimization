#!/usr/bin/env python3
"""T-086 reinplace目标级功能/性能worker；每次调用只运行一个独立OFF或ON进程。"""

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
TASK_ID = "T-086"
UNIT_ID = "AU-post-grad-reinplace-inplaceable-ops"
WORKLOAD_ID = "reinplace-index-put-community-shape"
TARGETS = {"reinplace-index-put": (TASK_ID, UNIT_ID, "reinplace_inplaceable_ops")}


def input_spec() -> list[dict[str, object]]:
    return [
        {"shape": [1024], "dtype": "torch.float32"},
        {"shape": [10], "dtype": "torch.int64"},
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def loaded_source_hashes() -> dict[str, str]:
    result = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith(("torch._inductor", "torch_npu._inductor", "triton")):
            continue
        source = getattr(module, "__file__", None)
        if source and Path(source).is_file():
            result[str(Path(source).resolve())] = sha256(Path(source))
    return result


def read_gate(path: Path, device: str) -> dict:
    """性能门禁必须绑定GPU reference、目标设备功能原件和当前worker。"""
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
        "fallback_scope": "graph-break-or-CPU-fallback-only",
        "allowed_device_lowering": (
            "registered-IndexPutFallback-extern" if device == "npu" else "none"
        ),
        "product_disabled": False,
        "measurement_workload": WORKLOAD_ID,
        "world_size": 1,
        "input_spec": input_spec(),
        "worker_sha256": sha256(Path(__file__)),
    }
    for key, value in expected.items():
        if gate.get(key) != value or type(gate.get(key)) is not type(value):
            raise ValueError(f"性能门禁 {key} 不符合要求：需要 {value!r}")
    if not gate.get("reviewed_at") or not gate.get("reviewer"):
        raise ValueError("门禁缺少复核时间或复核者")

    for name in ("gpu_reference", "target_functional"):
        record = gate.get(name, {})
        evidence = Path(record.get("path", ""))
        if not evidence.is_absolute():
            evidence = path.parent / evidence
        if not evidence.is_file() or sha256(evidence) != record.get("sha256"):
            raise ValueError(f"{name}原件不存在或sha256不符")
        parsed = json.loads(evidence.read_text(encoding="utf-8"))
        if name == "gpu_reference":
            raw_valid = (
                parsed.get("status") == "valid-reference-suite"
                and parsed.get("suite_valid") is True
                and any(
                    item.get("acceptance_unit_id") == UNIT_ID
                    and item.get("reference_valid") is True
                    for item in parsed.get("cases", [])
                )
            )
            reviewed_valid = (
                str(parsed.get("review_status", "")).startswith("accepted")
                and parsed.get("task_id") == TASK_ID
                and parsed.get("pytorch_commit") == COMMIT
                and UNIT_ID in parsed.get("acceptance_units", [])
            )
            if not (raw_valid or reviewed_valid):
                raise ValueError("GPU reference原件无效或未覆盖T-086单元")
        else:
            for field in (
                "task_id",
                "acceptance_unit_id",
                "backend",
                "pytorch_commit",
                "correctness",
                "target_rewrite",
                "graph_breaks",
                "fallbacks",
                "fallback_scope",
                "allowed_device_lowering",
                "product_disabled",
                "measurement_workload",
                "world_size",
                "input_spec",
                "worker_sha256",
            ):
                if parsed.get(field) != expected[field]:
                    raise ValueError(f"目标功能原件 {field} 与性能门禁不一致")
            if parsed.get("numerical_execution") is not True:
                raise ValueError("结构-only/FakeTensor不能签发性能门禁")
            arms = parsed.get("arms", {})
            if set(arms) != {"off", "on"}:
                raise ValueError("目标功能原件必须同时包含独立OFF与ON")
            if arms["off"].get("target_rewrite") != "disabled-control":
                raise ValueError("OFF功能原件未证明精确关闭")
            if arms["on"].get("target_rewrite") != "confirmed":
                raise ValueError("ON功能原件未证明目标改写")
            sources = parsed.get("source_files", {})
            if not isinstance(sources, dict) or not sources:
                raise ValueError("目标功能原件缺少实际源码sha256")
            for source, digest in sources.items():
                source_path = Path(source)
                if not source_path.is_file() or sha256(source_path) != digest:
                    raise ValueError("目标功能原件绑定源码已变化")
    return gate


def graph_counts(graph) -> dict[str, int]:
    import torch

    return {
        "index_put": len(
            graph.find_nodes(
                op="call_function", target=torch.ops.aten.index_put.default
            )
        ),
        "index_put_": len(
            graph.find_nodes(
                op="call_function", target=torch.ops.aten.index_put_.default
            )
        ),
        "copy_": len(
            graph.find_nodes(op="call_function", target=torch.ops.aten.copy_.default)
        ),
    }


def assert_target_state(mode: str, state: dict) -> None:
    snapshots = state["snapshots"]
    if not snapshots:
        raise RuntimeError("未进入reinplace_inplaceable_ops，拒绝签发功能或性能结果")
    if mode == "on":
        if not any(
            item["before"]["index_put"] >= 1
            and item["after"]["index_put"] < item["before"]["index_put"]
            and item["after"]["index_put_"] > item["before"]["index_put_"]
            for item in snapshots
        ):
            raise RuntimeError("ON未观察到index_put到index_put_的精确改写")
    else:
        if any(item["before"] != item["after"] for item in snapshots):
            raise RuntimeError("OFF仍改变了目标pass图，拒绝计时")


def main() -> None:
    process_started_at = datetime.now().astimezone().isoformat()
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
    if args.warmup < 1 or args.runs < 1:
        parser.error("warmup和runs必须为正整数")
    if args.phase == "benchmark" and args.gate is None:
        parser.error("benchmark阶段必须提供--gate")

    work_dir = Path(
        os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")
    ).resolve()
    if Path.cwd().resolve() != work_dir:
        raise RuntimeError(f"测试必须从 {work_dir} 启动")

    # backend选择发生在导入torch/torch_npu之前；每个arm由launcher创建新进程。
    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"
    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(output / "debug")
    os.environ["TORCH_TRACE"] = str(output / "trace")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(output / "inductor-cache")
    os.environ["TRITON_CACHE_DIR"] = str(output / "triton-cache")

    import torch
    from torch._inductor import config
    from torch._inductor.fx_passes import post_grad

    if torch.version.git_version != COMMIT:
        raise RuntimeError("实际加载PyTorch不是T-086冻结commit")
    backend = "inductor-default"
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu

        register_inductor_npu()
        if _InductorNpuRegistry._loaded_backend != "triton_experimental":
            raise RuntimeError("实际NPU注册后端不是triton_experimental")
        backend = "triton_experimental"
    runtime = getattr(torch, args.device)
    if not runtime.is_available() or runtime.device_count() < 1:
        raise RuntimeError(f"{args.device}实际设备不可用")
    runtime.set_device(0)
    device = f"{args.device}:0"
    if args.phase == "benchmark":
        gate = read_gate(args.gate.resolve(), args.device)
    else:
        gate = None

    state = {"pass_invocations": 0, "graph_changes": 0, "snapshots": []}
    original = post_grad.reinplace_inplaceable_ops

    @functools.wraps(original)
    def observed(fake_tensor_updater, graph):
        index = state["pass_invocations"]
        state["pass_invocations"] += 1
        before_text = str(graph)
        before = graph_counts(graph)
        (output / f"target-{index}-before.txt").write_text(
            before_text, encoding="utf-8"
        )
        result = None
        if args.mode == "on":
            result = original(fake_tensor_updater, graph)
        after_text = str(graph)
        after = graph_counts(graph)
        state["graph_changes"] += int(before_text != after_text)
        state["snapshots"].append({"before": before, "after": after})
        (output / f"target-{index}-after.txt").write_text(
            after_text, encoding="utf-8"
        )
        return result

    post_grad.reinplace_inplaceable_ops = observed

    def fn(x, idx):
        src = torch.ones(idx.size(0), device=x.device)
        x.index_put_((idx,), src)
        return x.expand((2, x.shape[0]))

    torch.manual_seed(20260909)
    base = torch.randn(1024, device=device)
    idx = torch.arange(10, device=device)
    eager_input = base.clone()
    expected = fn(eager_input, idx)
    compiled_input = base.clone()
    actual_spec = [
        {"shape": list(compiled_input.shape), "dtype": str(compiled_input.dtype)},
        {"shape": list(idx.shape), "dtype": str(idx.dtype)},
    ]
    if actual_spec != input_spec():
        raise RuntimeError("运行输入与社区shape/dtype合同不一致")

    settings = {"fx_graph_cache": False, "force_disable_caches": True}
    options = {"npu_backend": "triton_experimental"} if args.device == "npu" else None
    with config.patch(settings):
        started = time.perf_counter()
        compiled = torch.compile(fn, fullgraph=True, options=options)
        actual = compiled(compiled_input, idx)
        runtime.synchronize()
        compile_ms = (time.perf_counter() - started) * 1000
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(compiled_input, eager_input)
        if actual.stride() != expected.stride():
            raise RuntimeError("expand输出stride与eager不一致")
        if actual.untyped_storage().data_ptr() != compiled_input.untyped_storage().data_ptr():
            raise RuntimeError("正例输出没有保持对mutated input的alias")
        assert_target_state(args.mode, state)

        if args.phase == "functional":
            # 社区负例在同一功能进程中独立编译：原input仍被返回，必须阻止
            # index_put.default被错误地reinplace。它不参与性能采样。
            negative_snapshot_start = len(state["snapshots"])

            def negative_fn(x, index):
                updated = x.index_put((index,), torch.ones(index.size(0), device=x.device))
                return x, updated

            negative_base = base.clone()
            negative_eager_input = negative_base.clone()
            negative_expected = negative_fn(negative_eager_input, idx)
            negative_compiled_input = negative_base.clone()
            negative_compiled = torch.compile(
                negative_fn, fullgraph=True, options=options
            )
            negative_actual = negative_compiled(negative_compiled_input, idx)
            runtime.synchronize()
            torch.testing.assert_close(negative_actual, negative_expected)
            torch.testing.assert_close(negative_compiled_input, negative_eager_input)
            negative_snapshots = state["snapshots"][negative_snapshot_start:]
            if not negative_snapshots:
                raise RuntimeError("负例未进入目标pass")
            if args.mode == "on" and any(
                item["after"]["index_put_"] > item["before"]["index_put_"]
                or item["after"]["index_put"] < item["before"]["index_put"]
                for item in negative_snapshots
            ):
                raise RuntimeError("活跃input负例被错误reinplace")
            state["negative_guard"] = {
                "status": "passed",
                "snapshot_count": len(negative_snapshots),
                "input_preserved": True,
            }
            samples = None
            memory = None
        else:
            for _ in range(args.warmup):
                compiled(compiled_input, idx)
            runtime.synchronize()
            runtime.reset_peak_memory_stats()
            samples = {"host_ms": [], "event_ms": []}
            for _ in range(args.runs):
                runtime.synchronize()
                start = runtime.Event(enable_timing=True)
                end = runtime.Event(enable_timing=True)
                before = time.perf_counter()
                start.record()
                compiled(compiled_input, idx)
                end.record()
                runtime.synchronize()
                samples["host_ms"].append((time.perf_counter() - before) * 1000)
                samples["event_ms"].append(start.elapsed_time(end))
            memory = {
                "allocated": runtime.max_memory_allocated(),
                "reserved": runtime.max_memory_reserved(),
            }

    torch_root = Path(torch.__file__).resolve().parents[1]
    git_state = subprocess.run(
        ["git", "-C", str(torch_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    source_files = loaded_source_hashes()
    common = {
        "schema_version": "1.0",
        "task_id": TASK_ID,
        "acceptance_unit_id": UNIT_ID,
        "unit": args.unit,
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
        "fallback_scope": "graph-break-or-CPU-fallback-only",
        "allowed_device_lowering": (
            "registered-IndexPutFallback-extern" if args.device == "npu" else "none"
        ),
        "product_disabled": False,
        "measurement_workload": WORKLOAD_ID,
        "world_size": 1,
        "input_spec": input_spec(),
        "input_contract": [
            {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }
            for tensor in (compiled_input, idx)
        ],
        "state": state,
        "compile_ms": compile_ms,
        "worker_sha256": sha256(Path(__file__)),
        "loaded_source_sha256": source_files,
        "torch_version": torch.__version__,
        "torch_npu_version": getattr(sys.modules.get("torch_npu"), "__version__", None),
        "triton_version": getattr(__import__("triton"), "__version__", None),
        "physical_devices": os.environ.get(
            "ASCEND_RT_VISIBLE_DEVICES" if args.device == "npu" else "CUDA_VISIBLE_DEVICES"
        ),
        "pid": os.getpid(),
        "process_started_at": process_started_at,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    if args.phase == "functional":
        write_json(output / "functional_result.json", common)
        return

    record = {
        **common,
        "gate_sha256": sha256(args.gate.resolve()),
        "samples": samples,
        "memory": memory,
    }
    write_json(output / "worker_result.json", record)
    write_json(
        output / "arm_summary.json",
        {
            "acceptance_unit_id": UNIT_ID,
            "mode": args.mode,
            "world_size": 1,
            "timing": {
                metric: {
                    "p50": percentile(values, 0.50),
                    "p99": percentile(values, 0.99),
                }
                for metric, values in samples.items()
            },
            "rank_peak_memory": [memory],
            "compile_ms_per_rank": [compile_ms],
        },
    )


if __name__ == "__main__":
    main()
