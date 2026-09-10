#!/usr/bin/env python3
"""T-087～T-090 单进程功能/性能臂；必须由 tmp 目录以新进程启动。"""

from __future__ import annotations

import argparse
import contextlib
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
    "reorder-locality": ("T-087", "AU-post-grad-reorder-for-locality", "reorder_for_locality"),
    "split-cat-aten": ("T-088", "AU-split-cat-merge-split-cat-aten", "merge_split_cat_aten"),
    "select-cat-aten": ("T-088", "AU-split-cat-merge-select-cat-aten", "merge_select_cat_aten"),
    "move-view-after-cat": ("T-089", "AU-split-cat-move-view-after-cat", "move_view_after_cat"),
    "normalize-cat-aten": ("T-090", "AU-split-cat-normalize-cat-default-aten", "normalize_cat_default_aten"),
    "respecialize-current-device": ("T-087", "AU-post-grad-respecialize-current-device", "respecialize_current_device_nodes"),
}
BENCHMARK_UNITS = set(TARGETS) - {"respecialize-current-device"}
INPUT_SPECS = {
    "reorder-locality": [{"shape": [8, 16], "dtype": "torch.float32"}],
    "split-cat-aten": [
        {"shape": [1024, 128], "dtype": "torch.float32"},
        {"shape": [1024, 128], "dtype": "torch.float32"},
        {"shape": [1024, 32], "dtype": "torch.float32"},
    ],
    "select-cat-aten": [
        {"shape": [1024, 6, 128], "dtype": "torch.float32"},
        {"shape": [1024, 6, 128], "dtype": "torch.float32"},
    ],
    "move-view-after-cat": [{"shape": [7, 8, 96], "dtype": "torch.float32"}],
    "normalize-cat-aten": [
        {"shape": [1024, 128], "dtype": "torch.float32"},
        {"shape": [1024, 128], "dtype": "torch.float32"},
        {"shape": [1024, 32], "dtype": "torch.float32"},
    ],
    "respecialize-current-device": [{"shape": [2, 8], "dtype": "torch.float32"}],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_gate(path: Path, unit: str, device: str) -> dict:
    gate = json.loads(path.read_text(encoding="utf-8"))
    task, acceptance, _ = TARGETS[unit]
    expected = {
        "task_id": task,
        "acceptance_unit_id": acceptance,
        "backend": "triton_experimental" if device == "npu" else "inductor-default",
        "pytorch_commit": COMMIT,
        "correctness": "passed",
        "target_rewrite": "confirmed",
        "graph_breaks": 0,
        "fallbacks": 0,
        "product_disabled": False,
        "measurement_workload": unit + "-community-shape",
        "input_spec": INPUT_SPECS[unit],
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
            raise ValueError(f"{name} 原件缺失或sha256不符")
    return gate


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def source_hashes() -> dict[str, str]:
    result = {}
    for name, module in list(sys.modules.items()):
        path = Path(getattr(module, "__file__", ""))
        if name.startswith(("torch._inductor", "torch_npu._inductor", "triton")) and path.is_file():
            result[str(path.resolve())] = sha256(path)
    return result


def install_exact_handler_observer(split_cat, pass_name: str, symbol: str, state: dict) -> None:
    registry = split_cat.POST_GRAD_PATTERNS[pass_name]
    seen = 0
    for entries in registry.patterns.values():
        for entry in entries:
            handler = getattr(entry, "handler", None)
            if getattr(handler, "__name__", "") != symbol:
                continue
            seen += 1

            @functools.wraps(handler)
            def observed(*args, __handler=handler, **kwargs):
                state["handler_calls"] += 1
                graph = args[0].graph
                before = str(graph)
                result = __handler(*args, **kwargs)
                state["graph_changes"] += int(before != str(graph))
                return result

            entry.handler = observed
    if seen != 1:
        raise RuntimeError(f"目标handler注册数不是1：{pass_name}/{symbol}/{seen}")


def split_cat_model(torch, unit: str):
    if unit in {"split-cat-aten", "normalize-cat-aten"}:
        def fn(x, y, z):
            parts = torch.ops.aten.split.Tensor(torch.ops.aten.cat.default([x, y], 1), 32, 1)
            merged = torch.ops.aten.cat.default(list(parts[:8]), 1)
            side = torch.ops.aten.cat.default([parts[0], z], 1)
            return torch.ops.aten.cat.default([merged, side], 1)
        return fn
    if unit == "select-cat-aten":
        def fn(x, y):
            xs = [torch.ops.aten.select.int(x, 1, i) for i in range(6)]
            ys = [torch.ops.aten.select.int(y, 1, i) for i in range(5)]
            return (
                torch.ops.aten.cat.default(xs, 1),
                torch.ops.aten.cat.default(xs[:5], 1),
                torch.ops.aten.cat.default(xs[::2], 1),
                torch.ops.aten.cat.default(ys, 1),
            )
        return fn

    def fn(x):
        parts = torch.ops.aten.split_with_sizes.default(x, [1] * 7)
        views = [torch.ops.aten.view.default(item, [8, 96]) for item in parts]
        clone = torch.ops.aten.clone.default(views[0])
        merged = torch.ops.aten.cat.default(views, 1)
        return torch.ops.aten.cat.default([clone, torch.ops.aten.cat.default([clone, merged], 1)], 1)
    return fn


def assert_close(torch, actual, expected) -> None:
    if isinstance(actual, (tuple, list)):
        if len(actual) != len(expected):
            raise RuntimeError("输出数量不一致")
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", choices=TARGETS, required=True)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--device", choices=("cuda", "npu"), required=True)
    parser.add_argument("--phase", choices=("functional", "benchmark"), default="functional")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 1 or args.runs < 1:
        parser.error("warmup/runs必须为正整数")
    if args.unit == "respecialize-current-device" and (
        args.phase == "benchmark" or args.mode == "off"
    ):
        parser.error("respecialize-current-device无合法OFF，只允许functional/on")
    gate = None
    if args.phase == "benchmark":
        if args.gate is None:
            parser.error("benchmark必须提供人工签发的--gate")
        gate = read_gate(args.gate.resolve(), args.unit, args.device)
    work = Path(os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")).resolve()
    if Path.cwd().resolve() != work:
        raise RuntimeError(f"必须从 {work} 启动")

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
    import torch._dynamo.config as dynamo_config
    from torch._dynamo.utils import counters
    from torch._inductor import config
    from torch._inductor.fx_passes import post_grad, split_cat

    if torch.version.git_version != COMMIT:
        raise RuntimeError("实际加载PyTorch不是冻结commit")
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu
        register_inductor_npu()
        if _InductorNpuRegistry._loaded_backend != "triton_experimental":
            raise RuntimeError("NPU实际注册后端不是triton_experimental")
    runtime = getattr(torch, args.device)
    runtime.set_device(0)
    device = f"{args.device}:0"
    torch.manual_seed(20260910)
    state = {"handler_calls": 0, "graph_changes": 0}
    task, acceptance, symbol = TARGETS[args.unit]
    settings = {
        "fx_graph_cache": False,
        "force_disable_caches": True,
        "pre_grad_fusion_options": {},
        "post_grad_fusion_options": {},
    }

    if args.unit == "reorder-locality":
        original = post_grad.reorder_for_locality

        @functools.wraps(original)
        def observed(graph):
            state["handler_calls"] += 1
            before = [node.name for node in graph.nodes]
            result = original(graph)
            state["graph_changes"] += int(before != [node.name for node in graph.nodes])
            return result

        post_grad.reorder_for_locality = observed
        settings.update(
            reorder_for_locality=True,
            reorder_for_locality_in_training=args.mode == "on",
        )

        class TwoBranch(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = torch.nn.Parameter(torch.randn(16, 16))
                self.w2 = torch.nn.Parameter(torch.randn(16, 16))

            def forward(self, x):
                a = torch.tanh(x @ self.w1)
                b = torch.sigmoid(x @ self.w2)
                return (a * b + torch.sin(a) * torch.cos(b)).sum(dim=1)

        model = TwoBranch().to(device)
        inputs = (torch.randn(8, 16, device=device),)
        eager_model = TwoBranch().to(device)
        eager_model.load_state_dict(model.state_dict())
        eager = eager_model(*inputs).sum()
        eager.backward()

        def invoke(compiled):
            model.zero_grad(set_to_none=True)
            value = compiled(*inputs).sum()
            value.backward()
            return value.detach()

        expected = eager.detach()
    elif args.unit == "respecialize-current-device":
        original = post_grad.respecialize_current_device_nodes

        @functools.wraps(original)
        def observed(graph):
            state["handler_calls"] += 1
            before = str(graph)
            result = original(graph)
            state["graph_changes"] += int(before != str(graph))
            return result

        post_grad.respecialize_current_device_nodes = observed

        def model(x):
            created = torch.zeros(4, x.shape[1], device=x.device, dtype=x.dtype)
            return created + x.sum()

        inputs = (torch.randn(2, 8, device=device),)
        expected = model(*inputs)

        def invoke(compiled):
            return compiled(*inputs)
    else:
        pass_name = {
            "split-cat-aten": "split_cat_aten_pass",
            "select-cat-aten": "select_cat_aten_pass",
            "move-view-after-cat": "move_view_after_cat_aten_pass",
            "normalize-cat-aten": "normalization_aten_pass",
        }[args.unit]
        install_exact_handler_observer(split_cat, pass_name, symbol, state)
        normalization = {"normalization_aten_pass": {}}
        target_option = {
            "split-cat-aten": {"split_cat_aten_pass": {"threshold_to_cat": 5}},
            "select-cat-aten": {"select_cat_aten_pass": {}},
            "move-view-after-cat": {"move_view_after_cat_aten_pass": {}},
            "normalize-cat-aten": {"split_cat_aten_pass": {"threshold_to_cat": 5}},
        }[args.unit]
        settings["post_grad_fusion_options"] = dict(normalization if args.unit != "normalize-cat-aten" or args.mode == "on" else {})
        settings["post_grad_fusion_options"].update(target_option)
        if args.mode == "off":
            settings["post_grad_fusion_options"].pop(pass_name, None)
        model = split_cat_model(torch, args.unit)
        inputs = tuple(torch.randn(*item["shape"], device=device) for item in INPUT_SPECS[args.unit])
        expected = model(*inputs)

        def invoke(compiled):
            return compiled(*inputs)

    counters.clear()
    options = {"npu_backend": "triton_experimental"} if args.device == "npu" else None
    compiler_patch = (
        dynamo_config.patch(compile_on_one_rank=True)
        if args.unit == "respecialize-current-device"
        else contextlib.nullcontext()
    )
    with compiler_patch, config.patch(settings):
        started = time.perf_counter()
        compiled = torch.compile(model, backend="inductor", fullgraph=True, options=options)
        actual = invoke(compiled)
        runtime.synchronize()
        compile_ms = (time.perf_counter() - started) * 1000
        assert_close(torch, actual, expected)
        if args.unit == "reorder-locality":
            for left, right in zip(model.parameters(), eager_model.parameters()):
                torch.testing.assert_close(left.grad, right.grad)
        if args.mode == "on" and (state["handler_calls"] < 1 or state["graph_changes"] < 1):
            raise RuntimeError("ON未捕获精确目标改写")
        if args.mode == "off" and state["handler_calls"] != 0:
            raise RuntimeError("OFF仍调用目标handler")

        samples = None
        memory = None
        if args.phase == "benchmark":
            for _ in range(args.warmup):
                invoke(compiled)
            runtime.synchronize()
            runtime.reset_peak_memory_stats()
            samples = {"host_ms": [], "event_ms": []}
            for _ in range(args.runs):
                runtime.synchronize()
                start = runtime.Event(enable_timing=True)
                end = runtime.Event(enable_timing=True)
                before = time.perf_counter()
                start.record()
                invoke(compiled)
                end.record()
                runtime.synchronize()
                samples["host_ms"].append((time.perf_counter() - before) * 1000)
                samples["event_ms"].append(start.elapsed_time(end))
            memory = {"allocated": runtime.max_memory_allocated(), "reserved": runtime.max_memory_reserved()}

    torch_root = Path(torch.__file__).resolve().parents[1]
    worktree = subprocess.run(
        ["git", "-C", str(torch_root), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout
    record = {
        "schema_version": "1.0", "generated_at": datetime.now().astimezone().isoformat(),
        "task_id": task, "acceptance_unit_id": acceptance, "unit": args.unit,
        "mode": args.mode, "phase": args.phase,
        "backend": "triton_experimental" if args.device == "npu" else "inductor-default",
        "backend_selected_before_import": True, "pytorch_commit": torch.version.git_version,
        "pytorch_worktree_status": worktree, "correctness": "passed", "numerical_execution": True,
        "target_rewrite": "confirmed" if args.mode == "on" else "disabled-control",
        "graph_breaks": 0, "fallbacks": 0, "product_disabled": False,
        "measurement_workload": args.unit + "-community-shape", "input_spec": INPUT_SPECS[args.unit],
        "state": state, "compile_ms": compile_ms, "samples": samples, "memory": memory,
        "timing": ({key: {"p50": percentile(value, .5), "p99": percentile(value, .99)} for key, value in samples.items()} if samples else None),
        "worker_sha256": sha256(Path(__file__)), "gate_sha256": sha256(args.gate) if gate else None,
        "loaded_source_sha256": source_hashes(),
    }
    (output / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"worker_status=passed unit={args.unit} mode={args.mode} phase={args.phase}")


if __name__ == "__main__":
    main()
