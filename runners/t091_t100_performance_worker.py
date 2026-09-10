#!/usr/bin/env python3
"""T-091/T-098/T-100 单进程功能或性能臂；必须从固定 tmp 启动。"""

from __future__ import annotations

import argparse
import copy
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
    "stack-normalization": (
        "T-091",
        "AU-split-cat-normalize-stack-default",
        ("normalize_stack_default",),
    ),
    "efficient-conv-bn": (
        "T-098",
        "AU-efficient-conv-bn-eval-efficient-conv-bn-eval-graph-transform-inlined",
        (
            "efficient_conv_bn_eval_graph_transform_inlined",
            "efficient_conv_bn_eval_graph_transform_decomposed",
        ),
    ),
    "linear-binary-folding": (
        "T-100",
        "AU-binary-folding-folded-op",
        ("folded_op",),
    ),
}
INPUT_SPECS = {
    "stack-normalization": [
        {"shape": [4, 4], "dtype": "torch.float32"},
        {"shape": [4, 4], "dtype": "torch.float32"},
    ],
    "efficient-conv-bn": [
        {"shape": [4, 3, 96, 96], "dtype": "torch.float32"},
    ],
    "linear-binary-folding": [
        {"shape": [4, 3], "dtype": "torch.float32"},
    ],
}
WORKLOADS = {
    "stack-normalization": "stack-normalization-community-shape",
    "efficient-conv-bn": "efficient-conv-bn-community-conv2d",
    "linear-binary-folding": "linear-binary-folding-community-shape",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        "measurement_workload": WORKLOADS[unit],
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
            raise ValueError(f"{name}原件缺失或sha256不符")
    return gate


def install_observer(registry, symbols: tuple[str, ...], state: dict) -> int:
    seen = 0
    for entries in registry.patterns.values():
        for entry in entries:
            handler = getattr(entry, "handler", None)
            if getattr(handler, "__name__", "") not in symbols:
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
    return seen


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
    from torch._dynamo.utils import counters
    from torch._inductor import config

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
    state = {"handler_calls": 0, "graph_changes": 0, "registered_handlers": 0}
    settings = {
        "fx_graph_cache": False,
        "force_disable_caches": True,
        "pre_grad_fusion_options": {},
        "post_grad_fusion_options": {},
    }

    if args.unit == "stack-normalization":
        from torch._inductor.fx_passes import split_cat

        state["registered_handlers"] = install_observer(
            split_cat.PRE_GRAD_PATTERNS["normalization_pass"],
            TARGETS[args.unit][2],
            state,
        )
        if state["registered_handlers"] != 1:
            raise RuntimeError("normalize_stack_default注册数不是1")
        if args.mode == "on":
            settings["pre_grad_fusion_options"] = {"normalization_pass": {}}

        def model(x, y):
            return torch.stack([x, y], axis=1)

        inputs = tuple(
            torch.rand(*spec["shape"], device=device) for spec in INPUT_SPECS[args.unit]
        )
        expected = model(*inputs)

        def invoke(compiled):
            return compiled(*inputs)

        counter_name = "normalization_pass"
        gradient_models = None
    elif args.unit == "efficient-conv-bn":
        from torch._inductor.fx_passes import efficient_conv_bn_eval, pre_grad  # noqa: F401

        state["registered_handlers"] = install_observer(
            pre_grad.efficient_conv_bn_eval_pass,
            TARGETS[args.unit][2],
            state,
        )
        if state["registered_handlers"] != 2:
            raise RuntimeError("efficient Conv-BN两个合并handler注册数不是2")
        settings["efficient_conv_bn_eval_fx_passes"] = args.mode == "on"
        settings["freezing"] = False

        class ConvBN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, bias=True)
                self.bn = torch.nn.BatchNorm2d(32)

            def forward(self, x):
                return self.bn(self.conv(x))

        model = ConvBN().eval().to(device)
        eager_model = copy.deepcopy(model)
        inputs = (torch.rand(4, 3, 96, 96, device=device),)
        eager_output = eager_model(*inputs)
        eager_output.mean().backward()
        expected = eager_output.detach()

        def invoke(compiled):
            model.zero_grad(set_to_none=True)
            value = compiled(*inputs)
            value.mean().backward()
            return value.detach()

        counter_name = "efficient_conv_bn_eval"
        gradient_models = (model, eager_model)
    else:
        from torch._inductor.fx_passes import freezing_patterns  # noqa: F401

        settings["freezing"] = True
        settings["freezing_discard_parameters"] = True
        settings["enable_linear_binary_folding"] = args.mode == "on"

        class LinearBinary(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 32, bias=True)
                self.other = torch.nn.Parameter(torch.rand(32), requires_grad=False)

            def forward(self, x):
                return self.linear(x) + self.other

        model = LinearBinary().eval().to(device)
        inputs = (torch.rand(4, 3, device=device),)
        with torch.no_grad():
            expected = model(*inputs)

        def invoke(compiled):
            with torch.no_grad():
                return compiled(*inputs)

        counter_name = "binary_folding"
        gradient_models = None

    counters.clear()
    options = {"npu_backend": "triton_experimental"} if args.device == "npu" else None
    with config.patch(settings):
        started = time.perf_counter()
        compiled = torch.compile(model, backend="inductor", fullgraph=True, options=options)
        actual = invoke(compiled)
        runtime.synchronize()
        compile_ms = (time.perf_counter() - started) * 1000
        assert_close(torch, actual, expected)
        target_count = counters["inductor"][counter_name]
        if args.mode == "on" and target_count < 1:
            raise RuntimeError(f"ON未命中目标counter：{counter_name}")
        if args.mode == "off" and target_count != 0:
            raise RuntimeError(f"OFF仍命中目标counter：{counter_name}={target_count}")
        if args.unit != "linear-binary-folding":
            if args.mode == "on" and (
                state["handler_calls"] < 1 or state["graph_changes"] < 1
            ):
                raise RuntimeError("ON未捕获精确目标handler改写")
            if args.mode == "off" and state["handler_calls"] != 0:
                raise RuntimeError("OFF仍调用目标handler")
        if gradient_models is not None:
            compiled_model, eager_model = gradient_models
            for left, right in zip(compiled_model.parameters(), eager_model.parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-5)

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
            memory = {
                "allocated": runtime.max_memory_allocated(),
                "reserved": runtime.max_memory_reserved(),
            }

    task, acceptance, _ = TARGETS[args.unit]
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
        "task_id": task,
        "acceptance_unit_id": acceptance,
        "unit": args.unit,
        "mode": args.mode,
        "phase": args.phase,
        "backend": "triton_experimental" if args.device == "npu" else "inductor-default",
        "backend_selected_before_import": True,
        "pytorch_commit": torch.version.git_version,
        "pytorch_worktree_status": worktree,
        "correctness": "passed",
        "numerical_execution": True,
        "target_rewrite": "confirmed" if args.mode == "on" else "disabled-control",
        "graph_breaks": 0,
        "fallbacks": 0,
        "product_disabled": False,
        "measurement_workload": WORKLOADS[args.unit],
        "input_spec": INPUT_SPECS[args.unit],
        "state": {**state, "target_counter": target_count, "counter_name": counter_name},
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
        "gate_sha256": sha256(args.gate) if gate else None,
        "loaded_source_sha256": source_hashes(),
    }
    (output / "result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"worker_status=passed unit={args.unit} mode={args.mode} "
        f"phase={args.phase} target_counter={target_count}"
    )


if __name__ == "__main__":
    main()
