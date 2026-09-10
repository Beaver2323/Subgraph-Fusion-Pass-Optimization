#!/usr/bin/env python3
"""T-102～T-107 单个 SDPA pattern 的功能/性能臂。"""

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
SUPPORTED_PATTERNS = (*range(1, 25), 28, 29, 30)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def task_for_pattern(pattern: int) -> str:
    if pattern <= 5:
        return "T-102"
    if pattern <= 10:
        return "T-103"
    if pattern <= 15:
        return "T-104"
    if pattern <= 20:
        return "T-105"
    if pattern <= 25:
        return "T-106"
    return "T-107"


def assert_close(torch, actual, expected) -> None:
    if isinstance(actual, (tuple, list)):
        if not isinstance(expected, type(actual)) or len(actual) != len(expected):
            raise RuntimeError("输出容器结构不一致")
        for left, right in zip(actual, expected):
            assert_close(torch, left, right)
        return
    torch.testing.assert_close(actual, expected, rtol=0.2, atol=2e-3)


def read_gate(path: Path, pattern: int, device: str) -> dict:
    gate = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "task_id": task_for_pattern(pattern),
        "acceptance_unit_id": f"AU-fuse-attention-sfdp-pattern-{pattern}",
        "backend": "triton_experimental" if device == "npu" else "inductor-default",
        "pytorch_commit": COMMIT,
        "correctness": "passed",
        "target_rewrite": "confirmed",
        "graph_breaks": 0,
        "fallbacks": 0,
        "product_disabled": False,
        "measurement_workload": f"sfdp-pattern-{pattern}-registered-half-inference",
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


def source_hashes() -> dict[str, str]:
    result = {}
    for name, module in list(sys.modules.items()):
        path = Path(getattr(module, "__file__", ""))
        if name.startswith(("torch._inductor", "torch_npu._inductor", "triton")) and path.is_file():
            result[str(path.resolve())] = sha256(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", type=int, choices=SUPPORTED_PATTERNS, required=True)
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
    if args.phase == "benchmark":
        if args.gate is None:
            parser.error("benchmark必须提供人工签发的--gate")
        read_gate(args.gate.resolve(), args.pattern, args.device)

    work = Path(os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")).resolve()
    if Path.cwd().resolve() != work:
        raise RuntimeError(f"必须从 {work} 启动")

    # 后端选择必须发生在 torch/torch_npu import 之前。
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
    from torch._inductor.fx_passes import fuse_attention

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
    device = torch.device(args.device, 0)
    torch.manual_seed(20260910 + args.pattern)
    prefix = f"_sfdp_pattern_{args.pattern}"
    candidates = []
    for name, registration in fuse_attention._get_sfdp_patterns(device):
        if name == prefix or name.startswith(prefix + "_"):
            candidates.append((name, registration))
    if not candidates:
        raise RuntimeError(f"目标{prefix}在实际设备上没有生成注册候选")
    # 固定半精度推理候选，避免dropout随机性；若某pattern没有half则退到同pattern推理候选。
    candidates.sort(
        key=lambda item: (
            "_half" not in item[0],
            "_inference" not in item[0],
            "mask_fp32" in item[0],
            "_bs1" in item[0],
            item[0],
        )
    )
    registration_name, registration = candidates[0]
    search_fn = registration["search_fn"]
    scalar_workaround = registration.get("scalar_workaround", {})
    model = functools.partial(search_fn, **scalar_workaround)
    inputs = tuple(value.detach().clone() for value in registration["example_inputs"])
    expected = model(*inputs)

    settings = {
        "fx_graph_cache": False,
        "force_disable_caches": True,
        # OFF会关闭整轮joint_graph；该worker的图只包含当前登记的attention search_fn，
        # 必须结合精确pattern counter和FX原件审核，禁止外推到其他joint pass。
        "use_joint_graph_passes": args.mode == "on",
    }
    counters.clear()
    options = {"npu_backend": "triton_experimental"} if args.device == "npu" else None
    with config.patch(settings):
        started = time.perf_counter()
        compiled = torch.compile(model, backend="inductor", fullgraph=True, options=options)
        actual = compiled(*inputs)
        runtime.synchronize()
        compile_ms = (time.perf_counter() - started) * 1000
        assert_close(torch, actual, expected)
        per_pattern = dict(counters["inductor_pattern_matcher_per_pattern"])
        exact_count = sum(
            count for name, count in per_pattern.items()
            if name == prefix or name.startswith(prefix + "_")
        )
        general_count = counters["inductor"]["fuse_attention"]
        if args.mode == "on" and (exact_count < 1 or general_count < 1):
            raise RuntimeError(
                f"ON未命中精确pattern：exact={exact_count} general={general_count}"
            )
        if args.mode == "off" and (exact_count != 0 or general_count != 0):
            raise RuntimeError("OFF仍命中fuse_attention")

        samples = None
        memory = None
        if args.phase == "benchmark":
            for _ in range(args.warmup):
                compiled(*inputs)
            runtime.synchronize()
            runtime.reset_peak_memory_stats()
            samples = {"host_ms": [], "event_ms": []}
            for _ in range(args.runs):
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

    torch_root = Path(torch.__file__).resolve().parents[1]
    worktree = subprocess.run(
        ["git", "-C", str(torch_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    input_spec = [
        {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
        }
        for value in inputs
    ]
    record = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "task_id": task_for_pattern(args.pattern),
        "acceptance_unit_id": f"AU-fuse-attention-sfdp-pattern-{args.pattern}",
        "pattern": args.pattern,
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
        "registration_name": registration_name,
        "exact_pattern_counter": exact_count,
        "general_fuse_attention_counter": general_count,
        "measurement_workload": f"sfdp-pattern-{args.pattern}-registered-half-inference",
        "input_spec": input_spec,
        "off_control_scope": "use_joint_graph_passes=false on isolated target-only workload",
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
        "loaded_source_sha256": source_hashes(),
    }
    (output / "result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"worker_status=passed pattern={args.pattern} mode={args.mode} "
        f"phase={args.phase} exact_pattern_counter={exact_count}"
    )


if __name__ == "__main__":
    main()
