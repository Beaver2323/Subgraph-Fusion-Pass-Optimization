#!/usr/bin/env python3
"""T-078 门禁落盘前的目标级 fresh-process NPU 候选性能 worker。

正式候选测量已经完成。最终产品已关闭 addmm unfuse 和 baddbmm 非默认
标量路径；不要直接运行本文件绕过产品门禁。当前公开入口只校验已落盘结果。
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any


# 后端注册具有进程级生命周期；必须先于 torch / torch_npu 导入。
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"
os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")

WORK_DIR = Path("/home/z50063656/tmp")
if Path.cwd().resolve() != WORK_DIR:
    raise RuntimeError("NPU 性能测试必须从 /home/z50063656/tmp 启动")

import torch
import torch_npu
from torch._dynamo.utils import counters
from torch._inductor.utils import run_and_get_code


PYTORCH_COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
TORCH_NPU_ROOT = Path("/home/z50063656/Pass/src/torch_npu")
TARGETS = {
    "addcdiv": {
        "acceptance_unit_id": "AU-post-grad-fuse-addcdiv-to-fma",
        "handler_names": {
            "_fuse_addcdiv_to_fma",
            "npu_fuse_addcdiv_to_fma",
        },
        "expected_hits": 1,
        "nodeids": [
            "test/inductor/test_torchinductor.py::"
            "GPUTests.test_addcdiv_fma_bitwise_equal_cuda",
            "test/inductor/test_torchinductor.py::"
            "GPUTests.test_addcdiv_fma_uses_fma_and_div_rn_cuda",
        ],
        "community_benchmark": "absent",
    },
    "partial-reuse": {
        "acceptance_unit_id": "AU-post-grad-reuse-partial",
        "handler_names": {"reuse_partial"},
        "expected_hits": 3,
        "nodeids": [
            "test/inductor/test_pattern_matcher.py::"
            "TestPatternMatcher.test_successful_partial_reuse"
        ],
        "community_benchmark": "absent",
    },
    "unfuse-addmm": {
        "acceptance_unit_id": "AU-post-grad-unfuse-bias-add-to-pointwise",
        "handler_names": {"unfuse_bias_add_to_pointwise"},
        "expected_hits": 1,
        "nodeids": [
            "test/inductor/test_pattern_matcher.py::"
            "TestPatternMatcher.test_unfuse_bias_addmm"
        ],
        "community_benchmark": "absent",
    },
    "unfuse-baddbmm": {
        "acceptance_unit_id": (
            "AU-post-grad-unfuse-bias-baddbmm-to-pointwise"
        ),
        "handler_names": {"unfuse_bias_baddbmm_to_pointwise"},
        "expected_hits": 2,
        "nodeids": [
            "test/inductor/test_pattern_matcher.py::"
            "TestPatternMatcher.test_unfuse_broadcast_bias_baddbmm",
            "test/inductor/test_pattern_matcher.py::"
            "TestPatternMatcher.test_unfuse_broadcast_bias_baddbmm_alpha_beta",
        ],
        "community_benchmark": "absent",
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


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(TORCH_NPU_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_target_control(
    unit: str, mode: str, hits: dict[str, int]
) -> list[dict[str, Any]]:
    """只观测目标 handler；OFF 额外令这些 entry 的 extra_check 返回 False。"""
    from torch._inductor.fx_passes import post_grad

    target_names = TARGETS[unit]["handler_names"]
    seen: set[int] = set()
    selected: list[dict[str, Any]] = []
    for key, entries in post_grad.pass_patterns[2].patterns.items():
        for entry in entries:
            if id(entry) in seen:
                continue
            seen.add(id(entry))
            handler = getattr(entry, "handler", None)
            name = getattr(handler, "__name__", None)
            if name not in target_names:
                continue
            selected.append(
                {
                    "handler": name,
                    "registry_key": [str(key[0]), str(key[1])],
                    "entry_type": type(entry).__name__,
                }
            )
            original_handler = handler

            @functools.wraps(original_handler)
            def observed_handler(
                *args: Any,
                _handler=original_handler,
                _name=name,
                **kwargs: Any,
            ) -> Any:
                hits["total"] = hits.get("total", 0) + 1
                hits[_name] = hits.get(_name, 0) + 1
                return _handler(*args, **kwargs)

            entry.handler = observed_handler
            if mode == "off":
                entry.extra_check = lambda match: False

    if not selected:
        raise RuntimeError(f"没有找到 {unit} 的目标 pattern entry")
    if unit == "addcdiv" and not any(
        item["handler"] == "npu_fuse_addcdiv_to_fma" for item in selected
    ):
        raise RuntimeError("未加载 T-078 正式 NPU addcdiv handler，禁止性能测量")
    return selected


def make_workloads(
    unit: str, addcdiv_dtype: str = "float32"
) -> list[dict[str, Any]]:
    torch.manual_seed(20260906)
    torch.npu.manual_seed_all(20260906)
    if unit == "addcdiv":
        dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[addcdiv_dtype]
        s = torch.randn(64, 64, device="npu", dtype=dtype)
        t1 = torch.randn(64, 64, device="npu", dtype=dtype)
        t2 = (
            torch.randn(64, 64, device="npu", dtype=dtype)
            .abs()
            .clamp(min=0.1)
        )

        def make_fn(value: float):
            def fn(inp, tensor1, tensor2):
                return torch.addcdiv(inp, tensor1, tensor2, value=value)

            return fn

        # value=1 在 decomposition 中会消去乘 1，形成 div+add，因而不满足
        # div->mul->add 的目标 pattern。它保留为功能/bitwise 邻居，不伪造性能 ON。
        return [
            {
                "id": f"addcdiv-{addcdiv_dtype}-value-{int(value)}",
                "fn": make_fn(value),
                "inputs": (s, t1, t2),
                "shape_contract": (
                    f"{addcdiv_dtype}[64,64] 三输入，value={value:.1f}；"
                    "来自社区 bitwise/codegen 测例"
                ),
                "exact_contract": True,
                "required_on_tokens": (
                    ["triton.language.div_rn", "tl.fma"]
                    if value != 1
                    else ["triton.language.div_rn"]
                ),
            }
            for value in (2.0,)
        ]

    if unit == "partial-reuse":
        cases = [
            ("partial-amax-amax-2048", (2048, 2048), torch.amax, torch.amax),
            ("partial-amin-min-1024", (1024, 1024), torch.amin, torch.min),
            ("partial-amax-max-4096x512", (4096, 512), torch.amax, torch.max),
        ]
        workloads = []
        for case_id, shape, partial_fn, full_fn in cases:
            x = torch.randn(*shape, device="npu", dtype=torch.float32)

            def fn(value, _partial=partial_fn, _full=full_fn):
                return _partial(value, [0], True), _full(value)

            workloads.append(
                {
                    "id": case_id,
                    "fn": fn,
                    "inputs": (x,),
                    "shape_contract": (
                        f"float32{list(shape)}；同时返回 {partial_fn.__name__}"
                        f"(dim=[0],keepdim=True) 与 {full_fn.__name__}(x)"
                    ),
                    "exact_contract": True,
                    "required_on_tokens": [],
                }
            )
        return workloads

    if unit == "unfuse-addmm":
        bias = torch.randn(20, device="npu", dtype=torch.float32)
        left = torch.randn(10, 15, device="npu", dtype=torch.float32)
        right = torch.randn(15, 20, device="npu", dtype=torch.float32)

        def addmm_gelu(inp, a, b):
            return torch.nn.functional.gelu(torch.ops.aten.addmm(inp, a, b))

        return [
            {
                "id": "addmm-unfuse-fp32-gelu",
                "fn": addmm_gelu,
                "inputs": (bias, left, right),
                "shape_contract": (
                    "bias=float32[20]，mat1=float32[10,15]，"
                    "mat2=float32[15,20]，GELU consumer"
                ),
                "exact_contract": False,
                "required_on_tokens": [],
            }
        ]

    bias = torch.randn(4, 1, 8, device="npu", dtype=torch.float32)
    left = torch.randn(4, 6, 5, device="npu", dtype=torch.float32)
    right = torch.randn(4, 5, 8, device="npu", dtype=torch.float32)

    def baddbmm_gelu(inp, a, b):
        return torch.nn.functional.gelu(torch.ops.aten.baddbmm(inp, a, b))

    def baddbmm_alpha_beta(inp, a, b):
        return torch.nn.functional.relu(
            torch.ops.aten.baddbmm(inp, a, b, alpha=0.8, beta=0.2)
        )

    return [
        {
            "id": "baddbmm-unfuse-broadcast-gelu",
            "fn": baddbmm_gelu,
            "inputs": (bias, left, right),
            "shape_contract": (
                "bias=float32[4,1,8]，mat1=float32[4,6,5]，"
                "mat2=float32[4,5,8]，GELU consumer"
            ),
            "exact_contract": False,
            "required_on_tokens": [],
        },
        {
            "id": "baddbmm-unfuse-alpha-beta",
            "fn": baddbmm_alpha_beta,
            "inputs": (bias, left, right),
            "shape_contract": (
                "相同 broadcast 矩阵合同，alpha=0.8，beta=0.2，ReLU consumer"
            ),
            "exact_contract": False,
            "required_on_tokens": [],
        },
    ]


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in flatten_tensors(item)]
    raise TypeError(f"不支持的输出类型：{type(value)!r}")


def check_correctness(
    actual: Any, expected: Any, *, exact_contract: bool
) -> dict[str, Any]:
    actual_tensors = flatten_tensors(actual)
    expected_tensors = flatten_tensors(expected)
    if len(actual_tensors) != len(expected_tensors):
        raise AssertionError("compiled/eager 输出数量不同")
    bitwise = []
    max_abs_errors = []
    for got, want in zip(actual_tensors, expected_tensors):
        torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
        same = torch.equal(got, want)
        bitwise.append(same)
        max_abs_errors.append(float((got - want).abs().max().item()))
        if exact_contract and not same:
            raise AssertionError("社区精确合同要求 compiled/eager bitwise equal")
    return {
        "allclose": True,
        "bitwise_equal": all(bitwise),
        "per_output_bitwise_equal": bitwise,
        "max_abs_error": max(max_abs_errors, default=0.0),
    }


def debug_snapshot() -> set[Path]:
    roots = [
        Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/nonexistent")),
        Path(os.environ.get("TORCH_COMPILE_DEBUG_DIR", "/nonexistent")),
    ]
    names = {
        "fx_graph_readable.py",
        "fx_graph_transformed.py",
        "ir_pre_fusion.txt",
        "ir_post_fusion.txt",
        "output_code.py",
    }
    return {
        path.resolve()
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.name in names
    }


def preserve_debug_delta(
    before: set[Path], workload_id: str, artifact_dir: Path
) -> list[str]:
    destination = artifact_dir / "graphs"
    destination.mkdir(parents=True, exist_ok=True)
    added = sorted(debug_snapshot() - before)
    result = []
    for index, source in enumerate(added):
        target = destination / f"{workload_id}-{index:02d}-{source.name}"
        shutil.copyfile(source, target)
        result.append(str(target.relative_to(artifact_dir)))
    return result


def code_observation(code: str) -> dict[str, Any]:
    tokens = {}
    for name in ("addcdiv", "addmm", "baddbmm", "mm", "bmm"):
        candidates = (
            f"torch.ops.aten.{name}.default(",
            f"extern_kernels.{name}(",
        )
        tokens[name] = {
            "present": any(candidate in code for candidate in candidates),
            "spellings": {
                candidate: code.count(candidate) for candidate in candidates
            },
        }
    dispatch_lines = [
        line.strip()
        for line in code.splitlines()
        if not line.lstrip().startswith("#")
        and (
            re.search(r"(?:extern_kernels|torch\.ops\.aten)\.[\w.]+\(", line)
            or ".run(" in line
        )
    ]
    return {
        "operators": tokens,
        "contains_tl_fma": "tl.fma" in code,
        "contains_div_rn": "triton.language.div_rn" in code,
        "wrapper_dispatch_count": len(dispatch_lines),
        "wrapper_dispatch_lines": dispatch_lines,
        "scope": "generated wrapper 静态调用点，不等同于 profiler 的 NPU task 数",
    }


def check_structure(
    unit: str,
    mode: str,
    workload: dict[str, Any],
    observation: dict[str, Any],
) -> None:
    operators = observation["operators"]
    if unit == "addcdiv" and mode == "on":
        for token in workload["required_on_tokens"]:
            if token == "tl.fma" and not observation["contains_tl_fma"]:
                raise AssertionError("addcdiv ON 缺少 tl.fma")
            if token == "triton.language.div_rn" and not observation["contains_div_rn"]:
                raise AssertionError("addcdiv ON 缺少 div_rn")
    elif unit == "unfuse-addmm":
        if mode == "off" and not operators["addmm"]["present"]:
            raise AssertionError("addmm OFF 未保留 addmm")
        if mode == "on" and (
            operators["addmm"]["present"] or not operators["mm"]["present"]
        ):
            raise AssertionError("addmm ON 未形成 mm + pointwise")
    elif unit == "unfuse-baddbmm":
        if mode == "off" and not operators["baddbmm"]["present"]:
            raise AssertionError("baddbmm OFF 未保留 baddbmm")
        if mode == "on" and (
            operators["baddbmm"]["present"] or not operators["bmm"]["present"]
        ):
            raise AssertionError("baddbmm ON 未形成 bmm + pointwise")


def source_state() -> dict[str, Any]:
    paths = [
        TORCH_NPU_ROOT
        / "torch_npu/_inductor/triton_experimental/fx_passes.py",
        TORCH_NPU_ROOT
        / "torch_npu/_inductor/triton_experimental/lowering.py",
        TORCH_NPU_ROOT
        / "torch_npu/_inductor/triton_experimental/overrides.py",
        TORCH_NPU_ROOT
        / "torch_npu/_inductor/triton_experimental/npu_triton_helpers.py",
    ]
    return {
        "pytorch_commit": torch.version.git_version,
        "torch_npu_commit": git_output("rev-parse", "HEAD"),
        "torch_npu_worktree_dirty": bool(git_output("status", "--porcelain")),
        "relevant_file_sha256": {
            str(path.relative_to(TORCH_NPU_ROOT)): sha256(path) for path in paths
        },
    }


def validate_formal_source_overlay() -> dict[str, str]:
    expected_root = (
        TORCH_NPU_ROOT / "torch_npu/_inductor/triton_experimental"
    ).resolve()
    modules = {}
    for name in ("overrides", "fx_passes", "lowering", "npu_triton_helpers"):
        module = __import__(
            f"torch_npu._inductor.triton_experimental.{name}",
            fromlist=[name],
        )
        actual = Path(module.__file__).resolve()
        expected = expected_root / f"{name}.py"
        if actual != expected:
            raise RuntimeError(
                f"T-078 正式源码未加载：{name}: {actual} != {expected}"
            )
        modules[name] = str(actual)
    return modules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", choices=tuple(TARGETS), required=True)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--addcdiv-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="仅 addcdiv 单元使用；低精度功能门通过后才可选择 bfloat16",
    )
    args = parser.parse_args()
    if torch.version.git_version != PYTORCH_COMMIT:
        raise RuntimeError(
            f"PyTorch commit 不符：{torch.version.git_version} != {PYTORCH_COMMIT}"
        )
    if not torch.npu.is_available():
        raise RuntimeError("当前 fresh process 不可见 NPU")

    torch.npu.set_device("npu:0")
    from torch_npu.utils._dynamo import (
        _InductorNpuRegistry,
        register_inductor_npu,
    )

    register_inductor_npu()
    if _InductorNpuRegistry._loaded_backend != "triton_experimental":
        raise RuntimeError("实际加载的 NPU backend 不是 triton_experimental")
    loaded_modules = validate_formal_source_overlay()

    target_hits: dict[str, int] = {"total": 0}
    selected_entries = install_target_control(args.unit, args.mode, target_hits)
    if args.unit != "addcdiv" and args.addcdiv_dtype != "float32":
        raise RuntimeError("--addcdiv-dtype 仅适用于 addcdiv 单元")
    workloads = make_workloads(args.unit, args.addcdiv_dtype)
    artifact_dir = args.output.parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    workload_results = []
    counters.clear()

    for workload in workloads:
        expected = workload["fn"](*workload["inputs"])
        before_hits = target_hits["total"]
        before_debug = debug_snapshot()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        compile_started = time.perf_counter_ns()
        compiled = torch.compile(
            workload["fn"],
            fullgraph=True,
            options={"npu_backend": "triton_experimental"},
        )
        actual, codes = run_and_get_code(compiled, *workload["inputs"])
        torch.npu.synchronize()
        compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000
        correctness = check_correctness(
            actual,
            expected,
            exact_contract=workload["exact_contract"] and args.mode == "on",
        )

        for _ in range(args.warmup):
            actual = compiled(*workload["inputs"])
        torch.npu.synchronize()
        host_samples = []
        device_samples = []
        for _ in range(args.runs):
            start_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)
            host_started = time.perf_counter_ns()
            start_event.record()
            actual = compiled(*workload["inputs"])
            end_event.record()
            torch.npu.synchronize()
            host_samples.append(
                (time.perf_counter_ns() - host_started) / 1_000_000
            )
            device_samples.append(start_event.elapsed_time(end_event))

        final_correctness = check_correctness(
            actual,
            expected,
            exact_contract=workload["exact_contract"] and args.mode == "on",
        )
        code_text = "\n\n".join(codes)
        code_dir = artifact_dir / "generated_code"
        code_dir.mkdir(parents=True, exist_ok=True)
        code_path = code_dir / f"{workload['id']}.py"
        code_path.write_text(code_text, encoding="utf-8")
        observation = code_observation(code_text)
        check_structure(args.unit, args.mode, workload, observation)
        workload_results.append(
            {
                "workload_id": workload["id"],
                "shape_contract": workload["shape_contract"],
                "target_hits": target_hits["total"] - before_hits,
                "correctness_after_compile": correctness,
                "correctness_after_measurement": final_correctness,
                "timing": {
                    "compile_and_first_run_ms": compile_ms,
                    "host": summarize(host_samples),
                    "device_event": summarize(device_samples),
                },
                "memory": {
                    "max_allocated_bytes": torch.npu.max_memory_allocated(),
                    "max_reserved_bytes": torch.npu.max_memory_reserved(),
                },
                "generated_code": str(code_path.relative_to(artifact_dir)),
                "code_observation": observation,
                "debug_artifacts": preserve_debug_delta(
                    before_debug, workload["id"], artifact_dir
                ),
            }
        )

    expected_hits = 0 if args.mode == "off" else TARGETS[args.unit]["expected_hits"]
    if target_hits["total"] != expected_hits:
        raise AssertionError(
            f"{args.unit} {args.mode} 目标命中应为 {expected_hits}，"
            f"实际为 {target_hits['total']}"
        )
    if any(not item["debug_artifacts"] for item in workload_results):
        raise AssertionError("至少一个 workload 未捕获 FX/IR/output_code debug 工件")

    source = TARGETS[args.unit]
    result = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_id": "T-078",
        "acceptance_unit_id": source["acceptance_unit_id"],
        "unit": args.unit,
        "addcdiv_dtype": args.addcdiv_dtype if args.unit == "addcdiv" else None,
        "mode": args.mode,
        "round": args.round,
        "warmup": args.warmup,
        "runs": args.runs,
        "performance_case_source": {
            "kind": "tracker-derived-from-community-functional-case",
            "nodeids": source["nodeids"],
            "community_benchmark": source["community_benchmark"],
            "reused": [item["shape_contract"] for item in workloads],
            "added": [
                "目标 handler OFF/ON",
                "fresh-process 隔离",
                "host/NPU Event 计时、峰值显存与 debug 工件",
            ],
            "measurement_scope": (
                "单个 compiled 子图端到端调用；不是完整模型 benchmark"
            ),
        },
        "target_control": {
            "selected_entries": selected_entries,
            "only_difference": (
                "OFF 将目标 entry.extra_check 置 False；ON 保持目标 entry 原语义；"
                "两臂均安装同一观测 wrapper，未关闭全局 pattern matcher"
            ),
            "hits": target_hits,
            "expected_hits": expected_hits,
        },
        "workloads": workload_results,
        "environment": {
            "backend": _InductorNpuRegistry._loaded_backend,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "triton": getattr(__import__("triton"), "__version__", None),
            "device": torch.npu.get_device_name(0),
            "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "python": sys.version,
            "cwd": str(Path.cwd()),
        },
        "source_state": source_state(),
        "loaded_formal_modules": loaded_modules,
        "half_dtype_explicit_disable": (
            "NPU keep_addmm_fused_for_half_dtypes=true 路径不在 worker 中；"
            "不以 false 人为制造 ON"
            if args.unit == "unfuse-addmm"
            else None
        ),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("T078_PERFORMANCE=" + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
