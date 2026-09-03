#!/usr/bin/env python3
"""T-077 B2B GEMM 的 NPU 最小 capability 探针。

该脚本不修改产品源码。candidate 模式仅把 NPU FakeTensor 以 CUDA capability
视图传给 upstream ``is_b2b_gemm_good_on``，从而完整复用原 shape/profitability
启发式，并让真实 NPU IR、模板、autotune、codegen 和 runtime 接受验证。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

# backend 注册具有进程级生命周期，必须先于 torch/torch_npu 导入。
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"

import torch
import torch_npu
from torch._dynamo.utils import counters
from torch._inductor.utils import run_and_get_code


def left_gelu(a, b, c):
    return torch.mm(torch.nn.functional.gelu(torch.mm(a, b)), c)


def right_relu(a, b, c):
    return torch.mm(a, torch.nn.functional.relu(torch.mm(b, c)))


def trivial_left(a, b, c):
    return torch.mm(torch.mm(a, b), c)


def trivial_right(a, b, c):
    return torch.mm(a, torch.mm(b, c))


def bad_pattern(a, b, c):
    first = torch.mm(a, b)
    second = torch.mm(first, c)
    return torch.mm(first, second)


class _CudaCapabilityView:
    """只覆盖 device capability 属性，其余 metadata 委托给原 FakeTensor。"""

    is_cuda = True

    def __init__(self, value):
        self._value = value

    def __getattr__(self, name):
        return getattr(self._value, name)


def install_candidate_device_adapter():
    """复用 upstream 启发式，仅为 NPU capability 探针解除首层 device guard。"""
    from torch._inductor.fx_passes import b2b_gemm

    original = b2b_gemm.is_b2b_gemm_good_on
    if getattr(original, "_t077_npu_capability_probe", False):
        return

    def npu_capability_probe(is_left_assoc, a_node, b_node, c_node):
        nodes = (a_node, b_node, c_node)
        if all("val" in node.meta for node in nodes):
            values = tuple(node.meta["val"] for node in nodes)
            if all(getattr(value, "device", None).type == "npu" for value in values):
                proxy_nodes = tuple(
                    SimpleNamespace(meta={"val": _CudaCapabilityView(value)})
                    for value in values
                )
                return original(is_left_assoc, *proxy_nodes)
        return original(is_left_assoc, a_node, b_node, c_node)

    npu_capability_probe._t077_npu_capability_probe = True
    npu_capability_probe._t077_original = original
    b2b_gemm.is_b2b_gemm_good_on = npu_capability_probe


def install_autotune_exception_passthrough():
    """诊断时移除 upstream 对 AssertionError 的误导性 ``Incorrect result`` 包装。"""
    from torch._inductor import select_algorithm

    cache_type = select_algorithm.AlgorithmSelectorCache

    @classmethod
    def benchmark_choices(cls, choices, autotune_args, is_collective=False):
        if is_collective:
            raise RuntimeError("T-077 B2B capability 探针不支持 collective autotune")
        return {
            choice: cls.benchmark_choice(choice, autotune_args) for choice in choices
        }

    cache_type.benchmark_choices = benchmark_choices


def install_npu_benchmarker_dispatch():
    """让 upstream TritonTemplate 的通用 benchmarker 正确选择 NPU interface。"""
    from torch._inductor.runtime import benchmarking

    def benchmark_npu(benchmarker, callable_, *, warmup, rep, **kwargs):
        return benchmarker.benchmark_gpu(
            callable_,
            warmup=warmup,
            rep=rep,
            device_type="npu",
            **kwargs,
        )

    benchmarking.register_benchmarker("npu", benchmark_npu, override=True)


def make_inputs(left: bool, matrix_m: int = 256, matrix_n: int = 32):
    if left:
        shapes = (
            (matrix_m, matrix_n),
            (matrix_n, matrix_m),
            (matrix_m, matrix_n),
        )
    else:
        shapes = (
            (matrix_n, matrix_m),
            (matrix_m, matrix_n),
            (matrix_n, matrix_m),
        )
    return tuple(
        torch.randn(shape, device="npu", dtype=torch.float16) for shape in shapes
    )


def make_mlp_inputs(matrix_m: int, matrix_n: int):
    """复用社区 GELU-MLP 性能测例的 O=P=N shape 关系。"""
    shapes = (
        (matrix_m, matrix_n),
        (matrix_n, matrix_n),
        (matrix_n, matrix_n),
    )
    return tuple(
        torch.randn(shape, device="npu", dtype=torch.float16) for shape in shapes
    )


def fp32_reference(fn, inputs):
    promoted = tuple(value.float() for value in inputs)
    return fn(*promoted).to(torch.float16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--case", default="all")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--matrix-m", type=int, default=256)
    parser.add_argument("--matrix-n", type=int, default=32)
    parser.add_argument("--debug-autotune-exception", action="store_true")
    parser.add_argument("--require-template-selected", action="store_true")
    parser.add_argument(
        "--allow-positive-unmatched",
        action="store_true",
        help="性能网格扫描时记录 upstream profitability 拒绝，不将其视为失败",
    )
    args = parser.parse_args()

    if Path.cwd().resolve() != Path("/home/z50063656/tmp"):
        raise RuntimeError("NPU 测试必须从 /home/z50063656/tmp 启动")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if not torch.npu.is_available():
        raise RuntimeError("当前进程不可见 NPU")
    torch.npu.set_device("npu:0")
    from torch_npu.utils._dynamo import register_inductor_npu

    register_inductor_npu()
    if args.mode == "candidate":
        install_candidate_device_adapter()
        install_npu_benchmarker_dispatch()
    if args.debug_autotune_exception:
        install_autotune_exception_passthrough()

    cases = {
        "left-gelu-positive": (
            left_gelu,
            make_inputs(True, args.matrix_m, args.matrix_n),
            True,
            "left",
        ),
        "gelu-mlp-performance-probe": (
            left_gelu,
            make_mlp_inputs(args.matrix_m, args.matrix_n),
            True,
            "left",
        ),
        "right-relu-positive": (
            right_relu,
            make_inputs(False, args.matrix_m, args.matrix_n),
            True,
            "right",
        ),
        "trivial-left-positive": (
            trivial_left,
            make_inputs(True, args.matrix_m, args.matrix_n),
            True,
            "left",
        ),
        "trivial-right-positive": (
            trivial_right,
            make_inputs(False, args.matrix_m, args.matrix_n),
            True,
            "right",
        ),
        "bad-pattern-negative": (
            bad_pattern,
            make_inputs(True, args.matrix_m, args.matrix_n),
            False,
            None,
        ),
        "bad-shape-negative": (
            trivial_left,
            tuple(
                torch.randn((100, 100), device="npu", dtype=torch.float16)
                for _ in range(3)
            ),
            False,
            None,
        ),
    }
    functional_cases = [
        "left-gelu-positive",
        "right-relu-positive",
        "trivial-left-positive",
        "trivial-right-positive",
        "bad-pattern-negative",
        "bad-shape-negative",
    ]
    selected = functional_cases if args.case == "all" else [args.case]
    unknown = [case for case in selected if case not in cases]
    if unknown:
        raise ValueError(f"未知 case: {unknown}; 可选值为 {list(cases)} 或 all")

    observations = []
    with torch._inductor.config.patch(b2b_gemm_pass=True):
        for variant_id in selected:
            fn, inputs, use_fp32_reference, expected_marker = cases[variant_id]
            torch._dynamo.reset()
            counters.clear()
            eager_expected = fn(*inputs)
            fp32_expected = (
                fp32_reference(fn, inputs) if use_fp32_reference else None
            )
            started = time.perf_counter_ns()
            compiled = torch.compile(
                fn,
                dynamic=False,
                options={"npu_backend": "triton_experimental"},
            )
            actual, codes = run_and_get_code(compiled, *inputs)
            torch.npu.synchronize()
            compile_and_first_run_ms = (
                time.perf_counter_ns() - started
            ) / 1_000_000
            target_count = counters["inductor"]["b2b_gemm"]
            code_text = "\n\n".join(codes)
            markers = {
                "left": "B2B_GEMM_LEFT_TRITON_ENTRANCE" in code_text,
                "right": "B2B_GEMM_RIGHT_TRITON_ENTRANCE" in code_text,
            }
            is_positive = expected_marker is not None
            template_selected = (
                markers[expected_marker] if expected_marker is not None else False
            )
            # 社区正例在融合模板获选时使用 FP32 accumulation 参考；若 autotune
            # 选择未融合 fallback，则应回到原始 eager FP16 语义进行正确性判断。
            expected = (
                fp32_expected
                if template_selected and fp32_expected is not None
                else eager_expected
            )
            reference_kind = (
                "fp32-accumulation"
                if template_selected and fp32_expected is not None
                else "eager-fp16-fallback"
            )
            torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.01)
            if args.mode == "baseline":
                if target_count != 0 or any(markers.values()):
                    raise AssertionError(
                        f"baseline {variant_id} 不应命中 B2B: "
                        f"counter={target_count}, markers={markers}"
                    )
            elif is_positive:
                if target_count <= 0 and not args.allow_positive_unmatched:
                    raise AssertionError(f"candidate {variant_id} 未命中 B2B pattern")
                if args.require_template_selected and not markers[expected_marker]:
                    raise AssertionError(
                        f"candidate {variant_id} 虽命中 pattern，但未生成/选择 "
                        f"{expected_marker} B2B 模板: {markers}"
                    )
            elif target_count != 0 or any(markers.values()):
                raise AssertionError(
                    f"candidate 负例 {variant_id} 不应命中 B2B: "
                    f"counter={target_count}, markers={markers}"
                )

            code_path = args.artifact_dir / f"{variant_id}-generated_code.py"
            code_path.write_text(code_text, encoding="utf-8")
            observations.append(
                {
                    "variant_id": variant_id,
                    "input_shapes": [list(value.shape) for value in inputs],
                    "dtype": "float16",
                    "reference": reference_kind,
                    "target_count": target_count,
                    "markers": markers,
                    "template_selected": template_selected,
                    "selection_status": (
                        "b2b-template-selected"
                        if expected_marker is not None and markers[expected_marker]
                        else (
                            "unoptimized-fallback-selected"
                            if expected_marker is not None and target_count > 0
                            else (
                                "device-or-profitability-guard-rejected"
                                if expected_marker is not None
                                else "negative-remained-unmatched"
                            )
                        )
                    ),
                    "correctness": "passed",
                    "compile_and_first_run_ms": compile_and_first_run_ms,
                    "generated_code": str(code_path),
                }
            )

    result = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_id": "T-077",
        "acceptance_unit_id": "AU-b2b-gemm",
        "mode": args.mode,
        "backend": "triton_experimental",
        "adapter_scope": (
            "none"
            if args.mode == "baseline"
            else "test-only NPU-as-CUDA capability view passed only to upstream profitability heuristic"
        ),
        "product_source_modified": False,
        "tests_run": len(observations),
        "successful": True,
        "observations": observations,
        "environment": {
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(0),
            "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "cwd": str(Path.cwd()),
        },
    }
    result_path = args.artifact_dir / "capability_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("T077_B2B_CAPABILITY_RESULT=" + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
