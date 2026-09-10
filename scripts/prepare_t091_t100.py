#!/usr/bin/env python3
"""机械生成已经人工审核的 T-091～T-100 准备合同。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path


COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
TASKS = tuple(f"T-{number:03d}" for number in range(91, 101))
WORKER = "runners/t091_t100_performance_worker.py"
LAUNCHER = "scripts/run_t091_t100_performance.py"

DEFERRED = {
    "T-091": {
        "AU-split-cat-normalize-split-default": ("deferred-cpu-only-community-test", "两个映射测例都在CPU张量上运行，不能冻结原生GPU合同。"),
        "AU-split-cat-normalize-split-default-aten": ("deferred-cpu-only-community-test", "Aten normalization测例使用CPU输入，且一个总计数不能独立归属三个split handler。"),
        "AU-split-cat-normalize-split-with-size-default-aten": ("deferred-cpu-only-community-test", "共享CPU测例没有逐handler断言，不能重复计为独立GPU单元。"),
        "AU-split-cat-normalize-squeeze-default": ("deferred-no-direct-community-test", "冻结revision未找到直接社区测例。"),
    },
    "T-092": {
        "AU-split-cat-normalize-unbind-default": ("deferred-no-direct-community-test", "冻结revision未找到直接社区测例。"),
        "AU-split-cat-remove-split-with-size-one": ("deferred-no-direct-community-test", "冻结revision未找到直接社区测例。"),
        "AU-split-cat-replace-einsum-to-pointwise": ("deferred-no-direct-community-test", "冻结revision未找到直接社区测例。"),
        "AU-split-cat-simplify-split-cat": ("deferred-mismapped-or-cpu-only-tests", "列出的Aten测例实际归属post-grad merge_split_cat_aten；其余pre-grad测例为CPU，不能证明本handler的GPU入口。"),
        "AU-split-cat-split-cat-to-slices": ("deferred-cpu-multi-pattern-test", "test_split_cat_new_patterns使用CPU并同时覆盖多个handler，缺少独立GPU归属。"),
    },
    "T-093": {
        "AU-split-cat-split-stack-to-cats": ("deferred-mismapped-or-cpu-only-tests", "Aten测例实际归属另一个post-grad handler，pre-grad测例使用CPU。"),
        "AU-split-cat-unbind-cat-to-view": ("deferred-cpu-multi-pattern-test", "共享test_split_cat_new_patterns仅为CPU且没有逐handler计数。"),
        "AU-split-cat-unbind-stack-to-slices": ("deferred-cpu-multi-pattern-test", "共享test_split_cat_new_patterns仅为CPU且没有逐handler计数。"),
    },
    "T-094": {
        "AU-pre-grad-efficient-conv-bn-eval": ("deferred-structural-dispatcher-merged-into-t098", "这是pre_grad对efficient_conv_bn_eval_pass的调度入口；动态合同在T-098按实际handler计一次，避免重复分母。"),
        "AU-pre-grad-fuse-conv-bn": ("deferred-mismapped-community-test", "旧索引指向test_basic内部函数，实际测的是efficient_conv_bn_eval handlers，不是该legacy helper。"),
        "AU-pre-grad-linear-permute-fusion": ("deferred-pure-cpu-fx-helper-test", "test_fx_fusion直接对CPU symbolic_trace图调用helper，没有真实GPU torch.compile入口。"),
        "AU-pre-grad-permute-linear-fusion": ("deferred-pure-cpu-fx-helper-test", "test_fx_fusion直接对CPU symbolic_trace图调用helper，没有真实GPU torch.compile入口。"),
        "AU-pre-grad-permute-matmul-fusion": ("deferred-pure-cpu-fx-helper-test", "test_fx_fusion直接对CPU symbolic_trace图调用helper，没有真实GPU torch.compile入口。"),
    },
    "T-095": {
        "AU-pre-grad-remove-identity": ("deferred-no-direct-community-test", "冻结revision未找到直接社区测例或可归属的性能合同。"),
    },
    "T-096": {
        "AU-misc-patterns-e8m0-rceil": ("deferred-hardware-inapplicable-on-a100", "该分支只在NVIDIA SM100+注册并依赖Blackwell PTX；当前A100 reference会按产品硬件条件跳过。"),
        "AU-misc-patterns-randperm-index": ("deferred-no-direct-community-test", "冻结revision未找到直接社区测例。"),
        "AU-misc-patterns-randperm-index-add": ("deferred-no-direct-community-test", "冻结revision未找到直接社区测例。"),
        "AU-misc-patterns-randperm-index-full": ("deferred-no-direct-community-test", "冻结revision未找到直接社区测例。"),
    },
    "T-097": {
        "AU-replace-random-fuse-offset-creation": ("deferred-no-direct-community-test", "冻结revision未找到直接社区测例。"),
        "AU-replace-random-fuse-seed-creation": ("deferred-no-direct-community-test", "冻结revision未找到直接社区测例。"),
        "AU-replace-random-replace-randint": ("deferred-no-direct-community-test", "冻结revision未找到直接社区测例。"),
        "AU-replace-random-replace-random": ("deferred-fake-cpu-structural-test", "现有测例使用make_fx tracing_mode=fake和CPU样例，只验证stack metadata，不是设备执行或随机语义合同。"),
    },
    "T-098": {
        "AU-efficient-conv-bn-eval-efficient-conv-bn-eval-graph-transform": ("deferred-no-direct-call-module-attribution", "社区test_basic在Dynamo/ATen图上覆盖inlined/decomposed路径，未独立证明CallModule handler。"),
        "AU-efficient-conv-bn-eval-efficient-conv-bn-eval-graph-transform-decomposed": ("merged-into-inlined-contract", "同一个test_basic_cuda循环以decompose_nn_module false/true覆盖两条路径，合并到一个Conv-BN合同，禁止重复执行和重复计分母。"),
    },
    "T-099": {
        "AU-freezing-patterns-addmm-fuse-pattern-second": ("deferred-negative-or-cpu-unrelated-evidence", "CUDA freezing测例只保护unequal bias不误融合；另一测例是MKLDNN CPU，缺少GPU正例命中合同。"),
        "AU-freezing-patterns-int8-woq-fusion": ("deferred-mkldnn-cpu-only", "test_woq_int8是MKLDNN/CPU路径，不能转用为NPU或CUDA冻结合同。"),
        "AU-freezing-patterns-matmul-fuse": ("deferred-no-direct-community-test", "冻结revision未找到直接社区测例。"),
        "AU-freezing-patterns-matmul-fuse-pattern-two": ("deferred-no-direct-community-test", "冻结revision未找到直接社区测例。"),
        "AU-freezing-patterns-unnecessary-dtype-convert": ("deferred-no-direct-community-test", "冻结revision未找到直接社区测例。"),
    },
    "T-100": {},
}


def unit_t091() -> dict:
    nodeid = "test/inductor/test_split_cat_fx_passes.py::TestSplitCatFxPasses.test_stack_normalization_axis_kwarg"
    return {
        "acceptance_unit_id": "AU-split-cat-normalize-stack-default",
        "contract_name": "stack axis关键字规范化为dim",
        "stage": "pre_grad",
        "review_status": "prepared-awaiting-gpu-reference",
        "denominator_eligible": "yes-provisional",
        "coverage_phase": "awaiting-gpu-reference",
        "upstream_sources": [{"path": "torch/_inductor/fx_passes/split_cat.py", "line": 383, "symbol": "normalize_stack_default", "role": "normalization-pass-handler"}],
        "community_tests": [{"nodeid": nodeid, "role": "primary-positive", "evidence_scope": "真实GPU输入与torch.compile；axis=1规范化且数值等价"}],
        "variants": [{"variant_id": "axis-kwarg-positive", "kind": "positive", "expected_reference_match": True, "expected_behavior": "torch.stack(axis=1)改写为规范dim关键字且输出不变", "expected_counter": "人工FX复核确认目标节点变化", "reference_status": "not-run", "npu_status": "not-run"}],
        "tracking": {"reference_mode": "direct", "npu_mode": "triton-experimental-derived-from-community-functional-case", "allowed_local_deviation": ["NPU只替换设备并显式选择triton_experimental，保持shape和axis合同"], "forbidden_local_deviation": ["把CPU normalization计数转为GPU证据", "复用其他backend结果"]},
        "npu_control": {"config": "pre_grad_fusion_options.normalization_pass", "off": {}, "on": {"normalization_pass": {}}, "status": "prepared", "next_action": "GPU FX确认后在NPU真机验证精确handler、图改写和数值，再签性能门禁"},
    }


def unit_t096() -> dict:
    tests = [
        ("test_pattern_fires_and_is_correct", "primary-positive", "普通正数输入命中并与eager相等"),
        ("test_correct_for_values_one_ulp_above_power_of_two", "boundary-positive", "2^e上方1 ULP必须向上编码"),
        ("test_regression_gh178045_encoding_correctness", "regression-positive", "gh-178045边界编码回归"),
    ]
    prefix = "test/inductor/test_fp8.py::TestE8M0Log2PatternBitManip."
    return {
        "acceptance_unit_id": "AU-misc-patterns-e8m0-rceil-log2",
        "contract_name": "pre-SM100 E8M0 log2-ceil精确位运算替换",
        "stage": "post_grad",
        "review_status": "prepared-awaiting-gpu-reference",
        "denominator_eligible": "yes-provisional",
        "coverage_phase": "awaiting-gpu-reference",
        "upstream_sources": [{"path": "torch/_inductor/fx_passes/misc_patterns.py", "line": 170, "symbol": "e8m0_rceil_log2_pattern", "role": "cuda-registered-replacement"}],
        "community_tests": [{"nodeid": prefix + name, "role": role, "evidence_scope": scope} for name, role, scope in tests],
        "variants": [
            {"variant_id": "ordinary-positive", "kind": "positive", "expected_reference_match": True, "expected_behavior": "A100使用位域提取替代log2+ceil且输出等价", "expected_counter": "FX after不再保留software log2+ceil链", "reference_status": "not-run", "npu_status": "not-run"},
            {"variant_id": "one-ulp-boundary-positive", "kind": "positive", "expected_reference_match": True, "expected_behavior": "2^e上方1 ULP编码为bias+e+1", "expected_counter": "uint8逐元素精确相等", "reference_status": "not-run", "npu_status": "capability-pending"},
            {"variant_id": "gh178045-regression", "kind": "positive", "expected_reference_match": True, "expected_behavior": "避免software log2边界舍入少1", "expected_counter": "回归断言通过", "reference_status": "not-run", "npu_status": "capability-pending"},
        ],
        "tracking": {"reference_mode": "direct", "npu_mode": "triton-experimental-capability-pending", "allowed_local_deviation": ["先记录NPU原生未注册阻断；GPU冻结后才评审将generic CUDA device guard最小推广到NPU"], "forbidden_local_deviation": ["绕过guard制造ON收益", "把SM100 PTX分支用于A100或NPU"]},
        "npu_control": {"config": "none-registration-is-device-guarded", "off": "native-no-npu-registration", "on": "pending-reviewed-minimal-adaptation", "status": "capability-pending-not-explicit-product-disable", "next_action": "先冻结A100原生reference，再在NPU triton_experimental确认阻断并单独评审最小适配；适配前不计性能"},
    }


def unit_t098() -> dict:
    nodeid = "test/inductor/test_efficient_conv_bn_eval.py::EfficientConvBNEvalGpuTests.test_basic_cuda"
    return {
        "acceptance_unit_id": "AU-efficient-conv-bn-eval-efficient-conv-bn-eval-graph-transform-inlined",
        "contract_name": "Conv-BN eval在权重侧重参数化（inlined/decomposed合并合同）",
        "stage": "pre_grad",
        "review_status": "prepared-awaiting-gpu-reference",
        "denominator_eligible": "yes-provisional",
        "coverage_phase": "awaiting-gpu-reference",
        "upstream_sources": [
            {"path": "torch/_inductor/fx_passes/efficient_conv_bn_eval.py", "line": 158, "symbol": "efficient_conv_bn_eval_graph_transform_inlined", "role": "functional-batch-norm-handler"},
            {"path": "torch/_inductor/fx_passes/efficient_conv_bn_eval.py", "line": 273, "symbol": "efficient_conv_bn_eval_graph_transform_decomposed", "role": "aten-batch-norm-handler"},
        ],
        "community_tests": [{"nodeid": nodeid, "role": "primary-positive-and-negative-product", "evidence_scope": "真实CUDA；Conv/Linear/ConvTranspose、bias、SyncBN、decompose false/true、单/多用户；前向/反向/SGD和精确counter"}],
        "variants": [
            {"variant_id": "inlined-positive", "kind": "positive", "expected_reference_match": True, "expected_behavior": "functional batch_norm路径改写且单Conv计数1", "expected_counter": "efficient_conv_bn_eval delta=expected", "reference_status": "not-run", "npu_status": "not-run"},
            {"variant_id": "decomposed-positive", "kind": "positive", "expected_reference_match": True, "expected_behavior": "aten batch_norm路径改写且保持梯度/更新一致", "expected_counter": "efficient_conv_bn_eval delta=expected", "reference_status": "not-run", "npu_status": "not-run"},
            {"variant_id": "multi-user-protection", "kind": "mixed-positive-negative", "expected_reference_match": True, "expected_behavior": "仅融合单用户Conv-BN，保护重复Conv和多用户BN", "expected_counter": "MultiUserConvOp delta=3", "reference_status": "not-run", "npu_status": "not-run"},
        ],
        "tracking": {"reference_mode": "direct", "npu_mode": "triton-experimental-derived-from-community-functional-case", "allowed_local_deviation": ["性能仅取社区乘积中的Conv2d+BN2d代表形状；功能仍跑完整原生乘积"], "forbidden_local_deviation": ["将同一test_basic重复计为两个分母", "跳过梯度或target counter"]},
        "npu_control": {"config": "efficient_conv_bn_eval_fx_passes", "off": False, "on": True, "status": "prepared", "next_action": "GPU原生完整乘积通过后，NPU先跑代表功能合同并确认精确counter/梯度，再性能计时"},
    }


def unit_t100() -> dict:
    nodeid = "test/inductor/test_binary_folding.py::FreezingGpuTests.test_linear_binary_folding_cuda"
    return {
        "acceptance_unit_id": "AU-binary-folding-folded-op",
        "contract_name": "冻结期Linear后常量二元运算折入权重/偏置",
        "stage": "freezing",
        "review_status": "prepared-awaiting-gpu-reference",
        "denominator_eligible": "yes-provisional",
        "coverage_phase": "awaiting-gpu-reference",
        "upstream_sources": [{"path": "torch/_inductor/fx_passes/binary_folding.py", "line": 479, "symbol": "folded_op", "role": "freezing-binary-folding-handler"}],
        "community_tests": [{"nodeid": nodeid, "role": "primary-positive-negative-product", "evidence_scope": "真实CUDA复制类；2D/3D Linear、add/sub/mul/div、scalar/tensor广播正例与bad-shape负例，逐项counter"}],
        "variants": [
            {"variant_id": "linear-fold-positive", "kind": "positive", "expected_reference_match": True, "expected_behavior": "合法常量binary折入Linear参数且counter=1", "expected_counter": "binary_folding=1", "reference_status": "not-run", "npu_status": "not-run"},
            {"variant_id": "broadcast-shape-negative", "kind": "negative", "expected_reference_match": False, "expected_behavior": "输出相关不合法广播shape保持未融合", "expected_counter": "binary_folding=0", "reference_status": "not-run", "npu_status": "not-run"},
        ],
        "tracking": {"reference_mode": "direct", "npu_mode": "triton-experimental-derived-from-community-functional-case", "allowed_local_deviation": ["NPU代表性能图复用Linear(3,32)、[4,3]和[32]常量"], "forbidden_local_deviation": ["用被CUDNN skip的Conv测例充当GPU通过", "关闭freezing制造非等价基线"]},
        "npu_control": {"config": "freezing=true and enable_linear_binary_folding", "off": False, "on": True, "status": "prepared", "next_action": "GPU原生线性乘积通过后，在NPU确认冻结、counter、数值和无fallback，再签性能门禁"},
    }


SELECTED = {"T-091": [unit_t091()], "T-096": [unit_t096()], "T-098": [unit_t098()], "T-100": [unit_t100()]}


def iso_to_cst(value: str) -> str:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("时间戳必须包含时区")
    return parsed.strftime("%Y-%m-%d %H:%M:%S") + " CST（UTC+08:00）"


def manifest(task: str, timestamp: str, original: dict) -> dict:
    units = SELECTED.get(task, [])
    deferred = []
    for candidate in original[task]["units"]:
        unit_id = candidate["provisional_unit_id"]
        if unit_id in {unit["acceptance_unit_id"] for unit in units}:
            continue
        status, reason = DEFERRED[task][unit_id]
        deferred.append({
            "provisional_unit_id": unit_id,
            "status": status,
            "reason": reason,
            "denominator_eligible": False,
            "performance_readiness": "blocked-by-direct-gpu-functional-contract",
            "next_action": "补齐可独立归属的原生GPU命中、正确性和合法OFF/ON合同后重开；当前不执行不计数",
        })
    status = "prepared-awaiting-gpu-reference" if units else "reviewed-no-gpu-ready-units"
    return {
        "schema_version": "1.0",
        "generated_at": timestamp,
        "status": status,
        "source_baselines": {"pytorch": {"commit": COMMIT, "branch": "release/2.14", "relevant_tracked_files_state": "clean"}},
        "mapping_review": {
            "input_scope": f"{task}全部{len(original[task]['units'])}个provisional候选逐项复核源码、社区测例、设备执行、目标归属与合法OFF/ON",
            "selected_acceptance_units": [unit["acceptance_unit_id"] for unit in units],
            "corrections": ["CPU/Fake/间接或错配测例不冻结GPU分母。", "同一社区测试覆盖多个handler时按一个行为合同计数，旧ID保留延期映射。", "社区索引遗漏但冻结源码存在的真实GPU目标测例允许经人工复核补入。"],
        },
        "reference_contract": {
            "device_class": "GPU",
            "backend": "inductor-default",
            "execution_order": "native-community-test-first-then-minimal-adapter-if-required",
            "minimum_gpus": 1 if units else 0,
            "distributed_world_size": None,
            "scope": "原生CUDA编译、数值、目标命中/改写与必要梯度合同" if units else "本批无可执行GPU合同；只保留静态审核和延期理由",
            "suite_status": "pending-gpu-reference" if units else "reviewed-no-gpu-ready-units",
        },
        "counting_policy": {"current_manifest_units": len(units), "current_frozen_denominator_units": 0, "current_formally_closed_units": 0, "denominator_rule": "原生GPU suite有效、零skip并完成人工FX复核后冻结"},
        "acceptance_units": units,
        "deferred_candidates": deferred,
    }


def case_data(task: str) -> list[dict]:
    if task == "T-091":
        return [{"case_id": "REF-stack-axis-normalization-native", "acceptance_unit_id": SELECTED[task][0]["acceptance_unit_id"], "source_test": SELECTED[task][0]["community_tests"][0]["nodeid"], "tracking_mode": "direct", "variant_ids": ["axis-kwarg-positive"], "expected_match": True, "expected_assertions": ["真实GPU torch.compile执行且零skip", "FX确认axis关键字规范化为dim=1", "输出与eager一致"], "required_artifacts": ["fx-before", "fx-after"], "benchmark": "not-configured-functional-reference"}]
    if task == "T-096":
        mapping = [
            ("REF-e8m0-log2-pattern-native", 0, ["ordinary-positive"]),
            ("REF-e8m0-log2-one-ulp-native", 1, ["one-ulp-boundary-positive"]),
            ("REF-e8m0-log2-gh178045-native", 2, ["gh178045-regression"]),
        ]
        unit = SELECTED[task][0]
        return [{"case_id": case_id, "acceptance_unit_id": unit["acceptance_unit_id"], "source_test": unit["community_tests"][index]["nodeid"], "tracking_mode": "direct", "variant_ids": variants, "expected_match": True, "expected_assertions": ["A100 pre-SM100原生CUDA执行且零skip", "边界uint8编码逐元素精确相等", "FX after确认log2+ceil链被位运算替换"], "required_artifacts": ["fx-before", "fx-after"], "benchmark": "blocked-npu-capability-pending"} for case_id, index, variants in mapping]
    if task == "T-098":
        unit = SELECTED[task][0]
        return [{"case_id": "REF-efficient-conv-bn-eval-product-native", "acceptance_unit_id": unit["acceptance_unit_id"], "source_test": unit["community_tests"][0]["nodeid"], "tracking_mode": "direct", "variant_ids": ["inlined-positive", "decomposed-positive", "multi-user-protection"], "expected_match": True, "expected_assertions": ["真实CUDA完整参数乘积执行且零skip", "单/多用户目标counter分别为1/3", "前向、梯度及SGD后再次前向与eager一致"], "required_artifacts": ["fx-before", "fx-after"], "benchmark": "tracker-derived-after-functional-gate", "timeout_seconds": 7200}]
    if task == "T-100":
        unit = SELECTED[task][0]
        return [{"case_id": "REF-linear-binary-folding-native", "acceptance_unit_id": unit["acceptance_unit_id"], "source_test": unit["community_tests"][0]["nodeid"], "tracking_mode": "direct", "variant_ids": ["linear-fold-positive", "broadcast-shape-negative"], "expected_match": True, "expected_assertions": ["真实CUDA复制类执行且零skip", "合法2D/3D scalar/tensor组合counter=1", "非法广播shape counter=0且数值保持"], "required_artifacts": ["fx-before", "fx-after"], "benchmark": "tracker-derived-after-functional-gate", "timeout_seconds": 7200}]
    return []


def reference_plan(task: str, timestamp: str, units: list[dict]) -> dict:
    cases = case_data(task)
    implemented = task in {"T-091", "T-098", "T-100"}
    status = "gpu-ready-awaiting-native-execution" if units else "reviewed-no-gpu-ready-units"
    return {
        "schema_version": "1.0", "task_id": task, "generated_at": timestamp, "status": status,
        "performance_plan": {"path": f"upstream/{task.lower().replace('-', '')}_performance_plan.yaml", "status": "implemented-awaiting-runtime-validation" if implemented else "not-implemented-blocked-or-not-applicable"},
        "case_guide": {"path": f"docs/{task.replace('-', '')}_FUNCTION_PERFORMANCE_GUIDE.md", "language": "zh-CN"},
        "manifest": {"path": f"upstream/{task.lower().replace('-', '')}_manifest.yaml", "schema_version": "1.0", "pytorch_commit": COMMIT},
        "execution_policy": {"order": "native-community-test-first-then-minimal-adapter-if-required", "case_isolation": "fresh-process", "failure_behavior": "continue-and-record", "artifact_capture": "原生TORCH_COMPILE_DEBUG与FX before/after", "benchmark_gate": "functional-reference-valid-first", "default_timeout_seconds": 3600, "minimum_gpus": 1 if units else 0, "distributed_world_size": None},
        "cases": cases,
        "non_executed_variants": [],
    }


def perf_unit(task: str, unit: dict) -> dict:
    nodeids = [item["nodeid"] for item in unit["community_tests"]]
    if task == "T-091":
        return {"acceptance_unit_id": unit["acceptance_unit_id"], "worker_unit": "stack-normalization", "performance_status": "prepared-awaiting-functional-gate", "verdict": "planned", "case_source": {"kind": "tracker-derived-from-community-functional-case", "nodeids": nodeids, "reason": "未发现社区benchmark；严格复用torch.stack、axis=1、两个[4,4]输入"}, "functional_gate": "OFF不调用目标；ON精确handler调用且FX变化；输出与eager一致，无graph break/fallback。", "workloads": [{"workload_id": "stack-normalization-community-shape", "role": "primary", "shape_contract": "x/y=[4,4] fp32，torch.stack(axis=1)", "measurement_scope": "单个compiled stack图，不含数据加载或完整模型"}], "off_on_control": "只切换pre_grad_fusion_options中normalization_pass；每臂新进程。", "artifacts": ["GPU/NPU功能原件", "FX before/after", "handler调用/图变化", "host/Event原始样本、显存、编译耗时", "loaded source sha256"], "negative_cases": ["OFF仍命中、ON不改图、数值错误、fallback或产品disable均拒绝计时"]}
    if task == "T-096":
        return {"acceptance_unit_id": unit["acceptance_unit_id"], "worker_unit": "none", "performance_status": "blocked-capability-pending-npu-registration", "verdict": "planned-after-reviewed-adaptation", "case_source": {"kind": "tracker-derived-from-community-functional-case", "nodeids": nodeids, "reason": "社区无性能benchmark；边界向量可派生但NPU原生没有目标注册，适配评审前禁止制造ON"}, "functional_gate": "先证明A100原生路径；NPU triton_experimental原生阻断必须留证，再单独审核generic guard最小适配。", "workloads": [{"workload_id": "e8m0-log2-boundary-vector", "role": "planned-after-adaptation", "shape_contract": "float32正数与2^e上方1 ULP向量，uint8精确输出", "measurement_scope": "仅目标编码子图；不是模型端到端"}], "off_on_control": "当前无NPU ON；不得绕过device guard。批准适配后再定义等价OFF/ON。", "artifacts": ["GPU reference FX", "NPU原生阻断", "device guard源码", "未来适配diff与功能证据"], "negative_cases": ["SM100 PTX路径不适用于A100/NPU", "适配前不计性能"]}
    if task == "T-098":
        return {"acceptance_unit_id": unit["acceptance_unit_id"], "worker_unit": "efficient-conv-bn", "performance_status": "prepared-awaiting-functional-gate", "verdict": "planned", "case_source": {"kind": "tracker-derived-from-community-functional-case", "nodeids": nodeids, "reason": "无精确社区性能benchmark；从test_basic乘积选Conv2d+BN2d、bias=true、[4,3,96,96]代表图"}, "functional_gate": "OFF counter=0；ON精确counter=1且图改写；输出和参数梯度与eager一致，无break/fallback。", "workloads": [{"workload_id": "efficient-conv-bn-community-conv2d", "role": "primary", "shape_contract": "Conv2d(3,32,k=3,stride=2,bias=true)+BN2d eval，x=[4,3,96,96] fp32", "measurement_scope": "compiled forward+backward；不含optimizer、数据加载或完整模型"}], "off_on_control": "只切换efficient_conv_bn_eval_fx_passes false/true；freezing=false；每臂新进程。", "artifacts": ["GPU完整乘积与NPU代表功能原件", "counter/FX", "数值/梯度", "host/Event样本、显存、编译耗时", "source sha256"], "negative_cases": ["多用户保护失败、ON无改写、梯度错误、fallback或产品disable均拒绝计时"]}
    return {"acceptance_unit_id": unit["acceptance_unit_id"], "worker_unit": "linear-binary-folding", "performance_status": "prepared-awaiting-functional-gate", "verdict": "planned", "case_source": {"kind": "tracker-derived-from-community-functional-case", "nodeids": nodeids, "reason": "社区无独立benchmark；复用Linear(3,32)、[4,3]、[32]常量add代表图"}, "functional_gate": "freezing保持true；OFF counter=0；ON counter=1且数值正确，无break/fallback。", "workloads": [{"workload_id": "linear-binary-folding-community-shape", "role": "primary", "shape_contract": "Linear(3,32,bias=true)+[32]常量add，x=[4,3] fp32", "measurement_scope": "冻结推理单次compiled forward；不含数据加载或完整模型"}], "off_on_control": "只切换enable_linear_binary_folding false/true，freezing固定true；每臂新进程。", "artifacts": ["GPU/NPU功能原件", "binary_folding counter和FX", "数值", "host/Event样本、显存、编译耗时", "source sha256"], "negative_cases": ["非法广播误融合、OFF仍命中、ON无改写、fallback或产品disable均拒绝计时"]}


def performance_plan(task: str, timestamp: str, units: list[dict]) -> dict:
    implemented = task in {"T-091", "T-098", "T-100"}
    return {
        "schema_version": "1.0", "task_id": task, "generated_at": timestamp,
        "status": "prepared-awaiting-gpu-and-npu-functional-gates" if implemented else ("blocked-capability-pending-reviewed-adaptation" if task == "T-096" else "not-applicable-no-gpu-ready-unit"),
        "implementation": {"status": "implemented-awaiting-runtime-validation", "entrypoint": WORKER, "launcher": LAUNCHER} if implemented else {"status": "not-implemented", "entrypoint": None, "reason": "没有可执行GPU单元" if not units else "NPU原生注册缺失；等待真机阻断和最小适配评审"},
        "backend_contract": {"npu_backend": "triton_experimental", "selection_timing": "新进程内导入torch/torch_npu之前设置TORCHINDUCTOR_NPU_BACKEND", "process_isolation": "fresh-process-per-arm", "off_on_order": ["OFF1", "ON1", "ON2", "OFF2", "OFF3", "ON3"], "explicit_disable_policy": "产品明确disable则记录并免测；generic device guard遗漏归为capability-pending，禁止未经审核绕过"},
        "measurement_contract": {"correctness_gate": "GPU原生reference有效后，NPU先验证OFF/ON数值、梯度（若适用）、精确目标改写、零graph break/CPU fallback，再人工签gate", "warmup": 10, "runs": 100, "timing": ["host synchronized wall time p50/p99", "device Event p50/p99", "compile latency separate", "raw samples"], "memory": ["peak allocated", "peak reserved"], "verdict_thresholds": {"improved_percent": 5, "regressed_percent": -5, "otherwise": "PERF_NEUTRAL"}, "rounds": 3, "aggregation": "三OFF/三ON分别取各臂p50/p99后再取中位数"},
        "hardware_contract": {"gpu_reference_minimum_devices": 1 if units else 0, "performance_minimum_devices": 1 if implemented else 0, "performance_world_size": 1 if implemented else 0, "actual_device_execution_required": bool(units), "fake_tensor_allowed_as_performance_oracle": False},
        "benchmark_search": {"revision": COMMIT, "paths": ["benchmarks/", "test/inductor/"], "terms": [unit["upstream_sources"][0]["symbol"] for unit in units] or [task], "result": "未发现可直接控制目标pass的社区性能benchmark；有功能合同的单元按社区模型/shape派生，空批次不派生。", "source_priority": "社区benchmark优先；否则保留社区功能图、shape、dtype和执行阶段，不称为社区原生benchmark或模型端到端"},
        "acceptance_units": [perf_unit(task, unit) for unit in units],
    }


def guide(task: str, timestamp: str, data: dict) -> str:
    units = data["acceptance_units"]
    deferred = data["deferred_candidates"]
    header = [
        f"# {task} 功能与性能测例讲解", "",
        f"> 更新时间：{iso_to_cst(timestamp)}", f"> 状态：{len(units)} 个GPU-ready单元，{len(deferred)} 个候选明确延期。", "",
        "NPU 功能、修复验证和性能统一使用 `triton_experimental`，并在导入 `torch`/`torch_npu` 前选后端；OFF/ON 每臂使用新进程。", "",
    ]
    if not units:
        header += [
            "## 功能测例", "", "本批经审核没有可冻结的原生GPU直接合同。可做零设备静态校验，但一键入口会拒绝实际空跑：", "",
            "```bash", "cd /data/z50063656/tmp", "", "bash \\", "  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \\", f"  --task {task} \\", "  --validate-only", "```", "",
            "这不是SKIP或PASS；所有旧ID均保留为非计数延期记录。", "", "## 性能测例", "", "没有通过GPU功能门禁的合同，因此不派生性能测例、不制造ON路径，性能状态为不适用。", "",
        ]
    else:
        header += [
            "## GPU 一键运行", "", "```bash", "cd /data/z50063656/tmp", "", "bash \\", "  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \\", f"  --task {task} \\", "  --gpu 2 \\", "  --wait-gpu", "```", "",
            "先运行冻结 PyTorch revision 的原生社区测例；只有真实设备/backend/采集阻断才进入最小适配审核。", "",
        ]
    details = {
        "T-091": ["## AU-split-cat-normalize-stack-default", "", "```python", "# torch/_inductor/fx_passes/split_cat.py:383", "dim = get_arg_value(node, 1, \"dim\") or 0", "...", "new_node = graph.call_function(node.target, args=(tensors,), kwargs={\"dim\": dim})", "counters[backend][\"normalization_pass\"] += 1", "```", "", "功能测例用两个 `[4,4]` GPU 张量执行 `torch.stack(axis=1)`；意图是把 numpy 兼容关键字和负维统一成规范 FX 形式。GPU/NPU都需证明数值一致和实际改写。性能测例复用相同图，唯一变量是 `normalization_pass` OFF/ON；它是微图，不是模型端到端。", ""],
        "T-096": ["## AU-misc-patterns-e8m0-rceil-log2", "", "```python", "# torch/_inductor/fx_passes/misc_patterns.py:170", "log2_val = torch.log2(inp)", "ceil_val = torch.ceil(log2_val)", "biased = torch.clamp(ceil_val, min=-127, max=127) + 127", "# pre-SM100 replacement: 直接读取float32指数和mantissa位域", "```", "", "功能测例来自社区三条A100 CUDA测试：普通值、`2^e`上方1 ULP边界和gh-178045回归。替换意图是避免software log2在边界舍入到较小整数。源码当前只在CUDA注册且extra-check要求CUDA；这不是NPU产品明确disable，而是能力待评审。NPU原生阻断和最小适配审核完成前，性能测例仅有设计、不执行。", ""],
        "T-098": ["## AU-efficient-conv-bn-eval-efficient-conv-bn-eval-graph-transform-inlined", "", "```python", "# torch/_inductor/fx_passes/efficient_conv_bn_eval.py:24", "weight_coeff = torch.rsqrt(bn.running_var + bn.eps).reshape(target_shape)", "coefff_on_the_fly = bn.weight.view_as(weight_coeff) * weight_coeff", "weight_on_the_fly = conv.weight * coefff_on_the_fly", "bias_on_the_fly = bn.bias + coefff_on_the_fly.flatten() * (conv_bias - bn.running_mean)", "return functional_call(conv, {\"weight\": weight_on_the_fly, \"bias\": bias_on_the_fly}, x)", "```", "", "功能测例是社区真实CUDA完整乘积，覆盖Linear/Conv/ConvTranspose、bias、SyncBN、单/多用户、inlined/decomposed，并检查前向、梯度、SGD后结果和精确counter。两个handler合为一个行为合同。性能测例从乘积中固定Conv2d+BN2d代表图，测forward+backward；不是完整模型端到端。", ""],
        "T-100": ["## AU-binary-folding-folded-op", "", "```python", "# torch/_inductor/fx_passes/binary_folding.py:479", "def folded_op(match, *args, **kwargs):", "    counters[\"inductor\"][\"binary_folding\"] += 1", "    # 把Linear后的add/sub/mul/div常量预折入weight/bias", "```", "", "旧索引漏掉了CUDA复制类。原生功能测例覆盖2D/3D Linear、四种binary、scalar/tensor广播正例和bad-shape负例，并逐项断言counter。Conv版本受CUDNN accuracy skip，不用于本批分母。性能测例固定Linear(3,32)+[32]常量add，freezing始终开启，仅切换linear binary folding。", ""],
    }
    header += details.get(task, [])
    header += ["## 延期候选", "", "| 候选 | 状态 | 原因 |", "| --- | --- | --- |"]
    if deferred:
        for item in deferred:
            header.append(f"| `{item['provisional_unit_id']}` | `{item['status']}` | {item['reason']} |")
    else:
        header.append("| 无 | — | 本批唯一候选已纳入GPU-ready合同。 |")
    if units and task in {"T-091", "T-098", "T-100"}:
        header += ["", "## NPU 功能与性能入口", "", "GPU分母人工复核后，从NPU服务器的固定tmp目录先跑功能预检：", "", "```bash", "cd /home/z50063656/tmp", "", "python \\", "  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t091_t100_performance.py \\", f"  --task {task} \\", "  --phase functional \\", "  --device npu", "```", "", "功能原件人工签gate后，再使用 `--phase benchmark --gate-root <目录>` 执行固定六臂。", ""]
    return "\n".join(header).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    datetime.fromisoformat(args.timestamp)
    root = args.repo_root.resolve()
    backlog = json.loads((root / "upstream/task_backlog.json").read_text(encoding="utf-8"))
    original = {batch["task_id"]: batch for batch in backlog["batches"] if batch["task_id"] in TASKS}
    if set(original) != set(TASKS):
        raise ValueError("backlog未完整保留T-091～T-100")
    if not args.write:
        print("prepared_generation_check=OK tasks=10 selected=4 deferred=33")
        return
    for task in TASKS:
        suffix = task.lower().replace("-", "")
        data = manifest(task, args.timestamp, original)
        files = {
            root / "upstream" / f"{suffix}_manifest.yaml": data,
            root / "upstream" / f"{suffix}_reference_plan.yaml": reference_plan(task, args.timestamp, data["acceptance_units"]),
            root / "upstream" / f"{suffix}_performance_plan.yaml": performance_plan(task, args.timestamp, data["acceptance_units"]),
        }
        for path, value in files.items():
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (root / "docs" / f"{task.replace('-', '')}_FUNCTION_PERFORMANCE_GUIDE.md").write_text(guide(task, args.timestamp, data), encoding="utf-8")
        wrapper = root / "scripts" / f"run_{suffix}_reference_all.sh"
        wrapper_text = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "repo_root=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd -P)\"\n"
            "exec bash \"${repo_root}/scripts/run_reference_all.sh\" \\\n+"
            f"  --manifest-path upstream/{suffix}_manifest.yaml \\\n+"
            f"  --plan-path upstream/{suffix}_reference_plan.yaml \\\n+"
            "  \"$@\"\n"
        )
        wrapper.write_text(wrapper_text.replace("\n+", "\n"), encoding="utf-8")
        wrapper.chmod(0o755)
        incoming = root / "results" / "incoming" / task
        incoming.mkdir(parents=True, exist_ok=True)
        incoming.joinpath("README.md").write_text(f"# {task} GPU handoff 接收目录\n\n> 更新时间：{args.timestamp}\n\n仅提交由统一GPU runner生成并校验通过的文本handoff。无GPU-ready单元时本目录保留为空批次审计入口，不上传伪结果。\n", encoding="utf-8")
    print("prepared_generation_write=OK tasks=10 selected=4 deferred=33")


if __name__ == "__main__":
    main()
