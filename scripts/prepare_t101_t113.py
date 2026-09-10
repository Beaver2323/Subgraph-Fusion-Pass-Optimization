#!/usr/bin/env python3
"""机械生成已经人工审核的 T-101～T-113 准备合同。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path


COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
TASKS = tuple(f"T-{number:03d}" for number in range(101, 114))
ATTENTION_TASKS = tuple(f"T-{number:03d}" for number in range(102, 108))
ATTENTION_WORKER = "runners/t102_t107_attention_performance_worker.py"
ATTENTION_LAUNCHER = "scripts/run_t102_t107_attention_performance.py"
FSDP_WORKER = "runners/t112_dedup_reduce_scatter_worker.py"
FSDP_LAUNCHER = "scripts/run_t112_dedup_reduce_scatter.py"

PATTERN_LINES = {
    1: 43, 2: 65, 3: 87, 4: 109, 5: 129, 6: 150, 7: 171, 8: 208,
    9: 237, 10: 267, 11: 297, 12: 318, 13: 341, 14: 358, 15: 384,
    16: 421, 17: 464, 18: 504, 19: 552, 20: 588, 21: 628, 22: 654,
    23: 684, 24: 716, 25: 752, 26: 782, 27: 820, 28: 857, 29: 876,
    30: 906,
}
PATTERN_INTENTS = {
    1: ("QK转置→除以缩放→softmax→V", "无mask、无dropout，scale=1/inv_scale"),
    2: ("QK转置→乘缩放→softmax→V", "无mask、无dropout，直接保留scale_factor"),
    3: ("除法缩放attention并带dropout", "把训练态dropout概率交给SDPA"),
    4: ("乘法缩放attention并带dropout", "把显式matmul/softmax/dropout链收为SDPA"),
    5: ("带attention mask的除法缩放", "mask转为Q dtype后进入SDPA"),
    6: ("带mask和dropout的除法缩放", "同时保留mask、scale与dropout语义"),
    7: ("QKV先permute且softmax上采样FP32", "避免SDPA内部额外布局复制并保留dropout"),
    8: ("pattern 7的无dropout形式", "保留permute和FP32 softmax意图"),
    9: ("先缩放Q、再matmul并带dropout", "识别缩放位置不同但等价的attention链"),
    10: ("pattern 9的无dropout形式", "融合Q预缩放与FP32 softmax"),
    11: ("HuggingFace式QKV permute attention", "将显式转置后的链改为SDPA"),
    12: ("HuggingFace式permute并带dropout", "训练态保留dropout概率"),
    13: ("三维bmm attention", "用unsqueeze/squeeze映射到四维SDPA，且检查permute维"),
    14: ("BERT Large布局、加法mask", "转置QKV并传递同dtype/bool mask"),
    15: ("DistilBERT masked_fill布尔mask", "把0/1 mask规范为SDPA布尔mask"),
    16: ("BERT Large mask+dropout", "CUDA命中后故意保留数学路径保证数值；非CUDA可转SDPA"),
    17: ("DistilBERT masked_fill+dropout", "规范布尔mask并保留dropout"),
    18: ("GPT2 causal mask并返回K/V", "融合attention同时保持多输出K/V"),
    19: ("GPT2 causal mask叠加attention mask", "先合成mask再交给SDPA"),
    20: ("新版DistilBERT先缩放Q的mask+dropout", "覆盖不同缩放位置"),
    21: ("T5加法mask与FP32 softmax", "融合为无缩放SDPA"),
    22: ("T5加法mask且返回K/V", "保持多输出合同"),
    23: ("T5零mask消去且返回K/V", "用attn_mask=None的SDPA"),
    24: ("MBart/PLBart reshape+bmm与异dtype mask", "恢复四维后以scale=1融合"),
    25: ("T5 mask+dropout", "源码明确disable_cuda，只给XPU测试入口"),
    26: ("T5 mask+dropout并返回K/V", "源码明确disable_cuda，只给XPU测试入口"),
    27: ("T5零mask+dropout并返回K/V", "源码明确disable_cuda，只给XPU测试入口"),
    28: ("Visformer非连续QKV", "融合前显式contiguous，避免布局错误"),
    29: ("BERT式分摊sqrt缩放并带mask", "把Q/K双侧scale合并为scale平方"),
    30: ("pattern 29的无mask形式", "融合双侧scale与_safe_softmax"),
}


def iso_to_cst(value: str) -> str:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("时间戳必须包含时区")
    return parsed.strftime("%Y-%m-%d %H:%M:%S") + " CST（UTC+08:00）"


def task_patterns(task: str) -> list[int]:
    start = 1 + (int(task[-3:]) - 102) * 5
    values = list(range(start, start + 5))
    return [value for value in values if value not in {25, 26, 27}]


def gpu_nodeid(pattern: int) -> str:
    method = (
        "test_sdpa_rewriter_16_inference_gpu"
        if pattern == 16
        else f"test_sdpa_rewriter_{pattern}_gpu"
    )
    return f"test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.{method}"


def attention_unit(pattern: int) -> dict:
    intent, detail = PATTERN_INTENTS[pattern]
    nodeid = gpu_nodeid(pattern)
    gpu_behavior = (
        "CUDA命中专属pattern counter；replacement保留原数学链以满足紧精度合同"
        if pattern == 16
        else "CUDA应命中该编号pattern，并生成scaled-dot-product attention等价实现"
    )
    return {
        "acceptance_unit_id": f"AU-fuse-attention-sfdp-pattern-{pattern}",
        "contract_name": f"SDPA pattern {pattern}：{intent}",
        "stage": "joint_graph",
        "review_status": "prepared-awaiting-gpu-reference",
        "denominator_eligible": "yes-provisional",
        "coverage_phase": "awaiting-gpu-reference",
        "upstream_sources": [{
            "path": "torch/_inductor/fx_passes/fuse_attention.py",
            "line": PATTERN_LINES[pattern],
            "symbol": f"_sfdp_pattern_{pattern}",
            "role": "joint-graph-register-replacement-search-function",
        }],
        "community_tests": [{
            "nodeid": nodeid,
            "role": "primary-native-gpu-product",
            "evidence_scope": "真实CUDA torch.compile；数值、必要梯度、fuse_attention计数和生成代码；精确编号仍需FX/per-pattern counter复核",
        }],
        "variants": [{
            "variant_id": f"pattern-{pattern}-community-product",
            "kind": "positive-product",
            "expected_reference_match": True,
            "expected_behavior": f"{gpu_behavior}；{detail}",
            "expected_counter": f"inductor_pattern_matcher_per_pattern中_sfdp_pattern_{pattern}前缀>=1",
            "reference_status": "not-run",
            "npu_status": "not-run",
        }],
        "tracking": {
            "reference_mode": "direct",
            "npu_mode": "triton-experimental-derived-from-community-functional-case",
            "allowed_local_deviation": ["NPU使用注册器实际生成的同编号half inference输入，设备替换为NPU并保持pattern合同"],
            "forbidden_local_deviation": ["用通用fuse_attention总计数冒充精确编号", "复用default/DVM/MLIR结果", "把SDPA kernel benchmark冒充rewrite benchmark"],
        },
        "community_alignment": {
            "status": "backend-specific-partial-alignment" if pattern == 16 else "pending-device-comparison",
            "gpu_scope": gpu_behavior,
            "npu_scope": "必须在triton_experimental真机确认精确编号、数值和无fallback后才能定论",
        },
        "npu_control": {
            "config": "use_joint_graph_passes on/off on an isolated target-only workload",
            "off": False,
            "on": True,
            "status": "prepared",
            "next_action": "GPU原生reference人工复核后，在NPU运行独立fresh-process OFF/ON功能臂；功能通过才签性能gate",
        },
    }


def fsdp_unit() -> dict:
    nodeid = "test/distributed/test_inductor_collectives.py::TestCollectivesInductor.test_dedup_reduce_scatter"
    return {
        "acceptance_unit_id": "AU-fsdp-get-dedup-rs",
        "contract_name": "线性reduce-scatter相加去重",
        "stage": "post_grad",
        "review_status": "prepared-awaiting-gpu-reference",
        "denominator_eligible": "yes-provisional",
        "coverage_phase": "awaiting-gpu-reference",
        "upstream_sources": [{"path": "torch/_inductor/fx_passes/fsdp.py", "line": 101, "symbol": "_get_dedup_rs_pass", "role": "lazy-pattern-pass-builder"}, {"path": "torch/_inductor/fx_passes/fsdp.py", "line": 167, "symbol": "dedup_fsdp_reduce_scatter", "role": "fixpoint-pass-entry"}],
        "community_tests": [{"nodeid": nodeid, "role": "native-gpu-structural-positive", "evidence_scope": "真实CUDA compile与生成代码计数；默认world_size=1，只证明2个RS改成1个及数值，不证明多rank通信"}],
        "variants": [{"variant_id": "two-to-one-linear-rs-positive", "kind": "positive", "expected_reference_match": True, "expected_behavior": "RS(a)+RS(b)改写为RS(a+b)，生成代码reduce_scatter从2个降为1个", "expected_counter": "generated code exact reduce_scatter count=1", "reference_status": "not-run", "npu_status": "not-run"}],
        "tracking": {"reference_mode": "direct", "npu_mode": "triton-experimental-two-rank-derived-extension", "allowed_local_deviation": ["NPU补2-rank HCCL合同弥补原生CUDA用例world_size=1边界，代数与shape保持一致"], "forbidden_local_deviation": ["把world_size=1外推成多卡收益", "使用FakeTensor/CPU证明设备通信", "复用非triton_experimental历史结果"]},
        "npu_control": {"config": "dedup_reduce_scatters", "off": False, "on": True, "status": "prepared-two-rank-required", "next_action": "GPU原生结构reference通过后，NPU用2-rank HCCL确认2→1改写、数值和通信，再签性能gate"},
    }


SELECTED = {
    **{task: [attention_unit(pattern) for pattern in task_patterns(task)] for task in ATTENTION_TASKS},
    "T-112": [fsdp_unit()],
}


def deferred_reason(task: str, unit_id: str) -> tuple[str, str]:
    if task == "T-101":
        return "merged-existing-coverage-t085-stale-symbol-mapping", "当前源码的replacement属于partitioned scatter优化，已由T-085完成原生GPU、NPU功能和性能闭环；旧索引四条pattern-matcher基础设施测试不执行该优化。"
    if unit_id.endswith(("pattern-25", "pattern-26", "pattern-27")):
        return "deferred-explicit-cuda-disabled-xpu-only", "注册extra_check明确disable_cuda，GPU类也只在HAS_XPU_AND_TRITON时暴露该编号；禁止绕过CUDA关闭制造reference。"
    if task in {"T-108", "T-109"}:
        return "deferred-cpu-mkldnn-quantization-only", "该入口生成quantized_decomposed/MKLDNN CPU lowering；映射测例为空或只走CPU/MKLDNN，不能证明CUDA/NPU合同。"
    if task in {"T-110", "T-111"}:
        return "deferred-mkldnn-cpu-only-or-mismapped", "源码受MKLDNN可用性与CPU/XPU packed op约束；列出的CUDA卷积测试走通用select_algorithm，不执行MKLDNN fusion，其他测例为CPU。"
    if task == "T-113":
        return "deferred-structural-dispatcher-concrete-fusions-counted-separately", "group_batch_fusion_passes只按配置实例化并调度具体fusion；映射测例分别属于batch-linear、cat-linear或纯subset helper，容器不独立计功能/性能分母。"
    raise KeyError((task, unit_id))


def manifest(task: str, timestamp: str, original: dict) -> dict:
    units = SELECTED.get(task, [])
    selected_ids = {unit["acceptance_unit_id"] for unit in units}
    deferred = []
    for candidate in original[task]["units"]:
        unit_id = candidate["provisional_unit_id"]
        if unit_id in selected_ids:
            continue
        status, reason = deferred_reason(task, unit_id)
        deferred.append({
            "provisional_unit_id": unit_id,
            "status": status,
            "reason": reason,
            "denominator_eligible": False,
            "performance_readiness": "not-applicable-until-independent-gpu-contract",
            "next_action": "仅当冻结源码出现可独立归属的原生GPU合同且不违反显式关闭时重新审核；当前不执行、不计数",
        })
    ready = bool(units)
    return {
        "schema_version": "1.0",
        "generated_at": timestamp,
        "status": "prepared-awaiting-gpu-reference" if ready else "reviewed-no-gpu-ready-units",
        "source_baselines": {"pytorch": {"commit": COMMIT, "branch": "release/2.14", "relevant_tracked_files_state": "clean"}},
        "mapping_review": {
            "input_scope": f"{task}全部{len(original[task]['units'])}个provisional候选逐项复核源码、社区测例、设备入口、目标归属和合法OFF/ON",
            "selected_acceptance_units": sorted(selected_ids),
            "corrections": ["模板私有方法改为实际可执行GPU类nodeid。", "CPU/Fake/间接/错配测试不冻结GPU分母。", "显式CUDA关闭、重复实现和结构容器均不制造ON路径。"],
        },
        "reference_contract": {
            "device_class": "GPU",
            "backend": "inductor-default",
            "execution_order": "native-community-test-first-then-minimal-adapter-if-required",
            "minimum_gpus": 1 if ready else 0,
            "distributed_world_size": 1 if task == "T-112" else None,
            "scope": "原生CUDA编译、数值/梯度、目标编号或生成代码改写" if ready else "本批无合法GPU合同，只保留静态审核和延期原因",
            "suite_status": "pending-gpu-reference" if ready else "reviewed-no-gpu-ready-units",
        },
        "counting_policy": {"current_manifest_units": len(units), "current_frozen_denominator_units": 0, "current_formally_closed_units": 0, "denominator_rule": "原生GPU suite有效、零skip并完成人工目标归属复核后冻结"},
        "acceptance_units": units,
        "deferred_candidates": deferred,
    }


def reference_plan(task: str, timestamp: str, units: list[dict]) -> dict:
    cases = []
    for unit in units:
        pattern = unit["acceptance_unit_id"].removeprefix("AU-fuse-attention-sfdp-pattern-")
        if pattern.isdigit():
            case_id = f"REF-sfdp-pattern-{pattern}-native"
            assertions = ["真实CUDA torch.compile且零skip", f"人工FX/per-pattern counter确认编号{pattern}命中", "输出及社区要求的梯度与eager一致"]
        else:
            case_id = "REF-dedup-reduce-scatter-native"
            assertions = ["真实CUDA torch.compile且零skip", "生成代码reduce_scatter exact count=1", "数值与未融合函数一致；只宣称world_size=1结构证据"]
        cases.append({
            "case_id": case_id,
            "acceptance_unit_id": unit["acceptance_unit_id"],
            "source_test": unit["community_tests"][0]["nodeid"],
            "tracking_mode": "direct",
            "variant_ids": [variant["variant_id"] for variant in unit["variants"]],
            "expected_match": True,
            "expected_assertions": assertions,
            "required_artifacts": ["fx-before", "fx-after"],
            "benchmark": "not-configured-functional-reference",
        })
    return {
        "schema_version": "1.0",
        "task_id": task,
        "generated_at": timestamp,
        "status": "gpu-ready-awaiting-native-execution" if units else "reviewed-no-gpu-ready-units",
        "performance_plan": {"path": f"upstream/{task.lower().replace('-', '')}_performance_plan.yaml", "status": "implemented-awaiting-runtime-validation" if units else "not-applicable"},
        "case_guide": {"path": f"docs/{task.replace('-', '')}_FUNCTION_PERFORMANCE_GUIDE.md", "language": "zh-CN"},
        "manifest": {"path": f"upstream/{task.lower().replace('-', '')}_manifest.yaml", "schema_version": "1.0", "pytorch_commit": COMMIT},
        "execution_policy": {"order": "native-community-test-first-then-minimal-adapter-if-required", "case_isolation": "fresh-process", "failure_behavior": "continue-and-record", "artifact_capture": "原生TORCH_COMPILE_DEBUG与FX before/after", "benchmark_gate": "functional-reference-valid-first", "default_timeout_seconds": 7200 if task in ATTENTION_TASKS else 3600, "minimum_gpus": 1 if units else 0, "distributed_world_size": 1 if task == "T-112" else None},
        "cases": cases,
        "non_executed_variants": [],
    }


def performance_unit(task: str, unit: dict) -> dict:
    nodeid = unit["community_tests"][0]["nodeid"]
    if task in ATTENTION_TASKS:
        pattern = int(unit["acceptance_unit_id"].rsplit("-", 1)[1])
        return {
            "acceptance_unit_id": unit["acceptance_unit_id"],
            "worker_unit": f"pattern-{pattern}",
            "performance_status": "prepared-awaiting-functional-gate",
            "verdict": "planned",
            "case_source": {"kind": "tracker-derived-from-community-functional-case", "nodeids": [nodeid], "reason": "社区transformer/sdpa benchmark直接从已融合SDPA起步，不能测FX rewrite；worker改用注册器为同编号生成的half inference输入，保留实际pattern shape/stride/dtype/scalar合同。"},
            "functional_gate": f"GPU原生reference有效；NPU triton_experimental OFF无fuse_attention，ON精确_sfdp_pattern_{pattern} counter>=1；数值正确且无break/fallback。",
            "workloads": [{"workload_id": f"sfdp-pattern-{pattern}-registered-half-inference", "role": "target-level-derived", "shape_contract": "由冻结源码_get_sfdp_patterns(device)选择同编号half inference首个非bs1候选并把完整shape/stride/dtype写入结果", "measurement_scope": "单个attention rewrite微图；不是社区原生benchmark，也不是模型端到端"}],
            "off_on_control": "目标隔离图中只切use_joint_graph_passes false/true；每臂fresh process，并以精确编号counter排除错误归属。",
            "artifacts": ["GPU原生FX/计数", "NPU OFF/ON功能原件", "精确pattern counter与实际input_spec", "host/Event原始样本、显存、编译耗时", "worker/source sha256"],
            "negative_cases": ["ON未命中精确编号、OFF仍命中、数值/梯度错误、fallback或显式产品disable均拒绝计时", "不得把其他joint pass变化归因给本pattern"],
        }
    return {
        "acceptance_unit_id": unit["acceptance_unit_id"],
        "worker_unit": "dedup-reduce-scatter-two-rank",
        "performance_status": "prepared-awaiting-gpu-and-two-rank-npu-functional-gates",
        "verdict": "planned",
        "case_source": {"kind": "tracker-derived-from-community-functional-case", "nodeids": [nodeid], "reason": "社区只有world_size=1功能例且没有目标benchmark；性能worker保持RS(a)+RS(b)代数，扩成2-rank真实NCCL/HCCL和大hidden代表负载。"},
        "functional_gate": "GPU原生world_size=1结构证据有效；NPU triton_experimental 2-rank HCCL两侧数值一致，OFF代码2个RS、ON代码1个RS，无break/fallback。",
        "workloads": [{"workload_id": "dedup-reduce-scatter-two-rank-fp32", "role": "distributed-derived", "shape_contract": "world_size=2；每rank两个[256,4096] fp32输入，各RS输出[128,4096]", "measurement_scope": "真实2-rank collective微图；不是FSDP模型端到端"}],
        "off_on_control": "只切dedup_reduce_scatters false/true，reorder_for_compute_comm_overlap固定false；每臂由torchrun新进程启动2 rank。",
        "artifacts": ["GPU world_size=1原生结构证据", "NPU 2-rank每rank生成代码和结果", "HCCL backend/2→1计数", "host/Event样本、显存、编译耗时", "worker sha256"],
        "negative_cases": ["单rank结果不得宣称通信收益", "任一rank失败、ON非1个RS、OFF非2个RS、fallback或数值错误均拒绝计时"],
    }


def performance_plan(task: str, timestamp: str, units: list[dict]) -> dict:
    if task in ATTENTION_TASKS:
        implementation = {"status": "implemented-awaiting-runtime-validation", "entrypoint": ATTENTION_WORKER, "launcher": ATTENTION_LAUNCHER}
        devices, world = 1, 1
    elif task == "T-112":
        implementation = {"status": "implemented-awaiting-runtime-validation", "entrypoint": FSDP_WORKER, "launcher": FSDP_LAUNCHER}
        devices, world = 2, 2
    else:
        implementation = {"status": "not-implemented", "entrypoint": None, "reason": "本批没有合法GPU-ready单元，不派生性能worker"}
        devices, world = 0, 0
    return {
        "schema_version": "1.0", "task_id": task, "generated_at": timestamp,
        "status": "prepared-awaiting-gpu-and-npu-functional-gates" if units else "not-applicable-no-gpu-ready-unit",
        "implementation": implementation,
        "backend_contract": {"npu_backend": "triton_experimental", "selection_timing": "新进程内导入torch/torch_npu之前设置TORCHINDUCTOR_NPU_BACKEND", "process_isolation": "fresh-process-per-arm", "off_on_order": ["OFF1", "ON1", "ON2", "OFF2", "OFF3", "ON3"], "explicit_disable_policy": "产品明确disable则记录并免测；generic guard遗漏才进入单独最小适配审核"},
        "measurement_contract": {"correctness_gate": "GPU原生reference有效后，NPU先验证OFF/ON数值、精确改写、零graph break/CPU fallback，再人工签gate", "warmup": 10, "runs": 100, "timing": ["host synchronized wall time p50/p99", "device Event p50/p99", "compile latency separate", "raw samples"], "memory": ["peak allocated", "peak reserved"], "verdict_thresholds": {"improved_percent": 5, "regressed_percent": -5, "otherwise": "PERF_NEUTRAL"}, "rounds": 3, "aggregation": "三OFF/三ON分别聚合；分布式要求全部rank有效"},
        "hardware_contract": {"gpu_reference_minimum_devices": 1 if units else 0, "performance_minimum_devices": devices, "performance_world_size": world, "actual_device_execution_required": bool(units), "fake_tensor_allowed_as_performance_oracle": False},
        "benchmark_search": {"revision": COMMIT, "paths": ["benchmarks/", "test/inductor/"], "terms": [source["symbol"] for unit in units for source in unit["upstream_sources"]] or [task], "result": "attention现有社区benchmark从已融合SDPA起步；FSDP只找到单rank功能例；其余空批次无合法设备合同。", "source_priority": "社区目标级benchmark优先；没有时保留功能图/shape/dtype/阶段派生并明确非社区benchmark、非模型端到端"},
        "acceptance_units": [performance_unit(task, unit) for unit in units],
    }


def attention_guide(task: str, timestamp: str, data: dict) -> str:
    lines = [f"# {task} 功能与性能测例讲解", "", f"> 更新时间：{iso_to_cst(timestamp)}", f"> 状态：{len(data['acceptance_units'])} 个GPU-ready单元，{len(data['deferred_candidates'])} 个明确延期。", "", "NPU功能、修复验证和性能统一使用 `triton_experimental`；后端在导入`torch`/`torch_npu`前选择，OFF/ON每臂使用新进程。", "", "## GPU 一键运行", "", "```bash", "cd /data/z50063656/tmp", "", "bash \\", "  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \\", f"  --task {task} \\", "  --gpu 2 \\", "  --wait-gpu", "```", "", "社区GPU类由模板方法赋值/`functools.partialmethod`形成；runner只做静态解析，不导入PyTorch探测。每条测试一般覆盖多个dtype和训练/推理，分母仍按一个编号pattern合同计。", "", "## 功能测例", ""]
    for unit in data["acceptance_units"]:
        pattern = int(unit["acceptance_unit_id"].rsplit("-", 1)[1])
        intent, detail = PATTERN_INTENTS[pattern]
        lines += [f"### {unit['acceptance_unit_id']}", "", "```python", f"# torch/_inductor/fx_passes/fuse_attention.py:{PATTERN_LINES[pattern]}", f"def _sfdp_pattern_{pattern}(...):", f"    # {intent}", "    return explicit_matmul_softmax_attention", "", f"def _sfdp_replacement_{pattern}(...):", "    counters[\"inductor\"][\"fuse_attention\"] += 1", "    return _scaled_dot_product_attention(...)  # 或pattern 16的CUDA保真数学路径", "```", "", f"意图：{detail}。GPU执行 `{gpu_nodeid(pattern)}`；必须把通用`fuse_attention`计数与精确`_sfdp_pattern_{pattern}` counter/FX对应起来。NPU目前仅为待验证：必须在`triton_experimental`下证明同编号命中、数值正确且无fallback，不能借用CUDA或其他NPU backend结论。", ""]
    lines += ["## 性能测例", "", "社区 `benchmarks/transformer/sdpa.py` 已直接调用 fused SDPA，只能比较kernel，不能隔离本FX rewrite。性能worker因此调用冻结源码 `_get_sfdp_patterns(device)`，选择同编号half-inference注册输入；它保留真实shape、stride、dtype和scalar workaround，但明确属于tracker派生微图，不是社区原生benchmark或模型端到端。", "", "```bash", "cd /home/z50063656/tmp", "", "python \\", "  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t102_t107_attention_performance.py \\", f"  --task {task} \\", "  --phase functional \\", "  --device npu", "```", "", "功能原件人工签gate后，才改用`--phase benchmark --gate-root <目录>`运行OFF1/ON1/ON2/OFF2/OFF3/ON3。OFF使用隔离目标图并关闭joint_graph整轮，结论只归属于该图；若发现其他joint pass变化，性能结果作废。", "", "## 延期候选", "", "| 候选 | 状态 | 原因 |", "| --- | --- | --- |"]
    for item in data["deferred_candidates"]:
        lines.append(f"| `{item['provisional_unit_id']}` | `{item['status']}` | {item['reason']} |")
    if not data["deferred_candidates"]:
        lines.append("| 无 | — | 本批全部候选已进入GPU-ready合同。 |")
    return "\n".join(lines) + "\n"


def guide(task: str, timestamp: str, data: dict) -> str:
    if task in ATTENTION_TASKS:
        return attention_guide(task, timestamp, data)
    lines = [f"# {task} 功能与性能测例讲解", "", f"> 更新时间：{iso_to_cst(timestamp)}", f"> 状态：{len(data['acceptance_units'])} 个GPU-ready单元，{len(data['deferred_candidates'])} 个明确延期。", "", "NPU功能、修复验证和性能统一使用 `triton_experimental`，并在导入`torch`/`torch_npu`前选后端；OFF/ON每臂使用新进程。", "", "## 功能测例", ""]
    if task == "T-101":
        lines += ["```python", "# torch/_inductor/fx_passes/reduced_atomic_contention.py", "def replacement(index, index_size, src, dim_size):", "    # partitioned scatter optimization；真实合同已归入T-085", "    ...", "```", "", "旧索引把同名内层`replacement`错误关联到四条pattern-matcher注册机制测试。当前实现、GPU/NPU证据和性能处置均已由T-085覆盖，因此这里合并旧ID，不重复计数。", ""]
    elif task in {"T-108", "T-109"}:
        lines += ["```python", "# torch/_inductor/fx_passes/quantization.py", "quantized_decomposed = torch.ops.quantized_decomposed", "# qconv/qlinear/woq注册最终生成MKLDNN/quantized CPU lowering", "```", "", "这些候选是CPU量化/MKLDNN路径。没有原生CUDA目标命中合同，也没有可合法转成NPU `triton_experimental` 的同一实现；因此不是‘社区开启却漏测’，而是本GPU→NPU审计范围不适用。", ""]
    elif task in {"T-110", "T-111"}:
        lines += ["```python", "# torch/_inductor/fx_passes/mkldnn_fusion.py:41", "if torch._C._has_mkldnn:", "    mkldnn = torch.ops.mkldnn", "# packed convolution/linear/rnn与unary/binary lowerings", "```", "", "MKLDNN fusion面向CPU（部分XPU）packed算子。旧索引中的CUDA convolution测例走通用卷积/select_algorithm，不会调用这里的MKLDNN fusion，故不能作为GPU reference。", ""]
    elif task == "T-112":
        lines += ["### AU-fsdp-get-dedup-rs", "", "```python", "# torch/_inductor/fx_passes/fsdp.py:101,167", "# before: wait(RS(a)) + wait(RS(b))", "combined = aten.add.Tensor(input_a, input_b)", "rs = c10d.reduce_scatter_tensor.default(combined, reduce_op, group_size, group_name)", "return c10d.wait_tensor.default(rs)", "```", "", "该pass利用sum/avg reduce-scatter的线性性质，把两个通信归并为一个。社区CUDA测试是真实compile并检查生成代码只有1个RS，但默认`world_size=1`，只能冻结结构改写，不能证明跨rank通信收益。NPU功能扩展固定2-rank HCCL：OFF必须2个RS、ON必须1个RS，全部rank数值一致。", "", "```bash", "cd /data/z50063656/tmp", "", "bash \\", "  /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_gpu_reference_task.sh \\", "  --task T-112 \\", "  --gpu 2 \\", "  --wait-gpu", "```", "", "## 性能测例", "", "社区没有目标性能benchmark。派生worker保持同一代数，扩为2-rank、每rank两个`[256,4096]` fp32输入，通过NCCL/HCCL执行；它是collective微图，不是FSDP模型端到端。", "", "```bash", "cd /home/z50063656/tmp", "", "python \\", "  /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t112_dedup_reduce_scatter.py \\", "  --device npu \\", "  --phase functional", "```", "", "功能与生成代码人工签gate后才运行六臂性能；任一rank失败都会使整组无效。", ""]
    else:
        lines += ["```python", "# torch/_inductor/fx_passes/group_batch_fusion.py:1679", "def group_batch_fusion_passes(graph, pre_grad=True, fusion_options=None):", "    fusions = generate_fusion_from_config(...)  # 调度具体fusion", "    for rule in fusions:", "        rule.apply(graph)", "```", "", "这是调度容器，不是一个独立数学改写。映射测试实际归属于batch-linear、cat-linear或纯subset选择helper；具体fusion应各自形成合同，容器不能再计一个功能/性能分母。", ""]
    if task != "T-112":
        lines += ["## 性能测例", "", "本批没有通过真实GPU目标合同的独立单元，因此不派生OFF/ON worker、不执行设备性能，也不制造NPU ON路径。", ""]
    lines += ["## 延期候选", "", "| 候选 | 状态 | 原因 |", "| --- | --- | --- |"]
    for item in data["deferred_candidates"]:
        lines.append(f"| `{item['provisional_unit_id']}` | `{item['status']}` | {item['reason']} |")
    if not data["deferred_candidates"]:
        lines.append("| 无 | — | 唯一候选已进入GPU-ready合同。 |")
    return "\n".join(lines) + "\n"


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
        raise ValueError("backlog未完整保留T-101～T-113")
    if not args.write:
        print("prepared_generation_check=OK tasks=13 selected=28 deferred=24")
        return
    for task in TASKS:
        suffix = task.lower().replace("-", "")
        data = manifest(task, args.timestamp, original)
        payloads = {
            root / "upstream" / f"{suffix}_manifest.yaml": data,
            root / "upstream" / f"{suffix}_reference_plan.yaml": reference_plan(task, args.timestamp, data["acceptance_units"]),
            root / "upstream" / f"{suffix}_performance_plan.yaml": performance_plan(task, args.timestamp, data["acceptance_units"]),
        }
        for path, value in payloads.items():
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (root / "docs" / f"{task.replace('-', '')}_FUNCTION_PERFORMANCE_GUIDE.md").write_text(guide(task, args.timestamp, data), encoding="utf-8")
        wrapper = root / "scripts" / f"run_{suffix}_reference_all.sh"
        wrapper.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "repo_root=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd -P)\"\n"
            f"exec bash \"${{repo_root}}/scripts/run_reference_all.sh\" \\\n  --manifest-path upstream/{suffix}_manifest.yaml \\\n  --plan-path upstream/{suffix}_reference_plan.yaml \\\n  \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        incoming = root / "results" / "incoming" / task
        incoming.mkdir(parents=True, exist_ok=True)
        incoming.joinpath("README.md").write_text(
            f"# {task} GPU handoff 接收目录\n\n> 更新时间：{args.timestamp}\n\n仅提交统一GPU runner生成并校验通过的文本handoff。零GPU-ready批次只保留审核入口，不上传伪结果。\n",
            encoding="utf-8",
        )
    print("prepared_generation_write=OK tasks=13 selected=28 deferred=24")


if __name__ == "__main__":
    main()
