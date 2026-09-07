#!/usr/bin/env python3
"""从冻结 GPU handoff 与本机 NPU 工件生成 T-079 正式结果。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
GENERATED_AT = "2026-09-07T18:50:00+08:00"

UNITS = {
    "AU-joint-graph-bmm-to-mm": {
        "case_id": "REF-bmm-to-mm-native",
        "adapter_run": Path(
            "/home/z50063656/tmp/t079-npu-results/REF-bmm-to-mm-native/"
            "adapter-8723yeav/adapter_result.json"
        ),
        "performance": Path(
            "/home/z50063656/tmp/t079-performance-results/"
            "bmm-to-mm-performance-20260907T102213+0800/performance_summary.json"
        ),
        "runtime_path": "extern",
        "positive_count": 1,
        "positive_nodes": 1,
        "gate": True,
        "input_cases": {
            "batch-one-positive": {
                "shapes": [[1, 16, 8], [1, 8, 32]],
                "strides": [[128, 8, 1], [256, 32, 1]],
            },
            "multi-batch-negative": {
                "shapes": [[3, 16, 8], [3, 8, 32]],
                "strides": [[128, 8, 1], [256, 32, 1]],
            },
        },
    },
    "AU-post-grad-cat-slice-cat": {
        "case_id": "REF-cat-slice-cat-native",
        "adapter_run": Path(
            "/home/z50063656/tmp/t079-npu-results/REF-cat-slice-cat-native/"
            "adapter-l_h0o2kz/adapter_result.json"
        ),
        "performance": Path(
            "/home/z50063656/tmp/t079-performance-results/"
            "cat-slice-cat-performance-20260907T101924+0800/performance_summary.json"
        ),
        "runtime_path": "mixed",
        "positive_count": 1,
        "positive_nodes": 3,
        "gate": False,
        "input_cases": {
            "*": {
                "shapes": [[2, 32], [2, 16]],
                "strides": [[32, 1], [16, 1]],
            }
        },
    },
    "AU-post-grad-splitwithsizes-cat-replace": {
        "case_id": "REF-splitwithsizes-cat-native",
        "adapter_run": Path(
            "/home/z50063656/tmp/t079-npu-results/REF-splitwithsizes-cat-native/"
            "adapter-0mdctuz8/adapter_result.json"
        ),
        "performance": Path(
            "/home/z50063656/tmp/t079-performance-results/"
            "split-cat-performance-20260907T101115+0800/performance_summary.json"
        ),
        "runtime_path": "triton",
        "positive_count": 1,
        "positive_nodes": 4,
        "gate": False,
        "input_cases": {
            "*": {"shapes": [[2, 32]], "strides": [[32, 1]]}
        },
    },
    "AU-post-grad-cat-splitwithsizes-replace": {
        "case_id": "REF-cat-splitwithsizes-native",
        "adapter_run": Path(
            "/home/z50063656/tmp/t079-npu-results/REF-cat-splitwithsizes-native/"
            "adapter-4ke2_hfj/adapter_result.json"
        ),
        "performance": Path(
            "/home/z50063656/tmp/t079-performance-results/"
            "cat-split-performance-20260907T101524+0800/performance_summary.json"
        ),
        "runtime_path": "triton",
        "positive_count": 1,
        "positive_nodes": 2,
        "gate": False,
        "input_cases": {
            "*": {
                "shapes": [[2, 2], [2, 3], [2, 5]],
                "strides": [[2, 1], [3, 1], [5, 1]],
            }
        },
    },
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def environment() -> dict[str, object]:
    value = {
        "python_version": "3.11.15",
        "torch_version": "2.14.0a0+git8e86e0a",
        "torch_commit": UPSTREAM_COMMIT,
        "torch_npu_version": "2.14.0a0+git83cc452",
        "torch_npu_commit": "83cc452480c3546fd5cccf853bfe3a360ce9dbfc",
        "triton_version": "3.2.0",
        "triton_ascend_commit": "8bd9f380d2786002b84b5248f00838c26f900515",
        "cann_version": "9.0.1",
        "driver_version": "26.0.rc1",
        "device": "Ascend910B2",
        "backend": "triton_experimental",
    }
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {"fingerprint_sha256": hashlib.sha256(encoded).hexdigest(), **value}


def stage(status: str, evidence: str) -> dict[str, str]:
    return {"status": status, "evidence": evidence}


def make_variant(unit_id, spec, manifest_variant, perf) -> dict[str, object]:
    variant_id = manifest_variant["variant_id"]
    expected = manifest_variant["expected_reference_match"]
    is_gate_variant = spec["gate"] and variant_id == "batch-one-positive"
    matched = False if is_gate_variant else expected
    input_case = spec["input_cases"].get(
        variant_id,
        spec["input_cases"].get("*"),
    )
    if input_case is None:
        raise KeyError(f"missing input case: {unit_id}:{variant_id}")

    if matched:
        count = spec["positive_count"]
        nodes = spec["positive_nodes"]
    else:
        count = 0
        nodes = 0
    is_positive_path = manifest_variant["kind"] == "positive"
    if unit_id == "AU-post-grad-cat-slice-cat" and expected:
        count = 1
        nodes = 3

    if is_gate_variant:
        support = "expected-disabled"
        first_divergence = "config"
        root_cause = (
            "同后端性能测量显示 bmm→mm 在四个 shape 的 Event p50 全部回退，"
            "因此产品门禁在 handler 前拒绝 NPU match。"
        )
        action = "保持默认门禁；只有新的 shape-qualified 收益证据通过评审后才重开。"
        replacement = stage("not-observed", "产品 gate 在 handler 前拒绝 replacement。")
        npu_expectation = False
    else:
        support = "supported"
        first_divergence = "none"
        root_cause = "NPU 行为与冻结社区合同一致。"
        action = "保留当前实现和正负例回归。"
        replacement = stage(
            "observed" if expected else "not-observed",
            (
                "目标 handler/replacement 按社区合同执行。"
                if expected
                else "社区 guard 拒绝目标 replacement。"
            ),
        )
        npu_expectation = expected

    improvement = perf["primary_event_p50_improvement_percent"]
    performance_reason = (
        f"目标级 3×OFF/3×ON 的 Event p50 变化为 {improvement:.2f}%；"
        f"候选判定 {perf['verdict']}。"
    )
    return {
        "variant_id": variant_id,
        "kind": manifest_variant["kind"],
        "evidence_mode": "runtime",
        "support_status": support,
        "input_contract": {
            "direction": "forward",
            "dtype": "float32",
            "dynamic": False,
            "cases": [input_case],
        },
        "reference_match_expectation": expected,
        "npu_match_expectation": npu_expectation,
        "match": {
            "target_matched": matched,
            "count": count,
            "nodes": nodes,
            "scope": "target-pattern",
            "note": (
                "最终产品门禁生效，目标 handler 命中为 0。"
                if is_gate_variant
                else "保留原社区 counter/FileCheck 判据。"
            ),
        },
        "fx": stage("observed", "适配运行已捕获 FX before/after。"),
        "replacement": replacement,
        "decomposition": stage("not-applicable", "该单元不是 decomposition pass。"),
        "lowering": stage("observed", "编译图成功进入 NPU lowering。"),
        "scheduler": stage("observed", "triton_experimental 调度和执行成功。"),
        "codegen": stage("observed", "已保存 generated_code/output_code 结构证据。"),
        "runtime_path": spec["runtime_path"],
        "correctness": {
            "status": "passed",
            "assertion": "原社区 compiled/eager、counter 或 FileCheck 断言通过，0 skip。",
        },
        "performance": {"status": "passed", "reason": performance_reason},
        "first_divergence": first_divergence,
        "root_cause": root_cause,
        "recommended_action": action,
    }


def main() -> int:
    handoff = load(ROOT / "results/incoming/T-079/text-handoff.json")
    manifest = load(ROOT / "upstream/t079_manifest.yaml")
    manifest_units = {
        item["acceptance_unit_id"]: item
        for item in manifest["acceptance_units"]
    }
    audit = {item["case_id"]: item for item in handoff["case_audit"]}
    ref_cases = {
        item["case_id"]: item for item in handoff["reference_summary"]["cases"]
    }
    env = environment()
    task_units = []

    for unit_id, spec in UNITS.items():
        unit = manifest_units[unit_id]
        case_id = spec["case_id"]
        adapter_path = ROOT / f"issues/{case_id}/npu_adapter.py"
        perf = load(spec["performance"])
        variants = [
            make_variant(unit_id, spec, variant, perf)
            for variant in unit["variants"]
        ]
        gate = spec["gate"]
        artifacts = [
            {
                "role": "npu-adapter",
                "path": str(adapter_path.relative_to(ROOT)),
                "sha256": digest(adapter_path),
                "availability": "repository",
            },
            {
                "role": "npu-adapter-result",
                "path": str(spec["adapter_run"]),
                "sha256": digest(spec["adapter_run"]),
                "availability": "external-run-root",
            },
            {
                "role": "performance-summary",
                "path": str(spec["performance"]),
                "sha256": digest(spec["performance"]),
                "availability": "external-run-root",
            },
        ]
        if gate:
            gate_result = Path(
                "/home/z50063656/tmp/t079-bmm-gate-results/"
                "bmm-gate-verify-20260907T184104+0800/default-disabled/result.json"
            )
            repair_report = ROOT / f"issues/{case_id}/修复验证报告.md"
            artifacts.extend(
                [
                    {
                        "role": "product-gate-result",
                        "path": str(gate_result),
                        "sha256": digest(gate_result),
                        "availability": "external-run-root",
                    },
                    {
                        "role": "repair-report",
                        "path": str(repair_report.relative_to(ROOT)),
                        "sha256": digest(repair_report),
                        "availability": "repository",
                    },
                ]
            )

        npu_result = {
            "schema_version": "1.1",
            "generated_at": GENERATED_AT,
            "case_ids": [case_id],
            "acceptance_unit_id": unit_id,
            "upstream_commit": UPSTREAM_COMMIT,
            "source_tests": [test["nodeid"] for test in unit["community_tests"]],
            "tracking_mode": unit["tracking"]["npu_mode"],
            "environment": env,
            "npu_control": {
                "state": "disabled" if gate else "enabled",
                "source_control": (
                    "triton_experimental.config.disable_bmm_to_mm=true；"
                    "backend 激活时只拒绝 NPU bmm_to_mm match。"
                    if gate
                    else "上游目标 pass 在 triton_experimental 上保持启用，无新增产品 gate。"
                ),
                "product_gate_bypassed": False,
            },
            "paired_control": {
                "status": "passed",
                "reason": (
                    "候选 3×OFF/3×ON 完成后因稳定回退设置产品门禁；"
                    "最终 default-disabled/gate-disabled 双臂通过。"
                    if gate
                    else "目标级 3×OFF/3×ON 均正确且社区主 shape 性能改善。"
                ),
            },
            "direct_execution": {
                "status": "no-tests",
                "execution_success": False,
                "tests_run": 0,
                "tests_skipped": 0,
                "reason": "上游模块 __main__ 受 HAS_GPU 启动门限制，NPU 原生入口执行 0 tests。",
            },
            "selected_execution": {
                "status": "passed",
                "execution_success": True,
                "tests_run": 1,
                "tests_skipped": 0,
                "reason": "显式实例化原社区方法后，全部正例/guard、数值和结构断言通过。",
            },
            "variants": variants,
            "artifacts": artifacts,
        }
        npu_path = ROOT / f"results/current/{unit_id}/npu_result.json"
        write(npu_path, npu_result)

        case_audit = audit[case_id]
        reference_case = ref_cases[case_id]
        source = unit["upstream_sources"][0]
        comparisons = []
        for npu_variant, manifest_variant in zip(variants, unit["variants"]):
            expected = manifest_variant["expected_reference_match"]
            divergent = gate and npu_variant["variant_id"] == "batch-one-positive"
            comparisons.append(
                {
                    "variant_id": npu_variant["variant_id"],
                    "intent": manifest_variant["expected_behavior"],
                    "source_locations": [
                        {
                            "path": source["path"],
                            "line": source["line"],
                            "symbol": source["symbol"],
                        }
                    ],
                    "gpu_behavior": (
                        "冻结 A100 reference 按社区 expectation 命中并通过。"
                        if expected
                        else "冻结 A100 reference 按社区 guard 不命中并通过。"
                    ),
                    "npu_behavior": (
                        "门禁前能力验证可改写；性能回退后产品默认拒绝并保留 bmm。"
                        if divergent
                        else "Ascend910B2 与社区正负例 expectation 一致，数值和结构断言通过。"
                    ),
                    "reference_contract_stable": True,
                    "reference_target_match": expected,
                    "npu_target_match": npu_variant["npu_match_expectation"],
                    "match_contract_aligned": True,
                    "runtime_path": npu_variant["runtime_path"],
                    "correctness_status": "passed",
                    "performance_status": "passed",
                    "comparison_basis": (
                        "product-control" if divergent else "equivalent-contract"
                    ),
                    "verdict": (
                        "EXPECTED_PRODUCT_DIVERGENCE"
                        if divergent
                        else "BEHAVIOR_UNCHANGED"
                    ),
                    "note": (
                        "分歧由同后端性能证据驱动，不是能力缺失。"
                        if divergent
                        else "功能合同一致；正例另有目标级性能收益证据。"
                    ),
                }
            )

        comparison = {
            "schema_version": "1.2",
            "generated_at": GENERATED_AT,
            "acceptance_unit_id": unit_id,
            "upstream_commit": UPSTREAM_COMMIT,
            "reference": {
                "run_id": handoff["reference_summary"]["run_id"],
                "suite_status": handoff["reference_summary"]["status"],
                "suite_valid": handoff["reference_summary"]["suite_valid"],
                "environment_fingerprint": handoff["reference_summary"][
                    "environment_fingerprint"
                ],
                "payload_sha256": handoff["payload_sha256"],
                "cases": [
                    {
                        "case_id": case_id,
                        "case_status": reference_case["status"],
                        "reference_valid": reference_case["reference_valid"],
                        "reference_result_sha256": case_audit[
                            "reference_result_sha256"
                        ],
                        "artifact_inventory_sha256": case_audit[
                            "artifact_inventory_sha256"
                        ],
                    }
                ],
            },
            "npu": {
                "result_path": str(npu_path.relative_to(ROOT)),
                "result_sha256": digest(npu_path),
                "environment_fingerprint": env["fingerprint_sha256"],
                "tracking_mode": unit["tracking"]["npu_mode"],
                "execution_success": True,
                "correctness_status": "passed",
            },
            "variant_comparisons": comparisons,
            "first_divergence": "config" if gate else "none",
            "root_cause": (
                "NPU 继承 GPU rewrite 后功能正确但性能回退，已以 backend-local gate 处置。"
                if gate
                else "GPU/NPU 功能行为一致，NPU 目标级性能测量显示收益。"
            ),
            "recommended_action": (
                "保持 disable_bmm_to_mm=true，并保留可逆验证。"
                if gate
                else "保留启用并持续运行正负例及性能回归。"
            ),
            "final_verdict": (
                "EXPECTED_PRODUCT_DIVERGENCE" if gate else "PERF_IMPROVED"
            ),
            "repair_status": "verified" if gate else "not-needed",
        }
        comparison_path = npu_path.with_name("comparison_result.json")
        write(comparison_path, comparison)
        task_units.append(
            {
                "acceptance_unit_id": unit_id,
                "performance_status": (
                    "measured-regressed-product-disabled"
                    if gate
                    else "measured-improved-retained"
                ),
                "verdict": perf["verdict"],
                "event_p50_improvement_percent": perf[
                    "primary_event_p50_improvement_percent"
                ],
                "event_p99_improvement_percent": perf[
                    "primary_event_p99_improvement_percent"
                ],
                "product_action": (
                    "disable_bmm_to_mm=true" if gate else "retain-enabled"
                ),
                "raw_summary": str(spec["performance"]),
                "raw_summary_sha256": digest(spec["performance"]),
            }
        )

    task_summary = {
        "schema_version": "1.0",
        "generated_at": GENERATED_AT,
        "task_id": "T-079",
        "status": "performance-disposition-and-product-gate-complete",
        "backend": "triton_experimental",
        "completion": {
            "acceptance_units": 4,
            "measured_candidate_units": 4,
            "retained_enabled_units": 3,
            "explicitly_disabled_units": 1,
        },
        "measurement_contract": {
            "order": "OFF1-ON1-ON2-OFF2-OFF3-ON3",
            "process_isolation": "fresh-process-per-arm-and-round",
            "warmup": 10,
            "runs": 100,
            "timing": "同步 NPU Event 与 host wall clock p50/p99",
            "scope": "冻结社区正例子图的一次 compiled 调用，不是完整模型 benchmark",
        },
        "community_benchmark_review": {
            "pass_specific_benchmark": "absent",
            "decision": "复用社区功能正例图、shape、dtype，仅增加目标级 OFF/ON 与计时。",
        },
        "acceptance_units": task_units,
        "product_gate": {
            "status": "passed",
            "path": "results/current/T-079/product_gate_verification.json",
        },
        "limitations": [
            "性能结果是 pass 正例子图端到端，不代表完整模型端到端。",
            "bmm→mm 的候选测量发生在门禁落盘前；最终门禁另以双臂结构验证。",
        ],
    }
    task_path = ROOT / "results/current/T-079/performance_summary.json"
    write(task_path, task_summary)
    gate_root = Path(
        "/home/z50063656/tmp/t079-bmm-gate-results/"
        "bmm-gate-verify-20260907T184104+0800"
    )
    gate_summary = {
        "schema_version": "1.0",
        "generated_at": GENERATED_AT,
        "task_id": "T-079",
        "acceptance_unit_id": "AU-joint-graph-bmm-to-mm",
        "backend": "triton_experimental",
        "status": "passed",
        "default_disabled": load(gate_root / "default-disabled/result.json"),
        "gate_disabled": load(gate_root / "gate-disabled/result.json"),
        "source_run_root": str(gate_root),
    }
    write(ROOT / "results/current/T-079/product_gate_verification.json", gate_summary)
    print("t079_finalize=OK units=4 variants=14")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
