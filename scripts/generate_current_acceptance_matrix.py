#!/usr/bin/env python3
"""从活动 manifest/result 生成当前 acceptance-unit 矩阵。

本脚本只使用 Python 标准库，不导入 torch。旧 251 行 registration 矩阵不参与
当前 verdict；GPU reference 的 inductor-default 与 NPU 的 triton_experimental
分别记录，避免把 reference backend 误判为 NPU backend 混用。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "report/current_acceptance_unit_matrix.csv"
OUTPUT_MD = ROOT / "report/current_acceptance_unit_matrix.md"
REQUIRED_NPU_BACKEND = "triton_experimental"
REFERENCE_BACKEND = "inductor-default"

TASK_FILES = {
    "T-076": {
        "manifest": "upstream/manifest.yaml",
        "performance_plan": "upstream/t076_performance_plan.yaml",
    },
    "T-077": {
        "manifest": "upstream/t077_manifest.yaml",
        "performance_plan": "upstream/t077_performance_plan.yaml",
    },
    "T-078": {
        "manifest": "upstream/t078_manifest.yaml",
        "performance_plan": "upstream/t078_performance_plan.yaml",
    },
    "T-079": {
        "manifest": "upstream/t079_manifest.yaml",
        "performance_plan": "upstream/t079_performance_plan.yaml",
    },
    "T-080": {
        "manifest": "upstream/t080_manifest.yaml",
        "performance_plan": "upstream/t080_performance_plan.yaml",
    },
    "T-081": {
        "manifest": "upstream/t081_manifest.yaml",
        "performance_plan": "upstream/t081_performance_plan.yaml",
    },
    "T-082": {
        "manifest": "upstream/t082_manifest.yaml",
        "performance_plan": "upstream/t082_performance_plan.yaml",
    },
    "T-083": {
        "manifest": "upstream/t083_manifest.yaml",
        "performance_plan": "upstream/t083_performance_plan.yaml",
    },
    "T-084": {
        "manifest": "upstream/t084_manifest.yaml",
        "performance_plan": "upstream/t084_performance_plan.yaml",
    },
    "T-085": {
        "manifest": "upstream/t085_manifest.yaml",
        "performance_plan": "upstream/t085_performance_plan.yaml",
    },
    "T-086": {
        "manifest": "upstream/t086_manifest.yaml",
        "performance_plan": "upstream/t086_performance_plan.yaml",
    },
    "T-087": {
        "manifest": "upstream/t087_manifest.yaml",
        "performance_plan": "upstream/t087_performance_plan.yaml",
    },
    "T-088": {
        "manifest": "upstream/t088_manifest.yaml",
        "performance_plan": "upstream/t088_performance_plan.yaml",
    },
    "T-089": {
        "manifest": "upstream/t089_manifest.yaml",
        "performance_plan": "upstream/t089_performance_plan.yaml",
    },
    "T-090": {
        "manifest": "upstream/t090_manifest.yaml",
        "performance_plan": "upstream/t090_performance_plan.yaml",
    },
    "T-091": {
        "manifest": "upstream/t091_manifest.yaml",
        "performance_plan": "upstream/t091_performance_plan.yaml",
    },
    "T-092": {
        "manifest": "upstream/t092_manifest.yaml",
        "performance_plan": "upstream/t092_performance_plan.yaml",
    },
    "T-093": {
        "manifest": "upstream/t093_manifest.yaml",
        "performance_plan": "upstream/t093_performance_plan.yaml",
    },
    "T-094": {
        "manifest": "upstream/t094_manifest.yaml",
        "performance_plan": "upstream/t094_performance_plan.yaml",
    },
    "T-095": {
        "manifest": "upstream/t095_manifest.yaml",
        "performance_plan": "upstream/t095_performance_plan.yaml",
    },
    "T-096": {
        "manifest": "upstream/t096_manifest.yaml",
        "performance_plan": "upstream/t096_performance_plan.yaml",
    },
    "T-097": {
        "manifest": "upstream/t097_manifest.yaml",
        "performance_plan": "upstream/t097_performance_plan.yaml",
    },
    "T-098": {
        "manifest": "upstream/t098_manifest.yaml",
        "performance_plan": "upstream/t098_performance_plan.yaml",
    },
    "T-099": {
        "manifest": "upstream/t099_manifest.yaml",
        "performance_plan": "upstream/t099_performance_plan.yaml",
    },
    "T-100": {
        "manifest": "upstream/t100_manifest.yaml",
        "performance_plan": "upstream/t100_performance_plan.yaml",
    },
}

for task_number in range(101, 114):
    task_id = f"T-{task_number:03d}"
    suffix = task_id.lower().replace("-", "")
    TASK_FILES[task_id] = {
        "manifest": f"upstream/{suffix}_manifest.yaml",
        "performance_plan": f"upstream/{suffix}_performance_plan.yaml",
    }

FIELDNAMES = [
    "matrix_generated_at",
    "task_id",
    "acceptance_unit_id",
    "canonical_acceptance_unit_id",
    "independent_unit_contribution",
    "gpu_review_path",
    "npu_progress_path",
    "contract_name",
    "stage",
    "manifest_status",
    "review_status",
    "denominator_eligible",
    "variant_count",
    "verified_variant_count",
    "pending_variant_count",
    "coverage_status",
    "community_test_count",
    "reference_backend",
    "reference_status",
    "required_npu_backend",
    "observed_npu_backend",
    "npu_execution_status",
    "npu_correctness_status",
    "comparison_verdict",
    "repair_status",
    "community_alignment_status",
    "community_alignment_source",
    "community_aligned_scope",
    "community_divergent_scope",
    "community_open_scope",
    "community_alignment_disposition",
    "performance_status",
    "performance_verdict",
    "current_phase",
    "updated_at",
    "manifest_path",
    "npu_result_path",
    "comparison_result_path",
    "performance_evidence_path",
]


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON-compatible YAML/JSON：{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"顶层必须是对象：{path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_documents(pattern: str, id_key: str) -> dict[str, tuple[Path, dict]]:
    indexed: dict[str, tuple[Path, dict]] = {}
    for path in sorted(ROOT.glob(pattern)):
        payload = read_json(path)
        item_id = payload.get(id_key)
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{path} 缺少 {id_key}")
        if item_id in indexed:
            raise ValueError(f"{id_key} 重复：{item_id}: {indexed[item_id][0]} / {path}")
        indexed[item_id] = (path, payload)
    return indexed


def performance_backend(plan: dict, item: dict) -> str:
    return str(
        item.get("backend")
        or plan.get("backend_contract", {}).get("npu_backend")
        or ""
    )


def reference_status(unit: dict, reference_contract: dict) -> str:
    eligible = str(unit.get("denominator_eligible", ""))
    if eligible == "yes-frozen":
        status = str(reference_contract.get("suite_status") or "frozen-reference-valid")
        if unit.get("pending_variants"):
            if all(
                str(item.get("reference_status", "")).startswith("valid")
                for item in unit["pending_variants"]
            ):
                return status + "-with-npu-pending-extension"
            return status + "-with-pending-extension"
        return status
    return eligible or "unknown"


def verified_file(relative: str, digest: str) -> Path:
    path = (ROOT / relative).resolve()
    if (not path.is_relative_to(ROOT.resolve()) or not path.is_file()
            or sha256(path) != digest):
        raise ValueError(f"NPU 阶段证据路径或哈希不符：{relative}")
    return path


def verify_progress_artifacts(items: list[dict]) -> None:
    for item in items:
        verified_file(item["path"], item["sha256"])
    if not any(Path(item["path"]).name == "output_code.py" for item in items):
        raise ValueError("NPU 阶段证据缺少生成代码")


def npu_contract_progress(task_id: str, unit_id: str) -> tuple[Path, dict] | None:
    """只消费带原件哈希的功能阶段记录，绝不自动生成正式 comparison/gate。"""
    commit = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
    if unit_id == "AU-post-grad-reorder-for-locality":
        path = ROOT / "results/current/T-087/npu_training_review.json"
        if not path.is_file():
            return None
        data = read_json(path)
        arms = data.get("arms", {})
        if (data.get("task_id") != task_id or data.get("acceptance_unit_id") != unit_id
                or data.get("correctness") != "passed"
                or data.get("backend") != REQUIRED_NPU_BACKEND
                or set(arms) != {"off", "on", "master-off"}
                or data.get("performance_gate_issued") is not False
                or data.get("product_gate_bypassed") is not False):
            raise ValueError("训练阶段记录合同不完整")
        for mode, arm in arms.items():
            if (arm.get("backend") != REQUIRED_NPU_BACKEND or arm.get("torch_commit") != commit
                    or arm.get("numerical_execution") is not True
                    or arm.get("product_gate_bypassed") is not False):
                raise ValueError(f"训练三臂来源/设备不符：{mode}")
        if (arms["off"]["handler_calls"] != 0 or arms["master-off"]["handler_calls"] != 0
                or arms["on"]["handler_calls"] < 1 or not arms["on"]["graph_changed"]):
            raise ValueError("训练三臂目标改写不符")
        errors = data.get("cross_process_max_abs_errors", {})
        if len(errors) != 8 or any(error != 0 for error in errors.values()):
            raise ValueError("本次训练核验的数值证据不符")
        verify_progress_artifacts(data["artifacts"])
        return path, {"contract_complete": True, "updated_at": data["generated_at"]}
    path = ROOT / "results/current" / task_id / "npu_contract_progress.json"
    if not path.is_file():
        return None
    progress = read_json(path)
    if progress.get("task_id") != task_id:
        raise ValueError("NPU 阶段进度任务归属不符")
    data = progress.get("units", {}).get(unit_id)
    if data is None:
        return None
    if data.get("backend") != REQUIRED_NPU_BACKEND or data.get("performance_gate_issued") is not False:
        raise ValueError("NPU 阶段进度不得混后端或签发性能门禁")
    plan = read_json(ROOT / "upstream" / f"{task_id.lower().replace('-', '')}_reference_plan.yaml")
    expected = {c["case_id"]: c for c in plan["cases"] if c["acceptance_unit_id"] == unit_id}
    cases = data.get("cases", {})
    if not cases or not set(cases) <= set(expected):
        raise ValueError("NPU 阶段进度 case 集合不符")
    for case_id, item in cases.items():
        record = read_json(verified_file(item["result_path"], item["sha256"]))
        if (record.get("backend") != REQUIRED_NPU_BACKEND or record.get("pytorch_commit") != commit
                or record.get("task_id") != task_id or record.get("acceptance_unit_id") != unit_id
                or record.get("case_id") != case_id or record.get("status") != "community-contract-passed"
                or record.get("tests_ran") != 1 or record.get("tests_skipped") != 0
                or record.get("numerical_execution") is not True
                or record.get("body_or_assertions_modified") is not False
                or record.get("product_gate_bypassed") is not False
                or record.get("source_test") != expected[case_id]["source_test"]):
            raise ValueError(f"NPU 功能合同不符：{case_id}")
        if any(r["changed"] for r in record["handler_records"]) != expected[case_id]["expected_match"]:
            raise ValueError(f"NPU 目标正负判据不符：{case_id}")
        verify_progress_artifacts(item["artifacts"])
    complete = set(cases) == set(expected)
    if data.get("contract_complete") is not complete:
        raise ValueError("NPU 阶段完成度与 case 集合不一致")
    return path, {"contract_complete": complete, "updated_at": progress["updated_at"]}


def phase(row: dict) -> str:
    if row["pending_variant_count"]:
        if "regressed-on-device" in row["coverage_status"]:
            return "coverage-extension-npu-regression-open"
        return "coverage-extension-awaiting-gpu-reference"
    if row["comparison_result_path"]:
        return "functional-comparison-closed"
    if row["npu_result_path"]:
        return "npu-result-awaiting-comparison"
    if row["reference_status"].startswith(("pending", "yes-provisional")):
        return "awaiting-gpu-reference"
    return "awaiting-npu"


def community_alignment(
    comparison: dict | None,
    preparation: dict | None = None,
) -> dict[str, str]:
    """读取实测对齐结论；无实测时只显示 manifest 中的准备态边界。"""
    default = {
        "community_alignment_status": "PENDING_REVIEW",
        "community_alignment_source": "legacy-missing-explicit",
        "community_aligned_scope": "",
        "community_divergent_scope": "",
        "community_open_scope": "历史结果缺少显式社区对齐范围，需在后续再认证时补录",
        "community_alignment_disposition": "保留原功能/性能结论，但不得据此外推为完全社区对齐",
    }
    if comparison is None and isinstance(preparation, dict):
        gpu_scope = str(preparation.get("gpu_scope") or "")
        npu_scope = str(preparation.get("npu_scope") or "")
        return {
            "community_alignment_status": str(
                preparation.get("status") or "PENDING_REVIEW"
            ),
            "community_alignment_source": "manifest-preparation-contract",
            "community_aligned_scope": "",
            "community_divergent_scope": "",
            "community_open_scope": "；".join(
                value
                for value in (
                    f"GPU预期：{gpu_scope}" if gpu_scope else "",
                    f"NPU待验证：{npu_scope}" if npu_scope else "",
                )
                if value
            ),
            "community_alignment_disposition": (
                "源码边界已审核、设备行为尚未形成结论；GPU reference 与 NPU "
                "triton_experimental 实测后更新"
            ),
        }
    if not comparison:
        return default
    alignment = comparison.get("community_alignment")
    if not isinstance(alignment, dict):
        return default
    return {
        "community_alignment_status": str(alignment.get("status") or "PENDING_REVIEW"),
        "community_alignment_source": "explicit",
        "community_aligned_scope": "；".join(alignment.get("aligned_scope", [])),
        "community_divergent_scope": "；".join(alignment.get("divergent_scope", [])),
        "community_open_scope": "；".join(alignment.get("open_scope", [])),
        "community_alignment_disposition": str(alignment.get("disposition") or ""),
    }


def build_rows(generated_at: str) -> list[dict]:
    npu_results = index_documents("results/current/*/npu_result.json", "acceptance_unit_id")
    functional_results = index_documents(
        "results/current/T-*/functional/*.json", "acceptance_unit_id"
    )
    overlap = set(npu_results) & set(functional_results)
    if overlap:
        raise ValueError(f"NPU result 与紧凑功能结果重复：{sorted(overlap)}")
    npu_results.update(functional_results)
    comparisons = index_documents(
        "results/current/*/comparison_result.json", "acceptance_unit_id"
    )
    alignment_sidecars = index_documents(
        "results/current/T-*/community_alignment/*.json", "acceptance_unit_id"
    )

    performance: dict[str, tuple[str, Path, dict, dict]] = {}
    for summary_path in sorted(ROOT.glob("results/current/T-*/performance_summary.json")):
        summary = read_json(summary_path)
        task_id = str(summary.get("task_id", ""))
        if task_id not in TASK_FILES:
            raise ValueError(f"未知性能任务：{summary_path}: {task_id}")
        if summary.get("backend") != REQUIRED_NPU_BACKEND:
            raise ValueError(
                f"当前性能结论 backend 必须为 {REQUIRED_NPU_BACKEND}：{summary_path}"
            )
        for item in summary.get("acceptance_units", []):
            unit_id = item.get("acceptance_unit_id")
            if not isinstance(unit_id, str) or not unit_id:
                raise ValueError(f"性能汇总缺少 acceptance_unit_id：{summary_path}")
            if unit_id in performance:
                raise ValueError(f"性能单元重复：{unit_id}")
            performance[unit_id] = (task_id, summary_path, summary, item)

    rows: list[dict] = []
    unit_ids: set[str] = set()
    for task_id, files in TASK_FILES.items():
        manifest_path = ROOT / files["manifest"]
        plan_path = ROOT / files["performance_plan"]
        manifest = read_json(manifest_path)
        plan = read_json(plan_path)
        if plan.get("task_id") != task_id:
            raise ValueError(f"性能计划 task_id 不匹配：{plan_path}")

        reference_contract = manifest.get("reference_contract", {})
        if reference_contract.get("backend") != REFERENCE_BACKEND:
            raise ValueError(
                f"GPU reference backend 必须为 {REFERENCE_BACKEND}：{manifest_path}"
            )

        plan_items = {}
        for item in plan.get("acceptance_units", []):
            unit_id = item.get("acceptance_unit_id")
            if unit_id in plan_items:
                raise ValueError(f"性能计划单元重复：{unit_id}: {plan_path}")
            plan_items[unit_id] = item
            backend = performance_backend(plan, item)
            if backend != REQUIRED_NPU_BACKEND:
                raise ValueError(
                    f"性能计划 backend 必须为 {REQUIRED_NPU_BACKEND}："
                    f"{plan_path}: {unit_id}: {backend or '<empty>'}"
                )

        manifest_units = manifest.get("acceptance_units", [])
        manifest_unit_ids = {item.get("acceptance_unit_id") for item in manifest_units}
        if set(plan_items) != manifest_unit_ids:
            raise ValueError(f"manifest 与性能计划单元集合不一致：{task_id}")

        for unit in manifest_units:
            unit_id = unit.get("acceptance_unit_id")
            if not isinstance(unit_id, str) or not unit_id:
                raise ValueError(f"manifest 缺少 acceptance_unit_id：{manifest_path}")
            if unit_id in unit_ids:
                raise ValueError(f"跨任务 acceptance_unit_id 重复：{unit_id}")
            unit_ids.add(unit_id)

            npu_path, npu = npu_results.get(unit_id, (None, None))
            comparison_path, comparison = comparisons.get(unit_id, (None, None))
            compact_functional = (
                npu is not None and "numerical_execution" in npu
            )
            if comparison is None and compact_functional:
                # T-081～T-083 的紧凑原件同时绑定GPU复核、NPU数值、目标改图
                # 和性能门禁；在独立comparison文件补齐前可作为等价合同结论。
                comparison_path = npu_path
            if comparison is not None and npu is None:
                raise ValueError(f"comparison 缺少对应 NPU result：{unit_id}")

            observed_backend = ""
            npu_execution_status = "not-run"
            if npu is not None:
                observed_backend = str(
                    npu.get("environment", {}).get("backend")
                    or npu.get("backend", "")
                )
                if observed_backend != REQUIRED_NPU_BACKEND:
                    raise ValueError(
                        f"当前 NPU result backend 必须为 {REQUIRED_NPU_BACKEND}："
                        f"{npu_path}: {observed_backend or '<empty>'}"
                    )
                npu_execution_status = str(
                    npu.get("selected_execution", {}).get("status")
                    or ("passed" if compact_functional else "unknown")
                )

            correctness = "not-run"
            comparison_verdict = "not-run"
            repair_status = "not-run"
            if comparison is not None:
                correctness = str(
                    comparison.get("npu", {}).get("correctness_status") or "unknown"
                )
                comparison_verdict = str(comparison.get("final_verdict") or "unknown")
                repair_status = str(comparison.get("repair_status") or "unknown")
            elif compact_functional:
                correctness = str(npu.get("correctness") or "unknown")
                comparison_verdict = str(npu.get("comparison_verdict") or "BEHAVIOR_UNCHANGED")
                repair_status = str(npu.get("repair_status") or "not-needed")

            alignment_payload = comparison if not compact_functional else npu
            alignment_sidecar_path, alignment_sidecar = alignment_sidecars.get(
                unit_id, (None, None)
            )
            if alignment_sidecar is not None:
                if alignment_sidecar.get("task_id") != task_id:
                    raise ValueError(
                        f"社区对齐 sidecar 任务归属错误：{alignment_sidecar_path}"
                    )
                source = alignment_sidecar.get("source_functional", {})
                source_path = (
                    alignment_sidecar_path.parent / str(source.get("path", ""))
                ).resolve()
                if npu_path is None or source_path != npu_path.resolve():
                    raise ValueError(
                        f"社区对齐 sidecar 未绑定当前功能原件：{alignment_sidecar_path}"
                    )
                if not source_path.is_file() or sha256(source_path) != source.get("sha256"):
                    raise ValueError(
                        f"社区对齐 sidecar 功能原件哈希失效：{alignment_sidecar_path}"
                    )
                alignment_payload = alignment_sidecar
            alignment = community_alignment(
                alignment_payload,
                unit.get("community_alignment"),
            )

            plan_item = plan_items[unit_id]
            performance_status = str(plan_item.get("performance_status") or "unknown")
            performance_verdict = str(plan_item.get("verdict") or "planned")
            performance_path = plan_path
            source_times = [str(manifest.get("generated_at") or "")]
            if npu is not None:
                source_times.append(str(npu.get("generated_at") or ""))
            if comparison is not None:
                source_times.append(str(comparison.get("generated_at") or ""))
            if alignment_sidecar is not None:
                source_times.append(str(alignment_sidecar.get("generated_at") or ""))

            actual_performance = performance.get(unit_id)
            if actual_performance is not None:
                perf_task, performance_path, summary, perf_item = actual_performance
                if perf_task != task_id:
                    raise ValueError(f"性能汇总任务归属错误：{unit_id}")
                performance_status = str(
                    perf_item.get("performance_status") or "unknown"
                )
                performance_verdict = str(perf_item.get("verdict") or "unknown")
                source_times.append(str(summary.get("generated_at") or ""))

            if unit.get("pending_variants"):
                if correctness not in {"not-run", "unknown"}:
                    correctness += "-for-existing-variants"
                if comparison_verdict not in {"not-run", "unknown"}:
                    comparison_verdict += "-for-existing-variants"
                if repair_status not in {"not-run", "unknown"}:
                    repair_status += "-for-existing-variants"
                if performance_status not in {"not-run", "unknown"}:
                    performance_status += "-verified-variants-only"
                if performance_verdict not in {"planned", "unknown"}:
                    performance_verdict += "-for-existing-variants"

            row = {
                "matrix_generated_at": generated_at,
                "task_id": task_id,
                "acceptance_unit_id": unit_id,
                "canonical_acceptance_unit_id": unit.get("canonical_acceptance_unit_id", unit_id),
                "independent_unit_contribution": unit.get("independent_unit_contribution", 1),
                "gpu_review_path": "",
                "npu_progress_path": "",
                "contract_name": str(unit.get("contract_name") or ""),
                "stage": str(unit.get("stage") or ""),
                "manifest_status": str(manifest.get("status") or ""),
                "review_status": str(unit.get("review_status") or ""),
                "denominator_eligible": str(unit.get("denominator_eligible") or ""),
                "variant_count": len(unit.get("variants", []))
                + len(unit.get("pending_variants", [])),
                "verified_variant_count": len(unit.get("variants", [])),
                "pending_variant_count": len(unit.get("pending_variants", [])),
                "coverage_status": (
                    "verified="
                    + str(len(unit.get("variants", [])))
                    + "; pending="
                    + ",".join(
                        item["variant_id"]
                        + "[reference="
                        + str(item.get("reference_status", "unknown"))
                        + ";npu="
                        + str(item.get("npu_status", "unknown"))
                        + "]"
                        for item in unit.get("pending_variants", [])
                    )
                    if unit.get("pending_variants")
                    else "fully-covered"
                ),
                "community_test_count": len(unit.get("community_tests", [])),
                "reference_backend": REFERENCE_BACKEND,
                "reference_status": reference_status(unit, reference_contract),
                "required_npu_backend": REQUIRED_NPU_BACKEND,
                "observed_npu_backend": observed_backend,
                "npu_execution_status": npu_execution_status,
                "npu_correctness_status": correctness,
                "comparison_verdict": comparison_verdict,
                "repair_status": repair_status,
                **alignment,
                "performance_status": performance_status,
                "performance_verdict": performance_verdict,
                "current_phase": "",
                "updated_at": max(source_times),
                "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
                "npu_result_path": (
                    npu_path.relative_to(ROOT).as_posix() if npu_path else ""
                ),
                "comparison_result_path": (
                    comparison_path.relative_to(ROOT).as_posix()
                    if comparison_path
                    else ""
                ),
                "performance_evidence_path": performance_path.relative_to(ROOT).as_posix(),
            }
            row["current_phase"] = str(unit.get("coverage_phase") or phase(row))
            if row["independent_unit_contribution"] == 0:
                row["performance_status"] = "duplicate-contract-no-new-measurement"
                row["performance_verdict"] = "see-canonical-task-not-recertified-here"
            review_path = ROOT / "results/current" / task_id / "gpu_reference_review.json"
            if task_id >= "T-087" and review_path.is_file():
                review = read_json(review_path)
                source = (ROOT / review["input"]).resolve()
                if (not source.is_relative_to(ROOT.resolve()) or not source.is_file()
                        or sha256(source) != review["input_sha256"]
                        or review.get("reference_backend") != "inductor-default"
                        or unit_id not in review.get("acceptance_units", [])):
                    raise ValueError(f"GPU复核与原件不一致：{review_path}")
                row["gpu_review_path"] = review_path.relative_to(ROOT).as_posix()
                row["reference_status"] = (
                    "valid-reference-frozen" if row["denominator_eligible"] == "yes-frozen"
                    and review["status"] == "gpu-contract-reviewed-awaiting-npu" else review["status"]
                )
                if not npu and row["independent_unit_contribution"]:
                    row["current_phase"] = (
                        "gpu-target-attribution-pending"
                        if review["status"] == "native-passed-target-attribution-pending"
                        else "awaiting-npu"
                    )
            progress = npu_contract_progress(task_id, unit_id) if not npu else None
            if progress and row["independent_unit_contribution"]:
                progress_path, progress_data = progress
                row["npu_progress_path"] = progress_path.relative_to(ROOT).as_posix()
                row["observed_npu_backend"] = REQUIRED_NPU_BACKEND
                row["npu_execution_status"] = "community-contract-passed-not-final"
                row["npu_correctness_status"] = "passed-for-recorded-cases"
                row["current_phase"] = (
                    "npu-contract-passed-awaiting-performance-gate"
                    if progress_data["contract_complete"] else "npu-contract-partial"
                )
                row["community_alignment_disposition"] = "本轮社区功能合同通过；fallback/graph-break及性能图门禁待复核，未签发正式比较结论"
                row["updated_at"] = max(row["updated_at"], progress_data["updated_at"])
            blocker_path = ROOT / "results/current" / task_id / "npu_blocker_review.json"
            if not npu and blocker_path.is_file():
                blocked = read_json(blocker_path)
                if blocked.get("acceptance_unit_id") == unit_id:
                    evidence = blocked["result"]
                    raw = read_json(verified_file(evidence["path"], evidence["sha256"]))
                    if (blocked.get("backend") != REQUIRED_NPU_BACKEND or raw.get("backend") != REQUIRED_NPU_BACKEND
                            or raw.get("pytorch_commit") != "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
                            or raw.get("status") != "failed-contract"):
                        raise ValueError("NPU 失败记录的实际后端/来源/状态不符")
                    row.update(npu_progress_path=str(blocker_path.relative_to(ROOT)),
                        observed_npu_backend=REQUIRED_NPU_BACKEND, npu_execution_status="failed",
                        npu_correctness_status=blocked["correctness"], current_phase=blocked["current_phase"],
                        repair_status=blocked["repair_status"], community_alignment_status="PARTIALLY_ALIGNED",
                        community_alignment_source="explicit", community_divergent_scope=blocked["reason"],
                        community_alignment_disposition=blocked["next_action"],
                        updated_at=max(row["updated_at"], blocked["generated_at"]))
            rows.append(row)

    for row in rows:
        canonical = row["canonical_acceptance_unit_id"]
        contribution = row["independent_unit_contribution"]
        if type(contribution) is not int or contribution not in (0, 1):
            raise ValueError("独立单元贡献必须为0/1")
        if canonical not in unit_ids or (contribution == 0) != (canonical != row["acceptance_unit_id"]):
            raise ValueError("别名必须指向实际不同的canonical单元，且贡献为0")
    unknown_npu = set(npu_results) - unit_ids
    unknown_comparisons = set(comparisons) - unit_ids
    unknown_performance = set(performance) - unit_ids
    unknown_alignments = set(alignment_sidecars) - unit_ids
    if unknown_npu or unknown_comparisons or unknown_performance or unknown_alignments:
        raise ValueError(
            "current result 存在未纳入 manifest 的单元："
            f"npu={sorted(unknown_npu)}, comparison={sorted(unknown_comparisons)}, "
            f"performance={sorted(unknown_performance)}, "
            f"alignment={sorted(unknown_alignments)}"
        )
    return rows


def render_csv(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def md(value: object) -> str:
    text = str(value) if value not in (None, "") else "—"
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(rows: list[dict], generated_at: str) -> str:
    independent = sum(row["independent_unit_contribution"] for row in rows)
    frozen = sum(row["denominator_eligible"] == "yes-frozen" for row in rows)
    pending = sum(bool(row["pending_variant_count"]) for row in rows)
    compared = sum(bool(row["comparison_result_path"]) for row in rows)
    measured_or_disposed = sum(
        row["performance_evidence_path"].startswith("results/current/") for row in rows
    )
    observed = sorted(
        {row["observed_npu_backend"] for row in rows if row["observed_npu_backend"]}
    )
    lines = [
        "# 当前 Acceptance Unit 兼容性矩阵",
        "",
        f"> 生成时间：{generated_at}",
        "> 数据源：`upstream/*manifest.yaml`、`results/current/` 与逐任务性能计划/汇总。",
        "> 后端边界：GPU reference 固定为 `inductor-default`；NPU 动态验证、比较、修复验证与性能固定为 `triton_experimental`。",
        "> 历史 251 行 registration 矩阵不参与本表 verdict；其用途与边界见 `report/archive/legacy-20260820-0828/pass_src_20260820/README.md`。",
        "",
        "## 状态摘要",
        "",
        f"- 跟踪记录：**{len(rows)}**；去重后独立 acceptance units：**{independent}**；已冻结 reference：**{frozen}**；存在覆盖扩展未闭环：**{pending}**。",
        "- T-112 是 T-084 的同合同补证，独立分母贡献为 0，保留记录但不重复计数。",
        f"- 已形成 NPU/comparison：**{compared}**；已有正式性能处置：**{measured_or_disposed}**；其余为性能计划态。",
        "- `comparison`/性能处置数量只说明已登记 variants；存在 pending extension 的单元必须以“覆盖”和“当前阶段”列为准，不能外推为全域闭环。",
        f"- 当前 NPU 结果实际观测 backend：`{', '.join(observed) if observed else '无'}`。",
        "- 本表汇总已登记结论，不代表严格历史再认证通过；T-076/T-077 的独立补证状态见 [最新审计](../results/audits/latest.json)。",
        "- `npu_execution_status=failed` 不自动表示数值错误；例如产品 gate 关闭时，目标命中失败可与原图 correctness 通过同时成立，应结合 comparison verdict 阅读。",
        "- `npu_progress_path` 是带原件哈希的社区功能阶段进度，不是正式 comparison 或性能 gate；负例要求不改写，不能强制所有 case 改图。",
        "- `native-passed-target-attribution-pending` 表示原生测试通过但精确目标仍待归因；计划 expected_assertions 不能代替真实断言。别名行保留历史、独立贡献为 0。",
        "- 社区对齐列必须区分完全对齐、部分对齐、预期后端差异、需修复和待复核；旧结果缺少显式字段时一律显示待复核，不从既有 PASS 自动外推。",
        "",
        "## 单元矩阵",
        "",
        "| T | Acceptance unit | Stage | 覆盖 | Reference | NPU backend | NPU 执行 | Correctness | Comparison | Repair | 社区对齐/处置 | 性能处置 | 当前阶段 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        performance = f"{row['performance_status']} / {row['performance_verdict']}"
        lines.append(
            "| "
            + " | ".join(
                md(value)
                for value in (
                    row["task_id"],
                    row["acceptance_unit_id"],
                    row["stage"],
                    row["coverage_status"],
                    row["reference_status"],
                    row["observed_npu_backend"] or f"待测（要求 {REQUIRED_NPU_BACKEND}）",
                    row["npu_execution_status"],
                    row["npu_correctness_status"],
                    row["comparison_verdict"],
                    row["repair_status"],
                    row["community_alignment_status"]
                    + " / "
                    + row["community_alignment_disposition"],
                    performance,
                    row["current_phase"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 使用说明",
            "",
            "- 本 Markdown 便于阅读；完整字段、证据路径和生成时间以同目录 CSV 为准。",
            "- `reference_backend=inductor-default` 表示 CUDA/GPU 对照端，不能据此声称 NPU 使用了 default backend。",
            "- 只有 `observed_npu_backend=triton_experimental` 的动态结果可进入当前 NPU comparison。空值表示尚未运行，不表示可使用其他 backend。",
            "- 性能证据路径指向 `results/current/` 时表示已有处置；指向 `upstream/*_performance_plan.yaml` 时只表示测量合同已准备。",
            "- 社区对齐的完整已对齐/差异/未决范围保存在 CSV；`source=legacy-missing-explicit` 表示旧结果仍需显式再认证。",
            "- 修改 manifest/result 后运行 `python scripts/generate_current_acceptance_matrix.py --write` 更新，再运行 `--check` 做一致性校验。",
            "",
        ]
    )
    return "\n".join(lines)


def committed_timestamp() -> str:
    if not OUTPUT_CSV.is_file():
        raise ValueError(f"缺少已生成矩阵：{OUTPUT_CSV}")
    with OUTPUT_CSV.open(encoding="utf-8", newline="") as stream:
        first = next(csv.DictReader(stream), None)
    if not first or not first.get("matrix_generated_at"):
        raise ValueError("当前矩阵缺少 matrix_generated_at")
    return first["matrix_generated_at"]


def write_outputs(generated_at: str) -> None:
    rows = build_rows(generated_at)
    OUTPUT_CSV.write_text(render_csv(rows), encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(rows, generated_at), encoding="utf-8")
    print(f"current_acceptance_matrix=written units={len(rows)} generated_at={generated_at}")


def check_outputs() -> None:
    generated_at = committed_timestamp()
    rows = build_rows(generated_at)
    expected = {
        OUTPUT_CSV: render_csv(rows),
        OUTPUT_MD: render_markdown(rows, generated_at),
    }
    stale = [path for path, content in expected.items() if path.read_text(encoding="utf-8") != content]
    if stale:
        raise ValueError("当前矩阵未同步，请重新 --write：" + ", ".join(map(str, stale)))
    print(
        "current_acceptance_matrix=OK "
        f"units={len(rows)} compared={sum(bool(row['comparison_result_path']) for row in rows)} "
        f"backend={REQUIRED_NPU_BACKEND}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="生成并写入当前矩阵")
    action.add_argument("--check", action="store_true", help="检查已提交矩阵是否与数据源一致")
    parser.add_argument("--generated-at", help="--write 时覆盖生成时间（ISO 8601）")
    args = parser.parse_args()
    try:
        if args.write:
            generated_at = args.generated_at or datetime.now().astimezone().isoformat(
                timespec="seconds"
            )
            write_outputs(generated_at)
        else:
            if args.generated_at:
                parser.error("--generated-at 只能与 --write 一起使用")
            check_outputs()
    except ValueError as exc:
        print(f"current_acceptance_matrix=ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
