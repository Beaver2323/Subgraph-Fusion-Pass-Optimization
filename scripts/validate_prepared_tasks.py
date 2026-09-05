#!/usr/bin/env python3
"""静态校验 T-078～T-080 的 reference、性能计划和中文讲解。

该脚本只使用 Python 标准库，不导入 torch，也不访问设备。
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime
from pathlib import Path


TASKS = ("T-078", "T-079", "T-080")


def load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON-compatible YAML：{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"顶层必须是 object：{path}")
    return value


def require_nonempty(value: object, label: str) -> None:
    if value is None or (isinstance(value, str) and not value.strip()) or value == [] or value == {}:
        raise ValueError(f"缺少或为空：{label}")


def unique_records(records: object, key: str, label: str) -> dict:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} 必须是非空列表")
    result = {}
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get(key), str) or not item[key].strip():
            raise ValueError(f"{label} 缺少有效 {key}")
        if item[key] in result:
            raise ValueError(f"{label} 重复 {key}: {item[key]}")
        result[item[key]] = item
    return result


def validate_implementation(repo_root: Path, performance: dict) -> str:
    implementation = performance.get("implementation", {})
    status = implementation.get("status")
    entrypoint = implementation.get("entrypoint")
    if status == "not-implemented":
        if entrypoint is not None:
            raise ValueError("not-implemented 不得声明 worker entrypoint")
        return "plan-only"
    if status != "implemented-awaiting-runtime-validation":
        raise ValueError("implementation.status 必须如实声明未实现或待动态验证")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise ValueError("已实现 worker 必须声明 entrypoint")
    path = (repo_root / entrypoint).resolve()
    if not path.is_relative_to(repo_root.resolve()) or not path.is_file():
        raise ValueError("worker entrypoint 必须是仓库内实际文件")
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return "worker-static-only"


def validate_task(repo_root: Path, task_id: str) -> tuple[int, int, int]:
    suffix = task_id.lower().replace("-", "")
    manifest_path = repo_root / "upstream" / f"{suffix}_manifest.yaml"
    reference_path = repo_root / "upstream" / f"{suffix}_reference_plan.yaml"
    performance_path = repo_root / "upstream" / f"{suffix}_performance_plan.yaml"

    manifest = load_json(manifest_path)
    reference = load_json(reference_path)
    performance = load_json(performance_path)

    if reference.get("task_id") != task_id or performance.get("task_id") != task_id:
        raise ValueError(f"{task_id}：reference/performance task_id 不一致")
    validate_implementation(repo_root, performance)
    for name, data in (("manifest", manifest), ("reference", reference), ("performance", performance)):
        require_nonempty(data.get("generated_at"), f"{task_id}.{name}.generated_at")
        if datetime.fromisoformat(data["generated_at"]).utcoffset() is None:
            raise ValueError(f"{task_id}.{name}.generated_at 必须包含时区")

    declared_perf = reference.get("performance_plan", {}).get("path")
    declared_guide = reference.get("case_guide", {}).get("path")
    if declared_perf != performance_path.relative_to(repo_root).as_posix():
        raise ValueError(f"{task_id}：reference 未指向对应 performance plan")
    if not declared_guide:
        raise ValueError(f"{task_id}：reference 未声明中文 case guide")
    guide_path = repo_root / declared_guide
    guide_text = guide_path.read_text(encoding="utf-8")
    for marker in ("更新时间", "功能测例", "性能测例", "triton_experimental"):
        if marker not in guide_text:
            raise ValueError(f"{task_id}：{declared_guide} 缺少标记 {marker!r}")

    backend = performance.get("backend_contract", {})
    if backend.get("npu_backend") != "triton_experimental":
        raise ValueError(f"{task_id}：NPU 性能 backend 必须为 triton_experimental")
    if backend.get("process_isolation") != "fresh-process-per-arm":
        raise ValueError(f"{task_id}：OFF/ON 必须使用 fresh process")
    order = backend.get("off_on_order")
    if order != ["OFF1", "ON1", "ON2", "OFF2", "OFF3", "ON3"]:
        raise ValueError(f"{task_id}：OFF/ON 交错顺序不符合冻结合同")

    measurement = performance.get("measurement_contract", {})
    for key in ("correctness_gate", "warmup", "runs", "timing", "memory", "verdict_thresholds"):
        require_nonempty(measurement.get(key), f"{task_id}.measurement_contract.{key}")
    for key in ("warmup", "runs"):
        value = measurement[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{task_id}.measurement_contract.{key} 必须为正整数")

    manifest_by_id = unique_records(manifest.get("acceptance_units"), "acceptance_unit_id", f"{task_id}.manifest")
    performance_by_id = unique_records(performance.get("acceptance_units"), "acceptance_unit_id", f"{task_id}.performance")
    manifest_units = set(manifest_by_id)
    performance_units = set(performance_by_id)
    if manifest_units != performance_units:
        raise ValueError(
            f"{task_id}：manifest/performance units 不一致："
            f"manifest={sorted(manifest_units)} performance={sorted(performance_units)}"
        )
    for unit_id in manifest_units:
        if unit_id not in guide_text:
            raise ValueError(f"{task_id}：中文 guide 未逐单元讲解 {unit_id}")

    for unit in performance.get("acceptance_units", []):
        unit_id = unit["acceptance_unit_id"]
        manifest_nodeids = {
            test["nodeid"] for test in manifest_by_id[unit_id].get("community_tests", [])
        }
        for key in (
            "performance_status",
            "case_source",
            "functional_gate",
            "workloads",
            "off_on_control",
            "artifacts",
            "negative_cases",
        ):
            require_nonempty(unit.get(key), f"{task_id}.{unit_id}.{key}")
        source = unit["case_source"]
        if source.get("kind") not in {
            "community-benchmark-reuse",
            "tracker-derived-from-community-functional-case",
        }:
            raise ValueError(f"{task_id}.{unit_id}：未知 case_source.kind")
        nodeids = source.get("nodeids")
        if not isinstance(nodeids, list) or not nodeids or any(not isinstance(item, str) for item in nodeids):
            raise ValueError(f"{task_id}.{unit_id}：性能来源 nodeids 必须是非空字符串列表")
        if len(nodeids) != len(set(nodeids)):
            raise ValueError(f"{task_id}.{unit_id}：性能来源 nodeids 重复")
        for nodeid in nodeids:
            if nodeid not in manifest_nodeids:
                raise ValueError(f"{task_id}.{unit_id}：性能来源未登记于 manifest：{nodeid}")
        workloads = unique_records(unit["workloads"], "workload_id", f"{task_id}.{unit_id}.workloads")
        for workload in workloads.values():
            for key in ("workload_id", "role", "shape_contract", "measurement_scope"):
                require_nonempty(workload.get(key), f"{task_id}.{unit_id}.workload.{key}")

    case_count = len(reference.get("cases", []))
    variant_count = sum(
        len(case.get("variant_ids", [])) for case in reference.get("cases", [])
    )
    return len(manifest_units), case_count, variant_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args()
    repo_root = (args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    tasks = args.task or list(TASKS)
    for task_id in tasks:
        units, cases, variants = validate_task(repo_root, task_id)
        plan = load_json(repo_root / "upstream" / f"{task_id.lower().replace('-', '')}_performance_plan.yaml")
        readiness = validate_implementation(repo_root, plan)
        print(
            "prepared_task_validation=OK "
            f"task={task_id} units={units} cases={cases} variants={variants} "
            f"performance_units={units} guide=valid performance_readiness={readiness} "
            f"worker={plan['implementation']['status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
