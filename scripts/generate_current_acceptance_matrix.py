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
}

FIELDNAMES = [
    "matrix_generated_at",
    "task_id",
    "acceptance_unit_id",
    "contract_name",
    "stage",
    "manifest_status",
    "review_status",
    "denominator_eligible",
    "variant_count",
    "community_test_count",
    "reference_backend",
    "reference_status",
    "required_npu_backend",
    "observed_npu_backend",
    "npu_execution_status",
    "npu_correctness_status",
    "comparison_verdict",
    "repair_status",
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
        return str(reference_contract.get("suite_status") or "frozen-reference-valid")
    return eligible or "unknown"


def phase(row: dict) -> str:
    if row["comparison_result_path"]:
        return "functional-comparison-closed"
    if row["npu_result_path"]:
        return "npu-result-awaiting-comparison"
    if row["reference_status"].startswith("pending"):
        return "awaiting-gpu-reference"
    return "awaiting-npu"


def build_rows(generated_at: str) -> list[dict]:
    npu_results = index_documents("results/current/*/npu_result.json", "acceptance_unit_id")
    comparisons = index_documents(
        "results/current/*/comparison_result.json", "acceptance_unit_id"
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
            if comparison is not None and npu is None:
                raise ValueError(f"comparison 缺少对应 NPU result：{unit_id}")

            observed_backend = ""
            npu_execution_status = "not-run"
            if npu is not None:
                observed_backend = str(npu.get("environment", {}).get("backend", ""))
                if observed_backend != REQUIRED_NPU_BACKEND:
                    raise ValueError(
                        f"当前 NPU result backend 必须为 {REQUIRED_NPU_BACKEND}："
                        f"{npu_path}: {observed_backend or '<empty>'}"
                    )
                npu_execution_status = str(
                    npu.get("selected_execution", {}).get("status") or "unknown"
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

            plan_item = plan_items[unit_id]
            performance_status = str(plan_item.get("performance_status") or "unknown")
            performance_verdict = str(plan_item.get("verdict") or "planned")
            performance_path = plan_path
            source_times = [str(manifest.get("generated_at") or "")]
            if npu is not None:
                source_times.append(str(npu.get("generated_at") or ""))
            if comparison is not None:
                source_times.append(str(comparison.get("generated_at") or ""))

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

            row = {
                "matrix_generated_at": generated_at,
                "task_id": task_id,
                "acceptance_unit_id": unit_id,
                "contract_name": str(unit.get("contract_name") or ""),
                "stage": str(unit.get("stage") or ""),
                "manifest_status": str(manifest.get("status") or ""),
                "review_status": str(unit.get("review_status") or ""),
                "denominator_eligible": str(unit.get("denominator_eligible") or ""),
                "variant_count": len(unit.get("variants", [])),
                "community_test_count": len(unit.get("community_tests", [])),
                "reference_backend": REFERENCE_BACKEND,
                "reference_status": reference_status(unit, reference_contract),
                "required_npu_backend": REQUIRED_NPU_BACKEND,
                "observed_npu_backend": observed_backend,
                "npu_execution_status": npu_execution_status,
                "npu_correctness_status": correctness,
                "comparison_verdict": comparison_verdict,
                "repair_status": repair_status,
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
            row["current_phase"] = phase(row)
            rows.append(row)

    unknown_npu = set(npu_results) - unit_ids
    unknown_comparisons = set(comparisons) - unit_ids
    unknown_performance = set(performance) - unit_ids
    if unknown_npu or unknown_comparisons or unknown_performance:
        raise ValueError(
            "current result 存在未纳入 manifest 的单元："
            f"npu={sorted(unknown_npu)}, comparison={sorted(unknown_comparisons)}, "
            f"performance={sorted(unknown_performance)}"
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
    frozen = sum(row["denominator_eligible"] == "yes-frozen" for row in rows)
    pending = sum(row["reference_status"].startswith("pending") for row in rows)
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
        "> 历史 251 行 registration 矩阵不参与本表 verdict；其用途与边界见 `report/pass_src_20260820/README.md`。",
        "",
        "## 状态摘要",
        "",
        f"- 活动 acceptance units：**{len(rows)}**；已冻结 reference：**{frozen}**；等待 GPU reference：**{pending}**。",
        f"- 已形成 NPU/comparison：**{compared}**；已有正式性能处置：**{measured_or_disposed}**；其余为性能计划态。",
        f"- 当前 NPU 结果实际观测 backend：`{', '.join(observed) if observed else '无'}`。",
        "- `npu_execution_status=failed` 不自动表示数值错误；例如产品 gate 关闭时，目标命中失败可与原图 correctness 通过同时成立，应结合 comparison verdict 阅读。",
        "",
        "## 单元矩阵",
        "",
        "| T | Acceptance unit | Stage | Reference | NPU backend | NPU 执行 | Correctness | Comparison | Repair | 性能处置 | 当前阶段 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
                    row["reference_status"],
                    row["observed_npu_backend"] or f"待测（要求 {REQUIRED_NPU_BACKEND}）",
                    row["npu_execution_status"],
                    row["npu_correctness_status"],
                    row["comparison_verdict"],
                    row["repair_status"],
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
