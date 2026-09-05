#!/usr/bin/env python3
"""从冻结 inventory 与活动 manifest 生成草案批次；不生成或冻结 GPU 测例。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


ALIASES = {"AU-post-grad-move-constructors-to-cuda": "AU-post-grad-move-constructors-to-gpu"}
FAMILY_ORDER = (
    "joint_graph", "post_grad", "split_cat", "pre_grad", "misc_patterns",
    "replace_random", "efficient_conv_bn_eval", "freezing_patterns", "binary_folding",
    "reduced_atomic_contention", "fuse_attention", "quantization", "mkldnn_fusion", "fsdp",
)


def natural_key(value: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def build(repo_root: Path, generated_at: str) -> dict:
    if datetime.fromisoformat(generated_at).utcoffset() is None:
        raise ValueError("时间戳必须包含时区")
    inventory = repo_root / "report/upstream_pass_test_index_20260829/acceptance_units.csv"
    with inventory.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest_paths = [repo_root / "upstream/manifest.yaml", *sorted((repo_root / "upstream").glob("t*_manifest.yaml"))]
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    selected = [unit["acceptance_unit_id"] for data in manifests for unit in data["acceptance_units"]]
    if len(selected) != len(set(selected)):
        raise ValueError("活动 manifest 有重复单元")
    existing_path = repo_root / "upstream/task_backlog.json"
    if existing_path.exists():
        previous = json.loads(existing_path.read_text(encoding="utf-8"))
        if previous.get("planning_manifest_units", sorted(selected)) != sorted(selected):
            raise ValueError("manifest 单元集合已变化：请显式审核批次迁移并保留既有 T 编号，禁止自动重编号")
    inventory_ids = [row["acceptance_unit"] for row in rows]
    if len(inventory_ids) != len(set(inventory_ids)):
        raise ValueError("inventory 有重复单元")
    mapped = {ALIASES.get(unit_id, unit_id) for unit_id in inventory_ids}
    if set(selected) - mapped:
        raise ValueError(f"新 manifest 单元尚未建立 inventory 映射：{sorted(set(selected) - mapped)}")
    remaining = [row for row in rows if ALIASES.get(row["acceptance_unit"], row["acceptance_unit"]) not in selected]
    eligible = [row for row in remaining if row["denominator_eligible"] == "yes-provisional"]
    controls = [row for row in remaining if row["denominator_eligible"] != "yes-provisional"]
    groups = {}
    for row in eligible:
        family = Path(row["source"].split(";")[0]).stem
        groups.setdefault(family, []).append(row)
    families = sorted(groups, key=lambda family: (FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER), family))
    batches = []
    for family in families:
        group = sorted(groups[family], key=lambda row: natural_key(row["acceptance_unit"]))
        for start in range(0, len(group), 5):
            chunk = group[start:start + 5]
            batches.append({
                "task_id": f"T-{81 + len(batches):03d}",
                "family": family,
                "status": "draft-awaiting-contract-review",
                "reference_ready": False,
                "performance_readiness": "needs-community-benchmark-search-and-worker",
                "units": [{
                    "provisional_unit_id": row["acceptance_unit"],
                    "source_paths": row["source"].split(";"),
                    "community_test_candidates": list(filter(None, row["upstream_tests"].split(";"))),
                    "coverage_hint": row["test_coverage"],
                } for row in chunk],
            })
    return {
        "schema_version": "1.0", "generated_at": generated_at,
        "scope": "T-074 FX inventory 的剩余 provisional 单元；不是全 Inductor 分母",
        "source_inventory": {"path": inventory.relative_to(repo_root).as_posix(), "sha256": hashlib.sha256(inventory.read_bytes()).hexdigest()},
        "aliases": ALIASES,
        "planning_manifest_units": sorted(selected),
        "counts": {"inventory_units": len(rows), "selected_manifest_units": len(selected), "remaining_provisional_units": len(eligible), "non_counting_review_records": len(controls), "draft_batches": len(batches)},
        "batch_acceptance": [
            "逐单元审核 contract、正负例和真实测试入口；允许合并/拆分，但保留旧 ID 映射",
            "性能先检索社区 benchmark；没有则记录功能例派生理由和精确输入/输出/梯度合同",
            "准备 manifest、reference plan、功能/性能讲解、目标级 OFF/ON worker 与零设备回归",
            "先 GPU 原生 reference；NPU triton_experimental 原生阻断后才评审最小适配",
            "显式产品关闭只保留关闭证据/已有测量并免测；generic guard 进入能力评估，不能伪造 ON",
            "功能/命中/正确性通过再做独立进程 OFF/ON 性能；修复及性能归属本批 T",
        ],
        "batches": batches,
        "non_counting_review": [{"provisional_unit_id": row["acceptance_unit"], "source_paths": row["source"].split(";"), "reason": row["unit_role"]} for row in controls],
        "uncovered_independent_inventory": [
            "torch/_inductor/lowering.py 的 ATen→IR registrations；先界定优化合同，不按算子数计数",
            "torch/_inductor/kernel/ 与 select_algorithm.py 的 template/choice 注册；不得与 FX 合同重复计数",
        ],
    }


def markdown(data: dict) -> str:
    counts = data["counts"]
    lines = ["# 后续批次与覆盖边界", "", f"> 更新时间：{data['generated_at']}", "",
        "机器清单见 `upstream/task_backlog.json`；本表由 `scripts/build_task_backlog.py` 生成。", "",
        f"T-074 的 {counts['inventory_units']} 个 provisional 单元中，活动 manifest 已接入 {counts['selected_manifest_units']} 个；",
        f"剩余 {counts['remaining_provisional_units']} 个 provisional eligible 单元暂分 {counts['draft_batches']} 批，另有 {counts['non_counting_review_records']} 条非计数结构记录待审。", "",
        "这些 T 是待审核草案，**不是 GPU-ready**，不进入冻结分母。仅修正 constructor mover 的 cuda→gpu 名称映射，不重写 T-074 原始证据。", "",
        "| 草案任务 | 源码 family | 暂列单元数 | 状态 |", "| --- | --- | ---: | --- |"]
    for batch in data["batches"]:
        lines.append(f"| {batch['task_id']} | `{batch['family']}` | {len(batch['units'])} | 功能映射、性能来源与 worker 待准备 |")
    lines += ["", "## 每批准备与验收标准", ""]
    lines += [f"- {item}。" for item in data["batch_acceptance"]]
    lines += ["", "## 当前覆盖边界", "",
        "| 注册/执行层 | 当前证据 | 后续要求 |", "| --- | --- | --- |",
        "| FX register_graph_pattern / register_replacement | T-074 全部候选来自 fx_passes；部分已人工复核 | 按优化合同审核，不能按装饰器数量计数 |",
        "| PatternMatcherPass / pass_dict | 部分容器/调度项在 inventory | 关联具体优化，结构记录不独立冒充性能单元 |",
        "| register_lowering / ATen→IR | 已有 MM 回归等下游链路证据 | 独立注册清单尚缺；不可声称完整覆盖 |",
        "| template / choice / autotune | 已有 B2B/GEMM 局部选模证据 | 独立候选注册清单尚缺；需与已有 FX 合同去重 |", "",
        "`inductor-extension` 是旧 inventory 的分类标签，不证明 scheduler/codegen/lowering 全量覆盖。",
        "后两层先补 inventory 与去重关系，再确认新增批次编号；当前草案数量不是全项目最终 T 数量。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timestamp", default="2026-09-06T02:15:00+08:00")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        actual = json.loads((args.repo_root / "upstream/task_backlog.json").read_text(encoding="utf-8"))
        expected = build(args.repo_root, actual["generated_at"])
        if actual != expected or (args.repo_root / "docs/TASK_BACKLOG.md").read_text(encoding="utf-8") != markdown(expected):
            raise ValueError("backlog 与 inventory/manifest 或中文导航不一致，请重新生成并审核")
        print("task_backlog_validation=OK " + json.dumps(expected["counts"], ensure_ascii=False))
        return
    data = build(args.repo_root, args.timestamp)
    print(json.dumps(data, ensure_ascii=False, indent=2) if args.format == "json" else markdown(data), end="\n" if args.format == "json" else "")


if __name__ == "__main__":
    main()
