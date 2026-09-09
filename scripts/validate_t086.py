#!/usr/bin/env python3
"""T-086零设备准备校验；只用标准库，不导入torch。"""

from __future__ import annotations

import ast
import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTORCH = Path("/home/z50063656/Pass/src/pytorch")
TASK = "T-086"
UNIT = "AU-post-grad-reinplace-inplaceable-ops"
DEFERRED = {
    "AU-post-grad-reciprocal-sqrt-to-rsqrt",
    "AU-post-grad-remove-assert-ops",
    "AU-post-grad-remove-noop-ops",
    "AU-post-grad-remove-profiler-ops",
}


def load(relative: str) -> dict:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{relative}顶层必须是object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def aware_timestamp(value: object, label: str) -> None:
    require(isinstance(value, str) and bool(value), f"{label}缺时间戳")
    require(datetime.fromisoformat(value).utcoffset() is not None, f"{label}必须带时区")


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(pytorch_root: Path | None = None) -> dict[str, int]:
    pytorch_root = (pytorch_root or DEFAULT_PYTORCH).resolve()
    manifest = load("upstream/t086_manifest.yaml")
    reference = load("upstream/t086_reference_plan.yaml")
    performance = load("upstream/t086_performance_plan.yaml")
    backlog = load("upstream/task_backlog.json")
    for label, value in (
        ("manifest", manifest),
        ("reference", reference),
        ("performance", performance),
    ):
        require(value.get("schema_version") == "1.0", f"{label} schema错误")
        aware_timestamp(value.get("generated_at"), label)

    require(reference.get("task_id") == TASK, "reference task_id错误")
    require(performance.get("task_id") == TASK, "performance task_id错误")
    require(
        reference.get("performance_plan", {}).get("path")
        == "upstream/t086_performance_plan.yaml",
        "reference未指向T-086性能计划",
    )
    guide_relative = reference.get("case_guide", {}).get("path")
    require(guide_relative == "docs/T086_FUNCTION_PERFORMANCE_GUIDE.md", "中文guide路径错误")
    guide = (ROOT / guide_relative).read_text(encoding="utf-8")
    for marker in (
        "更新时间",
        "功能测例",
        "性能测例",
        "triton_experimental",
        UNIT,
        "实际设备",
        "非端到端",
    ):
        require(marker in guide, f"中文guide缺少{marker!r}")

    units = manifest.get("acceptance_units")
    require(isinstance(units, list) and len(units) == 1, "T-086只能选择1个已审核单元")
    require(units[0].get("acceptance_unit_id") == UNIT, "选择单元错误")
    require(
        manifest.get("counting_policy", {}).get("current_frozen_denominator_units") == 0,
        "设备运行前不得冻结分母",
    )
    deferred = {
        item.get("provisional_unit_id")
        for item in manifest.get("deferred_candidates", [])
    }
    require(deferred == DEFERRED, "deferred候选遗漏或多记")
    require(
        all(item.get("denominator_eligible") is False for item in manifest["deferred_candidates"]),
        "deferred候选不得进入分母",
    )

    backlog_task = next(item for item in backlog["batches"] if item["task_id"] == TASK)
    backlog_ids = {item["provisional_unit_id"] for item in backlog_task["units"]}
    require(backlog_ids == deferred | {UNIT}, "backlog候选未被完整处置")

    reference_runner = load_module("t086_reference_runner", "runners/reference_runner.py")
    counts = reference_runner.validate_contract(manifest, reference, pytorch_root)
    require(counts["acceptance_units"] == 1, "reference unit计数错误")
    require(counts["cases"] == 2 and counts["variants"] == 2, "reference case/variant计数错误")

    backend = performance.get("backend_contract", {})
    require(backend.get("npu_backend") == "triton_experimental", "NPU backend必须固定")
    require(backend.get("process_isolation") == "fresh-process-per-arm", "OFF/ON必须独立进程")
    require(
        backend.get("off_on_order") == ["OFF1", "ON1", "ON2", "OFF2", "OFF3", "ON3"],
        "六臂顺序错误",
    )
    perf_units = performance.get("acceptance_units")
    require(isinstance(perf_units, list) and len(perf_units) == 1, "性能计划单元错误")
    perf = perf_units[0]
    require(perf.get("acceptance_unit_id") == UNIT, "性能单元与manifest不一致")
    require(
        perf.get("case_source", {}).get("kind")
        == "tracker-derived-from-community-functional-case",
        "性能来源必须明确为社区功能图派生",
    )
    community = {item["nodeid"] for item in units[0]["community_tests"]}
    require(set(perf["case_source"]["nodeids"]) == community, "性能来源nodeid未绑定社区合同")
    require(performance["implementation"]["status"] == "implemented-awaiting-runtime-validation", "worker状态不能冒充实测")

    worker = ROOT / performance["implementation"]["entrypoint"]
    launcher = ROOT / performance["implementation"]["launcher"]
    require(worker.is_file() and launcher.is_file(), "性能worker或launcher不存在")
    worker_text = worker.read_text(encoding="utf-8")
    worker_tree = ast.parse(worker_text, filename=str(worker))
    top_imports = {
        alias.name
        for node in worker_tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    require("torch" not in top_imports and "torch_npu" not in top_imports, "worker不得顶层导入设备库")
    backend_position = worker_text.index('os.environ["TORCHINDUCTOR_NPU_BACKEND"]')
    torch_position = worker_text.index("    import torch\n", backend_position)
    require(backend_position < torch_position, "backend必须在import torch前选择")
    require("reinplace_inplaceable_ops = observed" in worker_text, "worker未做精确目标观察")
    require("negative_guard" in worker_text, "worker未复核活跃input负例")
    launcher_text = launcher.read_text(encoding="utf-8")
    require('ORDER = ("off1", "on1", "on2", "off2", "off3", "on3")' in launcher_text, "launcher六臂顺序错误")
    require("start_new_session=True" in launcher_text, "launcher未隔离子进程组")

    reference_launcher = ROOT / "scripts/run_t086_reference_all.sh"
    incoming = ROOT / "results/incoming/T-086/README.md"
    require(reference_launcher.is_file(), "T-086 reference入口缺失")
    require(incoming.is_file(), "T-086 incoming README缺失")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-root", type=Path, default=DEFAULT_PYTORCH)
    args = parser.parse_args()
    counts = validate(args.pytorch_root)
    print(
        "t086_preparation_validation=OK "
        f"units={counts['acceptance_units']} cases={counts['cases']} "
        f"variants={counts['variants']} deferred={len(DEFERRED)} "
        "device_execution=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
