#!/usr/bin/env python3
"""零设备校验T-085 manifest/reference/performance/讲解/入口。"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys


TASK_ID = "T-085"
EXPECTED_COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
EXPECTED_UNITS = {
    "AU-post-grad-overlap-scheduling-device-put-sync",
    "AU-post-grad-partitioned-scatter-optimization",
    "AU-post-grad-pointless-cumsum",
}


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"顶层必须是object：{path}")
    return value


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aware_timestamp(value: object, label: str) -> None:
    if not isinstance(value, str) or datetime.fromisoformat(value).utcoffset() is None:
        raise ValueError(f"{label}必须是带时区时间戳")


def validate(repo_root: Path, pytorch_root: Path) -> dict[str, int]:
    manifest_path = repo_root / "upstream/t085_manifest.yaml"
    reference_path = repo_root / "upstream/t085_reference_plan.yaml"
    performance_path = repo_root / "upstream/t085_performance_plan.yaml"
    guide_path = repo_root / "docs/T085_FUNCTION_PERFORMANCE_GUIDE.md"
    worker_path = repo_root / "runners/t085_performance_worker.py"
    launcher_path = repo_root / "scripts/run_t085_performance.sh"
    orchestrator_path = repo_root / "scripts/run_t085_performance.py"
    gpu_path = repo_root / "scripts/run_t085_gpu_all.sh"

    manifest = load(manifest_path)
    reference = load(reference_path)
    performance = load(performance_path)
    for label, value in (
        ("manifest", manifest),
        ("reference", reference),
        ("performance", performance),
    ):
        aware_timestamp(value.get("generated_at"), label)

    if reference.get("task_id") != TASK_ID or performance.get("task_id") != TASK_ID:
        raise ValueError("task_id不一致")
    if manifest["source_baselines"]["pytorch"]["commit"] != EXPECTED_COMMIT:
        raise ValueError("manifest冻结commit不符")
    if reference["manifest"]["pytorch_commit"] != EXPECTED_COMMIT:
        raise ValueError("reference冻结commit不符")
    if reference["execution_policy"]["order"] != "native-community-test-first-then-minimal-adapter-if-required":
        raise ValueError("必须先原生社区例再最小适配")
    if reference["execution_policy"]["minimum_gpus"] != 2:
        raise ValueError("T-085完整GPU suite必须声明两张卡")

    units = {unit["acceptance_unit_id"]: unit for unit in manifest["acceptance_units"]}
    if set(units) != EXPECTED_UNITS or len(units) != len(manifest["acceptance_units"]):
        raise ValueError("T-085 acceptance units集合或唯一性不符")
    frozen = manifest["counting_policy"]["current_frozen_denominator_units"]
    suite_status = manifest.get("reference_contract", {}).get("suite_status")
    if suite_status == "valid-reference-suite":
        if frozen != len(units):
            raise ValueError("GPU reference验收后冻结数必须等于已选单元数")
        review = load(repo_root / "results/current/T-085/gpu_reference_review.json")
        if (
            review.get("review_status") != "accepted-mixed-rank-gpu-reference"
            or review.get("pytorch_commit") != EXPECTED_COMMIT
            or review.get("suite", {}).get("valid_cases") != len(reference["cases"])
            or review.get("suite", {}).get("tests_skipped") != 0
            or review.get("evidence_scope", {}).get(
                "overlap_real_two_rank_cuda_nccl"
            )
            is not True
        ):
            raise ValueError("T-085 GPU复核记录与冻结合同不一致")
    elif frozen != 0:
        raise ValueError("GPU reference未验收前不得冻结分母")
    if manifest["counting_policy"]["current_formally_closed_units"] not in {
        0,
        len(units),
    }:
        raise ValueError("正式闭环数只能为0或已选单元总数")
    if len(manifest.get("deferred_candidates", [])) != 2:
        raise ValueError("两个不合格候选必须留有deferred记录")

    manifest_tests = {
        (unit_id, test["nodeid"])
        for unit_id, unit in units.items()
        for test in unit["community_tests"]
    }
    plan_tests = {
        (case["acceptance_unit_id"], case["source_test"])
        for case in reference["cases"]
    }
    if manifest_tests != plan_tests:
        raise ValueError("reference cases没有一一覆盖manifest community tests")
    if len({case["case_id"] for case in reference["cases"]}) != len(reference["cases"]):
        raise ValueError("case_id重复")

    generic = load_module("t085_reference_contract", repo_root / "runners/reference_runner.py")
    generic.validate_contract(manifest, reference, pytorch_root)

    performance_units = {
        unit["acceptance_unit_id"]: unit for unit in performance["acceptance_units"]
    }
    if set(performance_units) != EXPECTED_UNITS:
        raise ValueError("performance units与manifest不一致")
    backend = performance["backend_contract"]
    if backend.get("npu_backend") != "triton_experimental":
        raise ValueError("NPU后端必须是triton_experimental")
    if backend.get("process_isolation") != "fresh-process-per-arm":
        raise ValueError("OFF/ON必须新进程")
    if backend.get("off_on_order") != ["OFF1", "ON1", "ON2", "OFF2", "OFF3", "ON3"]:
        raise ValueError("六臂顺序不符")
    scatter = performance_units["AU-post-grad-partitioned-scatter-optimization"]
    if (
        scatter.get("worker_unit") != "partitioned-scatter"
        or "capability-functional-gate"
        not in scatter.get("performance_status", "")
    ):
        raise ValueError("partitioned-scatter必须先经过NPU capability功能门禁")
    if "force" not in scatter["off_on_control"]:
        raise ValueError("partitioned-scatter必须明确禁止force制造ON")
    if performance["implementation"]["status"] not in {
        "implemented-awaiting-runtime-validation",
        "implemented-runtime-validated",
    }:
        raise ValueError("性能worker状态非法")

    ast.parse(worker_path.read_text(encoding="utf-8"), filename=str(worker_path))
    ast.parse(
        orchestrator_path.read_text(encoding="utf-8"),
        filename=str(orchestrator_path),
    )
    worker = load_module("t085_worker_contract", worker_path)
    if set(worker.TARGETS) != {
        "pointless-cumsum",
        "overlap-device-put",
        "partitioned-scatter",
    }:
        raise ValueError("worker必须覆盖T-085三个验收单元")

    guide = guide_path.read_text(encoding="utf-8")
    for marker in (
        "更新时间",
        "功能测例",
        "性能测例",
        "triton_experimental",
        *sorted(EXPECTED_UNITS),
    ):
        if marker not in guide:
            raise ValueError(f"中文guide缺少{marker}")

    for path in (launcher_path, gpu_path, repo_root / "scripts/run_t085_reference_all.sh"):
        result = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
        if result.returncode:
            raise ValueError(f"shell语法错误：{path}: {result.stderr}")
    gpu_source = gpu_path.read_text(encoding="utf-8")
    if "--gpus" not in gpu_source or "CUDA_VISIBLE_DEVICES" not in gpu_source:
        raise ValueError("GPU一键入口未声明双卡")

    return {
        "units": len(units),
        "cases": len(reference["cases"]),
        "variants": sum(len(case["variant_ids"]) for case in reference["cases"]),
        "deferred": len(manifest["deferred_candidates"]),
        "performance_runnable": len(worker.TARGETS),
        "performance_blocked": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--pytorch-root", type=Path, default=Path("/home/z50063656/Pass/src/pytorch"))
    args = parser.parse_args()
    repo_root = (args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    counts = validate(repo_root, args.pytorch_root.resolve())
    print(
        "t085_preparation_validation=OK "
        + " ".join(f"{key}={value}" for key, value in counts.items())
        + " torch_imported=0 validator_device_execution=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
