#!/usr/bin/env python3
"""验证统一 NPU/comparison 结果；只使用标准库，不导入 torch。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SHA256_LENGTH = 64
FORMAL_VERDICTS = {
    "UPSTREAM_CHANGED",
    "NPU_REGRESSION",
    "NEWLY_SUPPORTED",
    "PERF_IMPROVED",
    "PERF_REGRESSED",
    "BEHAVIOR_UNCHANGED",
    "EXPECTED_PRODUCT_DIVERGENCE",
}


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} 顶层必须是 object")
    return value


def require_keys(value: dict[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys - value.keys())
    if missing:
        raise ValueError(f"{context} 缺少字段: {missing}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_sha256(value: Any, context: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} 不是合法 SHA256")


def environment_fingerprint(environment: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in environment.items()
        if key != "fingerprint_sha256"
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_npu_result(
    result: dict[str, Any],
    result_path: Path,
    repo_root: Path,
    manifest_unit: dict[str, Any],
) -> None:
    require_keys(
        result,
        {
            "schema_version",
            "generated_at",
            "case_ids",
            "acceptance_unit_id",
            "upstream_commit",
            "source_tests",
            "tracking_mode",
            "environment",
            "npu_control",
            "paired_control",
            "direct_execution",
            "selected_execution",
            "variants",
            "artifacts",
        },
        str(result_path),
    )
    if result["schema_version"] != "1.1":
        raise ValueError(f"{result_path} schema_version 必须为 1.1")
    if result["acceptance_unit_id"] != manifest_unit["acceptance_unit_id"]:
        raise ValueError(f"{result_path} acceptance_unit_id 与 manifest 不一致")
    if result["tracking_mode"] != manifest_unit["tracking"]["npu_mode"]:
        raise ValueError(f"{result_path} tracking_mode 与 manifest 不一致")
    manifest_tests = {
        test["nodeid"] for test in manifest_unit["community_tests"]
    }
    if not result["case_ids"] or len(result["case_ids"]) != len(
        set(result["case_ids"])
    ):
        raise ValueError(f"{result_path} case_ids 为空或重复")
    if not result["source_tests"] or set(result["source_tests"]) - manifest_tests:
        raise ValueError(f"{result_path} source_tests 不在 manifest")

    environment = result["environment"]
    require_keys(
        environment,
        {
            "fingerprint_sha256",
            "python_version",
            "torch_version",
            "torch_commit",
            "torch_npu_version",
            "torch_npu_commit",
            "triton_version",
            "triton_ascend_commit",
            "cann_version",
            "driver_version",
            "device",
            "backend",
        },
        f"{result_path}.environment",
    )
    validate_sha256(
        environment["fingerprint_sha256"],
        f"{result_path}.environment.fingerprint_sha256",
    )
    if environment_fingerprint(environment) != environment["fingerprint_sha256"]:
        raise ValueError(f"{result_path} environment fingerprint 不一致")
    if environment["torch_commit"] != result["upstream_commit"]:
        raise ValueError(f"{result_path} torch commit 不一致")

    npu_control = result["npu_control"]
    require_keys(
        npu_control,
        {"state", "source_control", "product_gate_bypassed"},
        f"{result_path}.npu_control",
    )
    if npu_control["state"] not in {"enabled", "disabled", "guarded", "patched"}:
        raise ValueError(f"{result_path} NPU control state 非法")
    paired_control = result["paired_control"]
    require_keys(
        paired_control,
        {"status", "reason"},
        f"{result_path}.paired_control",
    )
    if paired_control["status"] not in {
        "not-required",
        "not-run",
        "passed",
        "failed",
    }:
        raise ValueError(f"{result_path} paired control status 非法")

    direct = result["direct_execution"]
    selected = result["selected_execution"]
    for name, execution in (("direct", direct), ("selected", selected)):
        require_keys(
            execution,
            {"status", "execution_success", "tests_run", "tests_skipped", "reason"},
            f"{result_path}.{name}_execution",
        )
    if result["tracking_mode"] == "adapter" and direct["status"] not in {
        "failed",
        "skipped",
        "no-tests",
        "env-blocked",
    }:
        raise ValueError(f"{result_path} adapter 缺少有效 direct blocker")
    if selected["status"] not in {"passed", "failed"}:
        raise ValueError(f"{result_path} selected execution 状态非法")
    if selected["tests_run"] < 1 or selected["tests_skipped"] != 0:
        raise ValueError(f"{result_path} selected execution 测试计数无效")

    manifest_variants = {
        variant["variant_id"]: variant for variant in manifest_unit["variants"]
    }
    result_variants = result["variants"]
    result_variant_ids = [variant["variant_id"] for variant in result_variants]
    if set(result_variant_ids) != set(manifest_variants):
        raise ValueError(f"{result_path} variants 未完整覆盖 manifest")
    if len(result_variant_ids) != len(set(result_variant_ids)):
        raise ValueError(f"{result_path} variant_id 重复")
    for variant in result_variants:
        variant_id = variant["variant_id"]
        require_keys(
            variant,
            {
                "variant_id",
                "kind",
                "evidence_mode",
                "support_status",
                "input_contract",
                "reference_match_expectation",
                "npu_match_expectation",
                "match",
                "fx",
                "replacement",
                "decomposition",
                "lowering",
                "scheduler",
                "codegen",
                "runtime_path",
                "correctness",
                "performance",
                "first_divergence",
                "root_cause",
                "recommended_action",
            },
            f"{result_path}:{variant_id}",
        )
        manifest_variant = manifest_variants[variant_id]
        input_contract = variant["input_contract"]
        require_keys(
            input_contract,
            {"direction", "dtype", "dynamic", "cases"},
            f"{result_path}:{variant_id}.input_contract",
        )
        if variant["evidence_mode"] == "runtime" and not input_contract["cases"]:
            raise ValueError(f"{result_path}:{variant_id} runtime input cases 为空")
        for input_case in input_contract["cases"]:
            require_keys(
                input_case,
                {"shapes", "strides"},
                f"{result_path}:{variant_id}.input_case",
            )
            if len(input_case["shapes"]) != len(input_case["strides"]):
                raise ValueError(f"{result_path}:{variant_id} shape/stride 数量不一致")
        if variant["kind"] != manifest_variant["kind"]:
            raise ValueError(f"{result_path}:{variant_id} kind 与 manifest 不一致")
        manifest_reference_expectation = manifest_variant["expected_reference_match"]
        if not isinstance(manifest_reference_expectation, bool):
            manifest_reference_expectation = None
        if (
            variant["reference_match_expectation"]
            != manifest_reference_expectation
        ):
            raise ValueError(f"{result_path}:{variant_id} reference expectation 漂移")
        if variant["match"]["target_matched"] not in {True, False, None}:
            raise ValueError(f"{result_path}:{variant_id} NPU 目标 match 观测非法")
        if variant["support_status"] == "expected-disabled" and variant[
            "npu_match_expectation"
        ] is not False:
            raise ValueError(f"{result_path}:{variant_id} expected-disabled 必须不命中")
        if variant["evidence_mode"] == "runtime":
            if variant["correctness"]["status"] != "passed":
                raise ValueError(f"{result_path}:{variant_id} runtime correctness 未通过")
        elif variant["correctness"]["status"] not in {
            "not-run",
            "not-applicable",
        }:
            raise ValueError(f"{result_path}:{variant_id} 非运行证据 correctness 非法")

    for artifact in result["artifacts"]:
        require_keys(
            artifact,
            {"role", "path", "sha256", "availability"},
            f"{result_path}.artifacts",
        )
        validate_sha256(artifact["sha256"], f"{result_path}:{artifact['role']}")
        if artifact["availability"] == "repository":
            artifact_path = repo_root / artifact["path"]
            if not artifact_path.is_file():
                raise FileNotFoundError(f"仓库工件不存在: {artifact_path}")
            if sha256(artifact_path) != artifact["sha256"]:
                raise ValueError(f"仓库工件 SHA256 不一致: {artifact_path}")


def validate_comparison(
    comparison: dict[str, Any],
    comparison_path: Path,
    npu_result: dict[str, Any],
    npu_result_path: Path,
    manifest_unit: dict[str, Any],
) -> None:
    require_keys(
        comparison,
        {
            "schema_version",
            "generated_at",
            "acceptance_unit_id",
            "upstream_commit",
            "reference",
            "npu",
            "variant_comparisons",
            "first_divergence",
            "root_cause",
            "recommended_action",
            "final_verdict",
            "repair_status",
        },
        str(comparison_path),
    )
    if comparison["schema_version"] != "1.2":
        raise ValueError(f"{comparison_path} schema_version 必须为 1.2")
    if comparison["acceptance_unit_id"] != npu_result["acceptance_unit_id"]:
        raise ValueError(f"{comparison_path} acceptance_unit_id 不一致")
    if comparison["upstream_commit"] != npu_result["upstream_commit"]:
        raise ValueError(f"{comparison_path} upstream commit 不一致")

    reference = comparison["reference"]
    require_keys(
        reference,
        {
            "run_id",
            "suite_status",
            "suite_valid",
            "environment_fingerprint",
            "payload_sha256",
            "cases",
        },
        f"{comparison_path}.reference",
    )
    for key in ("environment_fingerprint", "payload_sha256"):
        validate_sha256(reference[key], f"{comparison_path}.reference.{key}")
    if reference["suite_status"] != "valid-reference-suite" or not reference[
        "suite_valid"
    ]:
        raise ValueError(f"{comparison_path} reference 无效")
    reference_case_ids = []
    for case in reference["cases"]:
        require_keys(
            case,
            {
                "case_id",
                "case_status",
                "reference_valid",
                "reference_result_sha256",
                "artifact_inventory_sha256",
            },
            f"{comparison_path}.reference.case",
        )
        if case["case_status"] != "passed" or not case["reference_valid"]:
            raise ValueError(f"{comparison_path} reference case 无效")
        for key in ("reference_result_sha256", "artifact_inventory_sha256"):
            validate_sha256(case[key], f"{comparison_path}.reference.case.{key}")
        reference_case_ids.append(case["case_id"])
    if set(reference_case_ids) != set(npu_result["case_ids"]):
        raise ValueError(f"{comparison_path} reference/NPU case_ids 不一致")

    npu = comparison["npu"]
    require_keys(
        npu,
        {
            "result_path",
            "result_sha256",
            "environment_fingerprint",
            "tracking_mode",
            "execution_success",
            "correctness_status",
        },
        f"{comparison_path}.npu",
    )
    if sha256(npu_result_path) != npu["result_sha256"]:
        raise ValueError(f"{comparison_path} npu result SHA256 不一致")
    if npu["environment_fingerprint"] != npu_result["environment"][
        "fingerprint_sha256"
    ]:
        raise ValueError(f"{comparison_path} NPU environment fingerprint 不一致")
    if npu["tracking_mode"] != npu_result["tracking_mode"]:
        raise ValueError(f"{comparison_path} NPU tracking_mode 不一致")
    if npu["correctness_status"] != "passed":
        raise ValueError(f"{comparison_path} NPU 数值正确性门禁未通过")
    if not npu["execution_success"] and comparison["final_verdict"] not in {
        "NPU_REGRESSION",
        "EXPECTED_PRODUCT_DIVERGENCE",
    }:
        raise ValueError(
            f"{comparison_path} 仅回归或经产品 gate 证明的预期分歧允许目标合同执行失败"
        )

    npu_variants = {
        variant["variant_id"]: variant for variant in npu_result["variants"]
    }
    comparisons = comparison["variant_comparisons"]
    comparison_ids = [variant["variant_id"] for variant in comparisons]
    if set(comparison_ids) != set(npu_variants):
        raise ValueError(f"{comparison_path} variant comparisons 不完整")
    if len(comparison_ids) != len(set(comparison_ids)):
        raise ValueError(f"{comparison_path} variant comparison 重复")
    for variant in comparisons:
        variant_id = variant["variant_id"]
        require_keys(
            variant,
            {
                "variant_id",
                "intent",
                "source_locations",
                "gpu_behavior",
                "npu_behavior",
                "reference_contract_stable",
                "reference_target_match",
                "npu_target_match",
                "match_contract_aligned",
                "runtime_path",
                "correctness_status",
                "performance_status",
                "comparison_basis",
                "verdict",
                "note",
            },
            f"{comparison_path}:{variant_id}",
        )
        npu_variant = npu_variants[variant_id]
        for field in ("intent", "gpu_behavior", "npu_behavior"):
            if not isinstance(variant[field], str) or not variant[field].strip():
                raise ValueError(
                    f"{comparison_path}:{variant_id}.{field} 不得为空"
                )
        source_locations = variant["source_locations"]
        if not isinstance(source_locations, list) or not source_locations:
            raise ValueError(
                f"{comparison_path}:{variant_id}.source_locations 不得为空"
            )
        manifest_sources = {
            (source["path"], source["symbol"])
            for source in manifest_unit["upstream_sources"]
        }
        for source in source_locations:
            require_keys(
                source,
                {"path", "line", "symbol"},
                f"{comparison_path}:{variant_id}.source_location",
            )
            if (source["path"], source["symbol"]) not in manifest_sources:
                raise ValueError(
                    f"{comparison_path}:{variant_id} 源码位置未登记在 manifest: "
                    f"{source['path']}::{source['symbol']}"
                )
        if not variant["reference_contract_stable"]:
            raise ValueError(f"{comparison_path}:{variant_id} reference 不稳定")
        if variant["npu_target_match"] != npu_variant["match"]["target_matched"]:
            raise ValueError(f"{comparison_path}:{variant_id} NPU match 不一致")
        if variant["runtime_path"] != npu_variant["runtime_path"]:
            raise ValueError(f"{comparison_path}:{variant_id} runtime path 不一致")
        if not variant["match_contract_aligned"]:
            raise ValueError(f"{comparison_path}:{variant_id} comparison 未闭环")
        evidence_mode = npu_variant["evidence_mode"]
        if evidence_mode == "runtime" and variant["correctness_status"] != "passed":
            raise ValueError(f"{comparison_path}:{variant_id} runtime comparison 未通过")
        if evidence_mode != "runtime" and variant["correctness_status"] not in {
            "not-run",
            "not-applicable",
        }:
            raise ValueError(f"{comparison_path}:{variant_id} 非运行 comparison 状态非法")

    if comparison["final_verdict"] == "NPU_REGRESSION":
        if comparison["repair_status"] not in {
            "queued",
            "in-progress",
            "verified",
            "blocked",
        }:
            raise ValueError(f"{comparison_path} NPU_REGRESSION 未进入 repair 流程")
    elif comparison["final_verdict"] in FORMAL_VERDICTS:
        if comparison["repair_status"] not in {"not-needed", "verified"}:
            raise ValueError(f"{comparison_path} formal verdict 的 repair_status 非闭环")
    elif comparison["final_verdict"] != "INCONCLUSIVE":
        raise ValueError(f"{comparison_path} final_verdict 非法")

    manifest_variant_ids = {
        variant["variant_id"] for variant in manifest_unit["variants"]
    }
    if set(comparison_ids) != manifest_variant_ids:
        raise ValueError(f"{comparison_path} 未覆盖 manifest 全部 variants")


def validate(repo_root: Path) -> None:
    manifest_paths = (
        repo_root / "upstream/manifest.yaml",
        repo_root / "upstream/t077_manifest.yaml",
    )
    manifests = [load_object(path) for path in manifest_paths]
    units: dict[str, dict[str, Any]] = {}
    for manifest_path, manifest in zip(manifest_paths, manifests):
        for unit in manifest["acceptance_units"]:
            unit_id = unit["acceptance_unit_id"]
            if unit_id in units:
                raise ValueError(
                    f"acceptance unit 跨 manifest 重复: {unit_id} ({manifest_path})"
                )
            units[unit_id] = unit
    for schema_name in (
        "schemas/npu_result.schema.json",
        "schemas/comparison_result.schema.json",
    ):
        schema = load_object(repo_root / schema_name)
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise ValueError(f"{schema_name} draft 不符合预期")

    comparison_paths = sorted(
        (repo_root / "results/current").glob("*/comparison_result.json")
    )
    if not comparison_paths:
        raise ValueError("results/current 中没有 comparison_result.json")

    formally_closed = 0
    variant_count = 0
    for comparison_path in comparison_paths:
        comparison = load_object(comparison_path)
        unit_id = comparison["acceptance_unit_id"]
        if unit_id not in units:
            raise ValueError(f"comparison unit 不在 manifest: {unit_id}")
        npu_result_path = repo_root / comparison["npu"]["result_path"]
        if not npu_result_path.is_file():
            raise FileNotFoundError(f"NPU result 不存在: {npu_result_path}")
        npu_result = load_object(npu_result_path)
        validate_npu_result(npu_result, npu_result_path, repo_root, units[unit_id])
        validate_comparison(
            comparison,
            comparison_path,
            npu_result,
            npu_result_path,
            units[unit_id],
        )
        variant_count += len(comparison["variant_comparisons"])
        formally_closed += comparison["final_verdict"] in FORMAL_VERDICTS

    expected_closed = sum(
        manifest["counting_policy"]["current_formally_closed_units"]
        for manifest in manifests
    )
    if expected_closed != formally_closed:
        raise ValueError(
            "manifest current_formally_closed_units 与 comparison 结果数不一致: "
            f"{expected_closed} != {formally_closed}"
        )
    print("comparison_data_validation=OK")
    print(f"comparison_units={len(comparison_paths)}")
    print(f"comparison_variants={variant_count}")
    print(f"formally_closed_units={formally_closed}")
    print("torch_imported=0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    validate(args.repo_root.resolve())


if __name__ == "__main__":
    main()
