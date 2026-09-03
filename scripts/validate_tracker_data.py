#!/usr/bin/env python3
"""验证 tracker 首批 manifest/pass_map；只使用标准库，不导入 torch。"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} 顶层必须是 object")
    return value


def require_keys(value: dict[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys - value.keys())
    if missing:
        raise ValueError(f"{context} 缺少字段: {missing}")


def python_qualnames(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def _visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            result.add(".".join([*self.scope, node.name]))
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

    Visitor().visit(tree)

    # PyTorch 的部分社区测试通过 copy_tests(Source, Target, device) 将模板类
    # 动态复制为可执行 GPU 类；静态验证在不导入 torch 的前提下展开这些名称。
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "copy_tests"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and isinstance(node.args[1], ast.Name)
        ):
            continue
        source = node.args[0].id
        target = node.args[1].id
        source_prefix = f"{source}."
        for qualname in list(result):
            if qualname.startswith(source_prefix):
                result.add(f"{target}.{qualname.removeprefix(source_prefix)}")
    return result


def validate_pytorch_evidence(
    units: list[dict[str, Any]], pytorch_root: Path
) -> None:
    qualname_cache: dict[Path, set[str]] = {}
    for unit in units:
        unit_id = unit["acceptance_unit_id"]
        for source in unit["upstream_sources"]:
            path = pytorch_root / source["path"]
            if not path.is_file():
                raise FileNotFoundError(f"{unit_id} source 不存在: {path}")
            lines = path.read_text(encoding="utf-8").splitlines()
            line = source["line"]
            if not isinstance(line, int) or not 1 <= line <= len(lines):
                raise ValueError(f"{unit_id} source line 越界: {path}:{line}")
            anchor_window = "\n".join(lines[max(0, line - 6) : line + 5])
            if source["symbol"] not in anchor_window:
                raise ValueError(
                    f"{unit_id} source symbol 不在锚点附近: "
                    f"{path}:{line}:{source['symbol']}"
                )

        for test in unit["community_tests"]:
            try:
                relative, qualname = test["nodeid"].split("::", 1)
            except ValueError as error:
                raise ValueError(
                    f"{unit_id} community nodeid 格式错误: {test['nodeid']}"
                ) from error
            path = pytorch_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"{unit_id} community test 不存在: {path}")
            qualnames = qualname_cache.setdefault(path, python_qualnames(path))
            if qualname not in qualnames:
                raise ValueError(
                    f"{unit_id} community test 方法不存在: {path}::{qualname}"
                )


def validate(repo_root: Path, pytorch_root: Path | None) -> None:
    manifest_path = repo_root / "upstream/manifest.yaml"
    map_path = repo_root / "upstream/pass_map.yaml"
    schema_path = repo_root / "upstream/manifest.schema.json"
    reference_plan_path = repo_root / "upstream/reference_plan.yaml"
    reference_plan_schema_path = repo_root / "upstream/reference_plan.schema.json"
    reference_result_schema_path = repo_root / "schemas/reference_result.schema.json"
    candidate_path = (
        repo_root
        / "report/upstream_pass_test_index_20260829/candidate_test_index.csv"
    )

    manifest = load_json(manifest_path)
    pass_map = load_json(map_path)
    schema = load_json(schema_path)
    reference_plan = load_json(reference_plan_path)
    reference_plan_schema = load_json(reference_plan_schema_path)
    reference_result_schema = load_json(reference_result_schema_path)
    require_keys(
        manifest,
        {
            "schema_version",
            "generated_at",
            "status",
            "source_baselines",
            "reference_contract",
            "counting_policy",
            "acceptance_units",
        },
        "manifest",
    )
    require_keys(
        pass_map,
        {
            "schema_version",
            "generated_at",
            "status",
            "source_candidate_index",
            "global_decisions",
            "mapping_entries",
        },
        "pass_map",
    )
    if manifest["schema_version"] != "1.0":
        raise ValueError("manifest schema_version 必须为 1.0")
    if pass_map["schema_version"] != manifest["schema_version"]:
        raise ValueError("manifest/pass_map schema_version 不一致")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError("manifest.schema.json draft 不符合预期")
    for name, value in (
        ("reference_plan.schema.json", reference_plan_schema),
        ("reference_result.schema.json", reference_result_schema),
    ):
        if value.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise ValueError(f"{name} draft 不符合预期")
    reference_contract = manifest["reference_contract"]
    require_keys(
        reference_contract,
        {
            "device_class",
            "device_identity",
            "backend",
            "execution_order",
            "runtime_fingerprint",
        },
        "reference_contract",
    )
    if reference_contract["device_class"] != "GPU":
        raise ValueError("首批 reference device_class 必须为 GPU")
    if reference_contract["execution_order"] != (
        "native-community-test-first-then-minimal-adapter-if-required"
    ):
        raise ValueError("reference execution_order 必须先原生测例再最小适配")

    units = manifest["acceptance_units"]
    if not isinstance(units, list) or len(units) != 5:
        raise ValueError(f"首批 manifest 必须恰为 5 个单元，实际 {len(units)}")
    unit_ids = [unit["acceptance_unit_id"] for unit in units]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("acceptance_unit_id 必须唯一")

    for unit in units:
        unit_id = unit["acceptance_unit_id"]
        require_keys(
            unit,
            {
                "acceptance_unit_id",
                "contract_name",
                "stage",
                "review_status",
                "denominator_eligible",
                "upstream_sources",
                "community_tests",
                "variants",
                "tracking",
                "npu_control",
                "historical_evidence",
            },
            unit_id,
        )
        if unit["review_status"] != "frozen":
            raise ValueError(f"{unit_id} reference 有效后必须 frozen")
        if unit["denominator_eligible"] != "yes-frozen":
            raise ValueError(f"{unit_id} reference 有效后必须进入冻结分母")
        if unit["stage"] not in {"pre_grad", "joint_graph", "post_grad"}:
            raise ValueError(f"{unit_id} stage 非法: {unit['stage']}")
        tracking = unit["tracking"]
        require_keys(
            tracking,
            {
                "reference_mode",
                "npu_mode",
                "allowed_local_deviation",
                "forbidden_local_deviation",
            },
            f"{unit_id}.tracking",
        )
        for mode_key in ("reference_mode", "npu_mode"):
            mode = tracking[mode_key]
            if mode not in {"direct", "adapter", "extracted"}:
                raise ValueError(f"{unit_id} {mode_key} 非法: {mode}")
        if "adapter" in {tracking["reference_mode"], tracking["npu_mode"]}:
            require_keys(tracking, {"adapter_reason"}, f"{unit_id}.tracking")
        if "extracted" in {tracking["reference_mode"], tracking["npu_mode"]}:
            require_keys(tracking, {"extraction_reason"}, f"{unit_id}.tracking")
        for source in unit["upstream_sources"]:
            require_keys(source, {"path", "line", "symbol", "role"}, unit_id)
        for test in unit["community_tests"]:
            require_keys(test, {"nodeid", "role", "evidence_scope"}, unit_id)
            if test["role"] not in {
                "primary-positive",
                "primary-negative",
                "related-regression",
            }:
                raise ValueError(f"{unit_id} community test role 非法")
        for variant in unit["variants"]:
            require_keys(
                variant,
                {
                    "variant_id",
                    "kind",
                    "expected_reference_match",
                    "expected_behavior",
                    "reference_status",
                    "npu_status",
                },
                unit_id,
            )
            if variant["kind"] not in {
                "positive",
                "negative",
                "guard",
                "regression",
            }:
                raise ValueError(f"{unit_id} variant kind 非法")
        variant_ids = [variant["variant_id"] for variant in unit["variants"]]
        if not variant_ids or len(variant_ids) != len(set(variant_ids)):
            raise ValueError(f"{unit_id} variant_id 缺失或重复")
        nodeids = [test["nodeid"] for test in unit["community_tests"]]
        if not nodeids or len(nodeids) != len(set(nodeids)):
            raise ValueError(f"{unit_id} community test 缺失或重复")

    with candidate_path.open(newline="", encoding="utf-8") as handle:
        candidates = {
            row["candidate_id"]: row for row in csv.DictReader(handle)
        }
    candidate_source = manifest["source_baselines"]["t074_candidate_index"]
    actual_candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if actual_candidate_sha != candidate_source["sha256"]:
        raise ValueError("T-074 candidate CSV SHA256 与 manifest 不一致")
    if len(candidates) != candidate_source["candidate_rows"]:
        raise ValueError("T-074 candidate CSV 行数与 manifest 不一致")
    mappings = pass_map["mapping_entries"]
    if not isinstance(mappings, list) or len(mappings) != 5:
        raise ValueError(f"首批 pass_map 必须恰为 5 条，实际 {len(mappings)}")
    mapped_candidates = [mapping["candidate_id"] for mapping in mappings]
    if len(mapped_candidates) != len(set(mapped_candidates)):
        raise ValueError("pass_map candidate_id 必须唯一")
    if set(mapping["acceptance_unit_id"] for mapping in mappings) != set(unit_ids):
        raise ValueError("pass_map 与 manifest 的 unit 集合不一致")

    units_by_id = {unit["acceptance_unit_id"]: unit for unit in units}
    for mapping in mappings:
        candidate_id = mapping["candidate_id"]
        if candidate_id not in candidates:
            raise ValueError(f"pass_map candidate 不在 T-074 v1: {candidate_id}")
        candidate = candidates[candidate_id]
        if mapping["t074_acceptance_unit"] != candidate["acceptance_unit"]:
            raise ValueError(f"{candidate_id} T-074 unit 与原 CSV 不一致")
        unit = units_by_id[mapping["acceptance_unit_id"]]
        if mapping.get("review_status") != unit["review_status"]:
            raise ValueError(
                f"{mapping['acceptance_unit_id']} manifest/pass_map review_status 不一致"
            )
        manifest_tests = {test["nodeid"] for test in unit["community_tests"]}
        mapped_tests = {
            test["nodeid"] for test in mapping["community_test_mapping"]
        }
        if mapped_tests != manifest_tests:
            raise ValueError(
                f"{mapping['acceptance_unit_id']} manifest/pass_map tests 不一致"
            )

    require_keys(
        reference_plan,
        {
            "schema_version",
            "generated_at",
            "status",
            "manifest",
            "execution_policy",
            "cases",
            "non_executed_variants",
        },
        "reference_plan",
    )
    if reference_plan["schema_version"] != "1.0":
        raise ValueError("reference_plan schema_version 必须为 1.0")
    if reference_plan["manifest"] != {
        "path": "upstream/manifest.yaml",
        "schema_version": manifest["schema_version"],
        "pytorch_commit": manifest["source_baselines"]["pytorch"]["commit"],
    }:
        raise ValueError("reference_plan manifest 锚点不一致")
    policy = reference_plan["execution_policy"]
    require_keys(
        policy,
        {
            "order",
            "case_isolation",
            "failure_behavior",
            "artifact_capture",
            "benchmark_gate",
            "default_timeout_seconds",
        },
        "reference_plan.execution_policy",
    )
    if policy["order"] != reference_contract["execution_order"]:
        raise ValueError("reference_plan 未保持 direct-first 顺序")
    if policy["case_isolation"] != "fresh-process":
        raise ValueError("reference case 必须 fresh-process")
    if policy["failure_behavior"] != "continue-and-record":
        raise ValueError("reference case 失败后必须继续")
    if policy["benchmark_gate"] != "functional-reference-valid-first":
        raise ValueError("benchmark 必须位于 reference 功能门禁之后")

    cases = reference_plan["cases"]
    if not isinstance(cases, list) or len(cases) != 13:
        raise ValueError(
            f"首批 reference plan 必须恰为 13 个 case，实际 {len(cases)}"
        )
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("reference case_id 必须唯一")
    manifest_test_pairs = {
        (unit["acceptance_unit_id"], test["nodeid"])
        for unit in units
        for test in unit["community_tests"]
    }
    plan_test_pairs: set[tuple[str, str]] = set()
    covered_variants: set[tuple[str, str]] = set()
    for case in cases:
        require_keys(
            case,
            {
                "case_id",
                "acceptance_unit_id",
                "source_test",
                "tracking_mode",
                "variant_ids",
                "expected_match",
                "expected_assertions",
                "required_artifacts",
                "benchmark",
            },
            case["case_id"],
        )
        if case["tracking_mode"] != "direct":
            raise ValueError(f"首批 case 必须先使用 direct: {case['case_id']}")
        if case["benchmark"] != "not-configured-first-reference-wave":
            raise ValueError(f"首批 case 不得提前配置 benchmark: {case['case_id']}")
        pair = (case["acceptance_unit_id"], case["source_test"])
        if pair in plan_test_pairs:
            raise ValueError(f"reference community test 重复: {pair}")
        plan_test_pairs.add(pair)
        unit = units_by_id.get(case["acceptance_unit_id"])
        if unit is None:
            raise ValueError(f"reference case unit 不存在: {pair}")
        known_variants = {
            variant["variant_id"] for variant in unit["variants"]
        }
        for variant_id in case["variant_ids"]:
            if variant_id not in known_variants:
                raise ValueError(
                    f"reference case variant 不存在: {pair}:{variant_id}"
                )
            covered_variants.add((case["acceptance_unit_id"], variant_id))
    if plan_test_pairs != manifest_test_pairs:
        raise ValueError("reference plan 未一一覆盖 13 个 manifest community tests")

    dispositions: set[tuple[str, str]] = set()
    disposition_status: dict[tuple[str, str], str] = {}
    for item in reference_plan["non_executed_variants"]:
        require_keys(
            item,
            {"acceptance_unit_id", "variant_id", "disposition", "reason"},
            "reference_plan.non_executed_variants",
        )
        key = (item["acceptance_unit_id"], item["variant_id"])
        if key in dispositions or key in covered_variants:
            raise ValueError(f"reference variant 重复执行/排除: {key}")
        dispositions.add(key)
        disposition_status[key] = item["disposition"]
    all_variants = {
        (unit["acceptance_unit_id"], variant["variant_id"])
        for unit in units
        for variant in unit["variants"]
    }
    if covered_variants | dispositions != all_variants:
        raise ValueError("reference plan 未完整处置 20 个 variants")

    for unit in units:
        unit_id = unit["acceptance_unit_id"]
        for variant in unit["variants"]:
            key = (unit_id, variant["variant_id"])
            expected_status = (
                "valid-reference"
                if key in covered_variants
                else disposition_status[key]
            )
            if variant["reference_status"] != expected_status:
                raise ValueError(
                    f"{unit_id}:{variant['variant_id']} reference_status 应为 "
                    f"{expected_status}"
                )

    if pytorch_root is not None:
        validate_pytorch_evidence(units, pytorch_root)

    variant_count = sum(len(unit["variants"]) for unit in units)
    test_count = sum(len(unit["community_tests"]) for unit in units)
    print("tracker_data_validation=OK")
    print(f"acceptance_units={len(units)}")
    print(f"mapped_candidates={len(mappings)}")
    print(f"variants={variant_count}")
    print(f"community_tests={test_count}")
    print(f"reference_cases={len(cases)}")
    print(f"reference_executed_variants={len(covered_variants)}")
    print(f"reference_non_executed_variants={len(dispositions)}")
    denominator_frozen = sum(
        unit["denominator_eligible"] == "yes-frozen" for unit in units
    )
    print(f"denominator_frozen={denominator_frozen}")
    print("torch_imported=0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--pytorch-root", type=Path)
    args = parser.parse_args()
    validate(
        args.repo_root.resolve(),
        args.pytorch_root.resolve() if args.pytorch_root else None,
    )


if __name__ == "__main__":
    main()
