#!/usr/bin/env python3
"""将 GPU/reference run 导出为可复制、可校验的紧凑 JSON 文本证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


FORMAT_VERSION = "1.0"
ROOT_EVIDENCE_FILES = (
    "environment.json",
    "manifest_snapshot.json",
    "reference_plan_snapshot.json",
    "reference_summary.json",
)
CASE_EVIDENCE_FILES = (
    "artifact_inventory.json",
    "benchmark.json",
    "fx_after.txt",
    "fx_before.txt",
    "metadata.json",
    "reference_result.json",
    "stderr.log",
    "stdout.log",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"缺少必需文件：{path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON 无法解析：{path}: {error}") from error


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是 JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} 必须是 JSON array")
    return value


def file_record(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"缺少证据文件：{path}")
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def case_audit(run_dir: Path, item: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_id = item.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("reference_summary.json 中存在无效 case_id")
    cases_root = (run_dir / "cases").resolve()
    case_dir = (cases_root / case_id).resolve()
    if not case_dir.is_relative_to(cases_root):
        raise ValueError(f"case_id 越出 cases 目录：{case_id}")

    result_path = case_dir / "reference_result.json"
    inventory_path = case_dir / "artifact_inventory.json"
    result = require_mapping(load_json(result_path), str(result_path))
    inventory = require_list(load_json(inventory_path), str(inventory_path))
    case = require_mapping(result.get("case"), f"{case_id}.case")
    source = require_mapping(result.get("source"), f"{case_id}.source")
    execution = require_mapping(result.get("execution"), f"{case_id}.execution")
    fx = require_mapping(result.get("fx"), f"{case_id}.fx")
    before = require_mapping(fx.get("before"), f"{case_id}.fx.before")
    after = require_mapping(fx.get("after"), f"{case_id}.fx.after")

    artifact_bytes = 0
    for artifact in inventory:
        artifact = require_mapping(artifact, f"{case_id}.artifact_inventory[]")
        size = artifact.get("bytes")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"{case_id} artifact bytes 无效")
        artifact_bytes += size

    audit = {
        "case_id": case_id,
        "acceptance_unit_id": case.get("acceptance_unit_id"),
        "variant_ids": case.get("variant_ids"),
        "status": execution.get("status"),
        "return_code": execution.get("return_code"),
        "duration_seconds": execution.get("duration_seconds"),
        "tests_ran": execution.get("tests_ran"),
        "tests_skipped": execution.get("tests_skipped"),
        "reference_valid": result.get("reference_valid"),
        "adapter_decision": result.get("adapter_decision"),
        "actual_commit": source.get("actual_commit"),
        "fx_before_captured": before.get("captured"),
        "fx_before_count": before.get("count"),
        "fx_before_signature": before.get("normalized_signature"),
        "fx_after_captured": after.get("captured"),
        "fx_after_count": after.get("count"),
        "fx_after_signature": after.get("normalized_signature"),
        "fx_stable_signature": fx.get("stable_signature"),
        "artifact_count": len(inventory),
        "artifact_bytes": artifact_bytes,
        "reference_result_sha256": sha256_file(result_path),
        "artifact_inventory_sha256": sha256_file(inventory_path),
    }
    evidence = [
        file_record(case_dir / name, run_dir)
        for name in CASE_EVIDENCE_FILES
        if (case_dir / name).is_file()
    ]
    return audit, evidence


def build_payload(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise ValueError(f"run 目录不存在：{run_dir}")
    environment = require_mapping(
        load_json(run_dir / "environment.json"), "environment.json"
    )
    summary = require_mapping(
        load_json(run_dir / "reference_summary.json"), "reference_summary.json"
    )
    summary_cases = require_list(summary.get("cases"), "reference_summary.json.cases")

    audits: list[dict[str, Any]] = []
    evidence = [
        file_record(run_dir / name, run_dir)
        for name in ROOT_EVIDENCE_FILES
        if (run_dir / name).is_file()
    ]
    for item in summary_cases:
        audit, case_evidence = case_audit(
            run_dir, require_mapping(item, "reference_summary.json.cases[]")
        )
        audits.append(audit)
        evidence.extend(case_evidence)

    payload: dict[str, Any] = {
        "handoff_format_version": FORMAT_VERSION,
        "source_run_dir": str(run_dir),
        "environment": environment,
        "reference_summary": summary,
        "case_audit": audits,
        "evidence_files": sorted(evidence, key=lambda item: item["path"]),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 reference run 导出为适合终端复制的单个 JSON 文本证据包"
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="reference-<timestamp> 目录")
    parser.add_argument("--output", type=Path, help="输出文件；省略时写到 stdout")
    parser.add_argument("--compact", action="store_true", help="输出单行 JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_dir = args.run_dir.resolve()
        if args.output is not None:
            output = args.output.resolve()
            if output == run_dir or output.is_relative_to(run_dir):
                raise ValueError("文本 handoff 必须写在原始 run 目录外")
        payload = build_payload(run_dir)
        indent = None if args.compact else 2
        content = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=indent
        ) + "\n"
        if args.output is None:
            sys.stdout.write(content)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
            print(f"text_handoff={args.output.resolve()}")
            print(f"payload_sha256={payload['payload_sha256']}")
            print(f"bytes={len(content.encode('utf-8'))}")
        return 0
    except (OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
