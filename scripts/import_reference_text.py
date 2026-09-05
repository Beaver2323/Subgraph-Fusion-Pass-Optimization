#!/usr/bin/env python3
"""校验原文 handoff 并在新目录恢复文本，仅作数据读取，绝不导入或执行生成代码。"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from export_reference_text import MAX_TEXT_BYTES, safe_relative_path, seal_payload


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是 JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} 必须是 JSON array")
    return value


def validate_record(value: Any, label: str) -> dict[str, Any]:
    record = require_mapping(value, label)
    safe_relative_path(record.get("path"))
    size = record.get("bytes")
    digest = record.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{label}.bytes 必须是非负整数")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label}.sha256 必须是小写 SHA256")
    return record


def validate_payload(payload: dict[str, Any]) -> tuple[str, dict[str, bytes]]:
    payload = require_mapping(payload, "handoff")
    expected = payload.get("payload_sha256")
    copied = dict(payload)
    seal_payload(copied)
    if expected != copied["payload_sha256"]:
        raise ValueError("handoff 整包 SHA256 不一致，可能复制不完整或被修改")
    if payload.get("handoff_format_version") != "1.1":
        raise ValueError("不是带原文的 1.1 handoff；紧凑 1.0 不能恢复 FX 正文")
    raw_text_files = require_list(payload.get("raw_text_files"), "raw_text_files")
    summary = require_mapping(payload.get("reference_summary"), "reference_summary")
    run_id = summary.get("run_id")
    if len(safe_relative_path(run_id).parts) != 1:
        raise ValueError("run_id 必须是单个目录名")
    data, total = {}, 0
    for index, value in enumerate(raw_text_files):
        item = validate_record(value, f"raw_text_files[{index}]")
        relative = item["path"]
        if (
            relative in data
            or item.get("encoding") != "utf-8"
            or not isinstance(item.get("text"), str)
        ):
            raise ValueError("原文存在重复路径或编码错误")
        content = item["text"].encode("utf-8")
        total += len(content)
        if total > MAX_TEXT_BYTES:
            raise ValueError("原文总量超过 64 MiB")
        if (
            len(content) != item["bytes"]
            or hashlib.sha256(content).hexdigest() != item["sha256"]
        ):
            raise ValueError(f"原文大小/哈希不一致：{relative}")
        data[relative] = content
    for relative in data:
        if any(
            str(parent) in data
            for parent in safe_relative_path(relative).parents
            if str(parent) != "."
        ):
            raise ValueError("原文路径存在文件/目录冲突")
    evidence_files = require_list(payload.get("evidence_files"), "evidence_files")
    for index, value in enumerate(evidence_files):
        record = validate_record(value, f"evidence_files[{index}]")
        content = data.get(record["path"])
        if (
            content is None
            or len(content) != record["bytes"]
            or hashlib.sha256(content).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"缺失或不一致的必需原文：{record['path']}")
    transfer = require_mapping(payload.get("raw_text_transfer"), "raw_text_transfer")
    if transfer["embedded_files"] != len(data) or transfer["embedded_bytes"] != total:
        raise ValueError("原文数量/字节统计不一致")
    # 结构化摘要与原始 JSON 也必须一致，不能用另一轮的正文冒充本轮摘要。
    for name in ("environment", "reference_summary"):
        if json.loads(data[f"{name}.json"]) != payload[name]:
            raise ValueError(f"摘要与原文不一致：{name}")
    indexed = {item["path"]: item for item in evidence_files}
    if len(indexed) != len(evidence_files):
        raise ValueError("必需证据存在重复路径")
    case_audit = require_list(payload.get("case_audit"), "case_audit")
    for case_index, value in enumerate(case_audit):
        case = require_mapping(value, f"case_audit[{case_index}]")
        case_id = case["case_id"]
        if len(safe_relative_path(case_id).parts) != 1:
            raise ValueError("case_id 必须是单个目录名")
        inventory_path = f"cases/{case_id}/artifact_inventory.json"
        inventory = require_list(
            json.loads(data[inventory_path]), f"{inventory_path}"
        )
        for item_index, value in enumerate(inventory):
            item = validate_record(value, f"{inventory_path}[{item_index}]")
            path = f"cases/{case_id}/{item['path']}"
            record = {"path": path, "bytes": item["bytes"], "sha256": item["sha256"]}
            if path in indexed and indexed[path] != record:
                raise ValueError(f"inventory 与正文登记冲突：{path}")
            indexed[path] = record
    omitted_files = require_list(transfer.get("omitted_files"), "omitted_files")
    for index, value in enumerate(omitted_files):
        validate_record(value, f"omitted_files[{index}]")
    omitted = {item["path"]: item for item in omitted_files}
    if len(omitted) != len(omitted_files) or set(data) & set(omitted):
        raise ValueError("缺项路径重复或同时存在正文")
    if set(indexed) != set(data) | set(omitted):
        raise ValueError("原文/缺项未完整覆盖已登记文件，或包含未登记文件")
    for path, record in indexed.items():
        if path in data:
            if (
                len(data[path]) != record["bytes"]
                or hashlib.sha256(data[path]).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"原文与 inventory 不一致：{path}")
        elif any(omitted[path].get(key) != record[key] for key in ("bytes", "sha256")):
            raise ValueError(f"缺项与 inventory 不一致：{path}")
    if transfer["all_registered_artifacts_embedded"] != (not omitted):
        raise ValueError("归档完整性标记不一致")
    return run_id, data


def restore(payload: dict[str, Any], output_root: Path) -> Path:
    run_id, data = validate_payload(payload)
    output_root = output_root.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if output_root.is_relative_to(repo_root):
        raise ValueError("原文恢复目录必须在 tracker 仓库外")
    output_root.mkdir(parents=True, exist_ok=True)
    container = Path(tempfile.mkdtemp(prefix="text-import-", dir=output_root))
    run_dir = container / run_id
    run_dir.mkdir()
    for relative, content in sorted(data.items()):
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
    receipt = {
        "payload_sha256": payload["payload_sha256"],
        "restored_files": len(data),
        "source_run_dir": payload["source_run_dir"],
        "local_run_dir": str(run_dir),
        "omitted_files": payload["raw_text_transfer"]["omitted_files"],
        "code_executed": False,
    }
    (container / "import_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只校验整包、逐文件哈希和 inventory 绑定，不恢复文件",
    )
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if args.validate_only:
            run_id, data = validate_payload(payload)
            print(
                f"handoff_validation=OK run_id={run_id} "
                f"restorable_text_files={len(data)} code_executed=false"
            )
            return 0
        if args.output_root is None:
            raise ValueError("恢复文件时必须指定 --output-root；只校验可使用 --validate-only")
        run_dir = restore(payload, args.output_root)
        print(f"restored_run={run_dir}")
        print("code_executed=false verdict=not-yet-reviewed")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"错误：{error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
