#!/usr/bin/env python3
"""将 GPU/reference run 导出为可复制、可校验的 JSON 文本证据。"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
import lzma
import sys
import zlib
from pathlib import Path, PurePosixPath
from typing import Any


FORMAT_VERSION = "1.0"
RAW_TEXT_FORMAT_VERSION = "1.1"
COMPRESSED_RAW_TEXT_FORMAT_VERSION = "1.2"
REVIEW_FORMAT_VERSION = "1.3"
BUNDLE_FORMAT_VERSION = "1.4"
SPLIT_FORMAT_VERSION = "1.0"
COMPRESSED_SPLIT_FORMAT_VERSION = "1.1"
DEFAULT_SPLIT_PART_BYTES = 48 * 1024
DEFAULT_AUTO_SPLIT_THRESHOLD_BYTES = 96 * 1024
MIN_SPLIT_PART_BYTES = 16 * 1024
MAX_SPLIT_PART_BYTES = 512 * 1024
BASE64_LINE_CHARS = 1024
TEXT_SUFFIXES = {
    ".asm",
    ".c",
    ".cpp",
    ".csv",
    ".cu",
    ".dot",
    ".h",
    ".hpp",
    ".html",
    ".json",
    ".jsonl",
    ".ll",
    ".log",
    ".md",
    ".mlir",
    ".ptx",
    ".py",
    ".s",
    ".ttgir",
    ".ttir",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_TEXT_BYTES = 64 * 1024 * 1024
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
REVIEW_ROOT_FILES = frozenset(("environment.json", "reference_summary.json"))
REVIEW_CASE_FILES = frozenset(
    (
        "artifact_inventory.json",
        "benchmark.json",
        "fx_after.txt",
        "fx_before.txt",
        "metadata.json",
        "reference_result.json",
    )
)
REVIEW_AUDIT_BASENAMES = frozenset(
    (
        "fx_graph_readable.py",
        "fx_graph_transformed.py",
        "ir_post_fusion.txt",
        "ir_pre_fusion.txt",
        "output_code.py",
        "stderr.log",
        "stdout.log",
    )
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


def safe_relative_path(relative: str) -> PurePosixPath:
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or any(ord(character) < 32 for character in relative)
    ):
        raise ValueError("证据路径必须是非空 POSIX 相对路径")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(
        part in {"", ".", "..", ".git"} for part in relative.split("/")
    ):
        raise ValueError(f"证据路径不安全：{relative}")
    return path


def checked_file(root: Path, relative: str) -> Path:
    path = root
    for part in safe_relative_path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"证据路径不得含软链接：{relative}")
    if not path.is_file():
        raise ValueError(f"缺少证据文件：{relative}")
    return path


def seal_payload(payload: dict[str, Any]) -> None:
    payload.pop("payload_sha256", None)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()


def write_split_payload(
    content: bytes,
    output_dir: Path,
    *,
    payload_sha256: str,
    part_bytes: int,
    compress_transport: bool = True,
) -> dict[str, Any]:
    """将完整 handoff 序列化文本拆为多个可单独粘贴的 JSON 分片。"""
    if not MIN_SPLIT_PART_BYTES <= part_bytes <= MAX_SPLIT_PART_BYTES:
        raise ValueError(
            f"--split-part-bytes 必须在 {MIN_SPLIT_PART_BYTES}～{MAX_SPLIT_PART_BYTES} 之间"
        )
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"分片输出目录必须不存在：{output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    source_sha256 = hashlib.sha256(content).hexdigest()
    transport = lzma.compress(content, preset=6) if compress_transport else content
    version = COMPRESSED_SPLIT_FORMAT_VERSION if compress_transport else SPLIT_FORMAT_VERSION
    chunks = [
        transport[offset : offset + part_bytes]
        for offset in range(0, len(transport), part_bytes)
    ]
    if not chunks:
        chunks = [b""]
    records = []
    for index, raw in enumerate(chunks, start=1):
        name = f"part-{index:04d}.json"
        encoded = base64.b64encode(raw).decode("ascii")
        part = {
            "split_format_version": version,
            "source_sha256": source_sha256,
            "index": index,
            "part_count": len(chunks),
            "source_offset": (index - 1) * part_bytes,
            "payload_bytes": len(raw),
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
            "encoding": "base64",
            "data": [
                encoded[offset : offset + BASE64_LINE_CHARS]
                for offset in range(0, len(encoded), BASE64_LINE_CHARS)
            ],
        }
        part_content = json.dumps(part, ensure_ascii=False, indent=2) + "\n"
        with (output_dir / name).open("x", encoding="utf-8") as handle:
            handle.write(part_content)
        records.append(
            {
                "path": name,
                "index": index,
                "source_offset": part["source_offset"],
                "payload_bytes": len(raw),
                "payload_sha256": part["payload_sha256"],
                "transport_file_bytes": len(part_content.encode("utf-8")),
            }
        )
    manifest = {
        "split_format_version": version,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_name": "text-handoff.json",
        "source_bytes": len(content),
        "source_sha256": source_sha256,
        "payload_sha256": payload_sha256,
        "part_payload_bytes": part_bytes,
        "part_count": len(records),
        "parts": records,
    }
    if compress_transport:
        manifest.update(
            transport_encoding="xz",
            transport_bytes=len(transport),
            transport_sha256=hashlib.sha256(transport).hexdigest(),
        )
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def review_text_paths(
    payload: dict[str, Any], records: dict[str, dict[str, Any]]
) -> set[str]:
    """返回评审正文；保留小型图/IR/代码证据，省略大 trace 与 runnable。"""
    selected = set(REVIEW_ROOT_FILES)
    for case in payload["case_audit"]:
        case_id = case["case_id"]
        selected.update(
            f"cases/{case_id}/{name}" for name in REVIEW_CASE_FILES
        )
        case_prefix = f"cases/{case_id}/"
        selected.update(
            path
            for path in records
            if path.startswith(case_prefix)
            and PurePosixPath(path).name in REVIEW_AUDIT_BASENAMES
        )
    return selected


def include_raw_text(
    payload: dict[str, Any],
    run_dir: Path,
    *,
    compress: bool = False,
    profile: str = "archive",
) -> None:
    """嵌入已登记文件的 UTF-8 原文；二进制只列缺项，不编码伪装成文本。"""
    if profile not in {"archive", "review"}:
        raise ValueError(f"未知 handoff profile：{profile}")
    if profile == "review" and not compress:
        raise ValueError("review profile 必须使用压缩原文")
    records = {item["path"]: item for item in payload["evidence_files"]}
    for case in payload["case_audit"]:
        case_id = case["case_id"]
        safe_relative_path(case_id)
        if "/" in case_id:
            raise ValueError("case_id 必须是单个目录名")
        inventory = load_json(
            checked_file(run_dir, f"cases/{case_id}/artifact_inventory.json")
        )
        seen = set()
        for item in require_list(inventory, "artifact_inventory"):
            relative = item["path"]
            safe_relative_path(relative)
            if relative in seen:
                raise ValueError(f"inventory 存在重复路径：{relative}")
            seen.add(relative)
            path = f"cases/{case_id}/{relative}"
            record = {"path": path, "bytes": item["bytes"], "sha256": item["sha256"]}
            if path in records and records[path] != record:
                raise ValueError(f"文件与 inventory 哈希不一致：{path}")
            records[path] = record
    selected = review_text_paths(payload, records) if profile == "review" else None
    if selected is not None and not selected <= set(records):
        missing = ", ".join(sorted(selected - set(records)))
        raise ValueError(f"review profile 缺少必需文件：{missing}")
    embedded, omitted, total, compressed_total = [], [], 0, 0
    for relative, record in sorted(records.items()):
        path = checked_file(run_dir, relative)
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"证据文件大小/哈希与登记值不一致：{relative}")
        if selected is not None and relative not in selected:
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"证据文件大小/哈希与登记值不一致：{relative}")
            omitted.append(dict(record, reason="review-profile-hash-only"))
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"证据文件大小/哈希与登记值不一致：{relative}")
            omitted.append(dict(record, reason="non-text-artifact"))
            continue
        if total + record["bytes"] > MAX_TEXT_BYTES:
            raise ValueError("原文总量超过 64 MiB，停止导出；不截断证据")
        try:
            # read_bytes + strict decode 保留 CRLF，回传可恢复相同字节/哈希。
            raw = path.read_bytes()
            if (
                len(raw) != record["bytes"]
                or hashlib.sha256(raw).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"证据文件大小/哈希与登记值不一致：{relative}")
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            omitted.append(dict(record, reason="not-utf8"))
            continue
        if "\x00" in content:
            omitted.append(dict(record, reason="contains-nul"))
            continue
        if compress:
            compressed = zlib.compress(raw, level=9)
            embedded.append(
                dict(
                    record,
                    encoding="zlib+base64",
                    compressed_bytes=len(compressed),
                    data=base64.b64encode(compressed).decode("ascii"),
                )
            )
            compressed_total += len(compressed)
        else:
            embedded.append(dict(record, encoding="utf-8", text=content))
        total += record["bytes"]
    required = {item["path"] for item in payload["evidence_files"]}
    if selected is not None:
        required &= selected
    if not required <= {item["path"] for item in embedded}:
        raise ValueError("必需摘要/FX/日志文件不能作为 UTF-8 原文回传，停止导出")
    transfer = {
        "embedded_files": len(embedded),
        "embedded_bytes": total,
        "omitted_files": omitted,
        "all_registered_artifacts_embedded": not omitted,
        "boundary": (
            (
                "包含评审范围原文供离线复核；未执行任何回传代码。未嵌入文件"
                "只保留 inventory 哈希，不宣称完整归档或重新运行通过。"
            )
            if profile == "review"
            else (
                "包含已登记 UTF-8 原文供离线复核；未执行任何回传代码。"
                "缺失二进制只保留哈希，不宣称完整二进制归档或重新运行通过。"
            )
        ),
        "profile": profile,
    }
    if compress:
        transfer.update(
            transport_encoding="zlib+base64-per-file",
            compressed_bytes=compressed_total,
        )
    payload.update(
        handoff_format_version=(
            REVIEW_FORMAT_VERSION
            if profile == "review"
            else (
                COMPRESSED_RAW_TEXT_FORMAT_VERSION
                if compress
                else RAW_TEXT_FORMAT_VERSION
            )
        ),
        handoff_profile=profile,
        raw_text_files=embedded,
        raw_text_transfer=transfer,
    )
    seal_payload(payload)


def bundle_raw_text(payload: dict[str, Any]) -> None:
    """跨文件压缩并复用相同字节；保留每个原路径、长度和哈希。"""
    chunks = []
    locations = {}
    offset = 0
    for item in payload["raw_text_files"]:
        raw = zlib.decompress(base64.b64decode(item.pop("data")))
        item.pop("compressed_bytes")
        key = (item["sha256"], item["bytes"])
        if key not in locations:
            locations[key] = offset
            chunks.append(raw)
            offset += len(raw)
        item.update(encoding="bundle-utf8", offset=locations[key])
    raw_bundle = b"".join(chunks)
    compressed = lzma.compress(raw_bundle, preset=6)
    payload["raw_text_bundle"] = {
        "encoding": "xz+base64",
        "bytes": len(raw_bundle),
        "sha256": hashlib.sha256(raw_bundle).hexdigest(),
        "compressed_bytes": len(compressed),
        "data": base64.b64encode(compressed).decode("ascii"),
    }
    payload["handoff_format_version"] = BUNDLE_FORMAT_VERSION
    payload["raw_text_transfer"].update(
        transport_encoding="xz+base64-bundle",
        compressed_bytes=len(compressed),
        unique_bytes=len(raw_bundle),
    )
    seal_payload(payload)


def case_audit(
    run_dir: Path, item: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        "tests_expected": execution.get("tests_expected"),
        "tests_expected_failures": execution.get("tests_expected_failures"),
        "tests_unexpected_successes": execution.get("tests_unexpected_successes"),
        "reference_valid": result.get("reference_valid"),
        "correctness_status": result.get("correctness", {}).get("status"),
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
    seal_payload(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 reference run 导出为适合终端复制的单个 JSON 文本证据包"
    )
    parser.add_argument(
        "--run-dir", required=True, type=Path, help="reference-<timestamp> 目录"
    )
    parser.add_argument("--output", type=Path, help="输出文件；省略时写到 stdout")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="输出单行 JSON；统一 runner 默认启用以降低网页传输体积",
    )
    parser.add_argument(
        "--include-raw-text",
        action="store_true",
        help="嵌入 FX、生成代码、IR、日志与 JSON 原文；不重新运行 GPU，不包含二进制",
    )
    parser.add_argument(
        "--compress-raw-text",
        action="store_true",
        help="将原文逐文件 zlib 压缩并 Base64 编码为 1.2 handoff；须与 --include-raw-text 同用",
    )
    parser.add_argument(
        "--profile",
        choices=("summary", "review", "archive"),
        help=(
            "summary=结构化摘要；review=摘要+FX/关键 case 正文（推荐网页回传）；"
            "archive=全部已登记文本"
        ),
    )
    parser.add_argument(
        "--bundle-raw-text",
        action="store_true",
        help="1.4 跨文件压缩与相同正文去重；须与 --profile review/archive 同用",
    )
    parser.add_argument(
        "--allow-derived-output",
        action="store_true",
        help="仅允许在 run 内新建保留文件 text-handoff.json；不覆盖原证据",
    )
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        help="将 handoff 拆成 manifest.json 和多个小 JSON；可与 --output 同用",
    )
    parser.add_argument(
        "--split-part-bytes",
        type=int,
        default=DEFAULT_SPLIT_PART_BYTES,
        help=f"每个分片承载的整包压缩字节数，默认 {DEFAULT_SPLIT_PART_BYTES}",
    )
    parser.add_argument(
        "--auto-split-over-bytes",
        type=int,
        help=(
            "仅当完整 handoff 超过指定字节数时生成 --split-output-dir；"
            f"省略则保持显式分片行为；统一入口使用 {DEFAULT_AUTO_SPLIT_THRESHOLD_BYTES}"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_dir = args.run_dir.resolve()
        if args.output is not None:
            output = args.output.resolve()
            reserved = run_dir / "text-handoff.json"
            if args.output.is_symlink():
                raise ValueError("输出不得是软链接")
            if (output == run_dir or output.is_relative_to(run_dir)) and not (
                args.allow_derived_output and output == reserved
            ):
                raise ValueError("文本 handoff 必须写在原始 run 目录外")
        if args.split_output_dir is not None:
            if args.split_output_dir.exists() or args.split_output_dir.is_symlink():
                raise ValueError(
                    f"分片输出目录必须不存在：{args.split_output_dir.resolve()}"
                )
            split_output_dir = args.split_output_dir.resolve()
            reserved_split = run_dir / "text-handoff-parts"
            if (
                split_output_dir == run_dir
                or split_output_dir.is_relative_to(run_dir)
            ) and not (
                args.allow_derived_output and split_output_dir == reserved_split
            ):
                raise ValueError("文本 handoff 分片必须写在原始 run 目录外")
        if args.auto_split_over_bytes is not None:
            if args.split_output_dir is None:
                raise ValueError("--auto-split-over-bytes 需要 --split-output-dir")
            if args.auto_split_over_bytes <= 0:
                raise ValueError("--auto-split-over-bytes 必须是正整数")
        if args.bundle_raw_text and args.profile not in {"review", "archive"}:
            raise ValueError("--bundle-raw-text 需要 --profile review/archive")
        payload = build_payload(run_dir)
        if args.profile is not None and (
            args.include_raw_text or args.compress_raw_text
        ):
            raise ValueError("--profile 不能与旧的原文选项同时使用")
        if args.compress_raw_text and not args.include_raw_text:
            raise ValueError("--compress-raw-text 必须与 --include-raw-text 同时使用")
        if args.profile in {"review", "archive"}:
            include_raw_text(
                payload,
                run_dir,
                compress=True,
                profile=args.profile,
            )
        elif args.profile == "summary":
            payload["handoff_profile"] = "summary"
            seal_payload(payload)
        elif args.include_raw_text:
            include_raw_text(payload, run_dir, compress=args.compress_raw_text)
        if args.bundle_raw_text:
            bundle_raw_text(payload)
        indent = None if args.compact else 2
        content = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=indent
        ) + "\n"
        wrote_output = False
        content_bytes = content.encode("utf-8")
        should_split = args.split_output_dir is not None and (
            args.auto_split_over_bytes is None
            or len(content_bytes) > args.auto_split_over_bytes
        )
        if should_split:
            manifest = write_split_payload(
                content_bytes,
                args.split_output_dir,
                payload_sha256=payload["payload_sha256"],
                part_bytes=args.split_part_bytes,
            )
            print("handoff_upload_mode=split")
            print(
                "handoff_upload_input="
                f"{(args.split_output_dir / 'manifest.json').resolve()}"
            )
            print(f"split_manifest={(args.split_output_dir / 'manifest.json').resolve()}")
            print(f"parts={manifest['part_count']}")
            print(f"source_bytes={manifest['source_bytes']}")
            print(f"transport_bytes={manifest['transport_bytes']}")
            print(f"source_sha256={manifest['source_sha256']}")
            wrote_output = True
        elif args.split_output_dir is not None:
            print("handoff_upload_mode=single-file")
            if args.output is not None:
                print(f"handoff_upload_input={args.output.resolve()}")
        if args.output is None and not wrote_output:
            sys.stdout.write(content)
        elif args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(content)
            print(f"text_handoff={args.output.resolve()}")
            print(f"payload_sha256={payload['payload_sha256']}")
            print(f"bytes={len(content_bytes)}")
        return 0
    except (OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
