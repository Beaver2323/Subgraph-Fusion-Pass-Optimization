#!/usr/bin/env python3
"""按任务重新打包已有 GPU 证据；不运行测试、不覆盖原 run 或旧 handoff。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tempfile

from export_reference_text import (
    DEFAULT_SPLIT_PART_BYTES,
    build_payload,
    bundle_raw_text,
    include_raw_text,
    write_split_payload,
)
from import_reference_text import load_input, validate_payload


def reexport(task: str, result_root: Path | None = None) -> Path:
    if not re.fullmatch(r"T-\d{3}", task):
        raise ValueError("任务格式应为 T-098")
    root = (result_root or Path(f"/data/z50063656/tmp/t{task[2:]}-reference-results")).resolve()
    run = (root / "latest").resolve(strict=True)
    if not run.is_dir() or run.parent != root:
        raise ValueError("latest 必须指向该任务结果根目录下的实际 run")
    payload = build_payload(run)
    if payload["reference_summary"].get("run_id") != run.name:
        raise ValueError("run_id 与 latest 指向的目录不一致")
    include_raw_text(payload, run, compress=True, profile="review")
    bundle_raw_text(payload)
    expected = validate_payload(payload)
    content = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode()
    output = Path(tempfile.mkdtemp(prefix="handoff-reexport-", dir=root))
    parts = output / "text-handoff-parts"
    manifest = write_split_payload(
        content, parts, payload_sha256=payload["payload_sha256"],
        part_bytes=DEFAULT_SPLIT_PART_BYTES,
    )
    path = parts / "manifest.json"
    if validate_payload(load_input(path)) != expected:
        raise ValueError("重新打包往返校验不一致")
    print(f"handoff_validation=OK run_id={expected[0]} restorable_text_files={len(expected[1])} code_executed=false")
    print(f"parts={manifest['part_count']} source_bytes={len(content)} transport_bytes={manifest['transport_bytes']}")
    print(f"handoff_upload_input={path}")
    print(f"upload_destination=results/incoming/{task}/text-handoff-parts/")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--result-root", type=Path, help="非默认GPU结果目录；需要已有latest入口")
    args = parser.parse_args()
    try:
        reexport(args.task, args.result_root)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"错误：{error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
