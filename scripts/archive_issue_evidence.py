#!/usr/bin/env python3
"""归档单次 issue 文本证据和哈希；不运行产物、不覆盖历史、不复制编译缓存。"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"result.json", "stdout.log", "stderr.log", "execution.json", "run_result.json", "adapted_test.py",
         "fx_graph_readable.py", "fx_graph_transformed.py", "ir_pre_fusion.txt", "ir_post_fusion.txt", "output_code.py"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", required=True)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    for value in (args.issue, args.label):
        if Path(value).name != value or value in (".", ".."):
            parser.error("issue/label 必须是单个目录名")
    base = args.run.resolve(strict=True)
    if not base.is_relative_to(Path("/home/z50063656/tmp")):
        parser.error("只接收本机 tmp 下的指定运行")
    destination = ROOT / "issues" / args.issue / "evidence" / args.label
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for path in sorted(base.rglob("*")):
        rel = path.relative_to(base)
        if not path.is_file() or any(p.endswith("-cache") or p == "trace" for p in rel.parts):
            continue
        if path.name not in NAMES and not path.name.startswith(("target-", "output_code_")):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(base):
            raise ValueError("拒绝运行目录外链接")
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records.append({"path":str(target.relative_to(ROOT)), "sha256":hashlib.sha256(target.read_bytes()).hexdigest(), "bytes":target.stat().st_size})
    inventory = {"generated_at":datetime.now().astimezone().isoformat(), "source_run":str(base), "files":records,
                 "generated_output_code_present":any(Path(r['path']).name == 'output_code.py' for r in records)}
    (destination / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+"\n")
    print(f"issue_archive={destination} text_files={len(records)}")


if __name__ == '__main__':
    main()
