#!/usr/bin/env python3
"""从既有 T-076/T-077 GPU run 导出日志补证；不重跑，不复制 FX/cache，不改原件。"""

import argparse
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import zlib

from export_reference_text import checked_file, seal_payload, write_split_payload


RUNS = {"T-076": "reference-20260901T180826+0800", "T-077": "reference-20260902T125636+0800"}
LIMIT = 16 * 1024 * 1024


def build_logs(task, run):
    summary = json.loads(checked_file(run, "reference_summary.json").read_text())
    if summary["run_id"] != RUNS[task]:
        raise ValueError("历史补证必须使用原始 run；新测试不得冒充历史运行")
    files, empty = [], []
    for case in summary["cases"]:
        prefix = f"cases/{case['case_id']}/"
        inventory = json.loads(checked_file(run, prefix + "artifact_inventory.json").read_text())
        for name in ("stdout.log", "stderr.log"):
            matches = [item for item in inventory if item["path"] == name]
            if len(matches) != 1:
                raise ValueError(f"日志 inventory 缺失或重复：{prefix}{name}")
            record = matches[0]
            data = checked_file(run, prefix + name).read_bytes()
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"日志与历史 inventory 不一致：{prefix}{name}")
            if not data:
                empty.append(prefix + name)
                continue
            encoded = base64.b64encode(zlib.compress(data, 9)).decode("ascii")
            files.append({"path": prefix + name, "bytes": len(data), "sha256": record["sha256"],
                          "encoding": "zlib+base64", "data": [encoded[i:i + 120] for i in range(0, len(encoded), 120)]})
    payload = {"history_logs_format": "1.0", "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "task_id": task, "run_id": summary["run_id"], "files": files, "empty_logs": empty}
    seal_payload(payload)
    validate_logs(payload, task, summary["run_id"])
    return payload


def validate_logs(payload, task, run_id):
    copied = dict(payload)
    seal_payload(copied)
    if copied["payload_sha256"] != payload.get("payload_sha256"):
        raise ValueError("日志补证整包 SHA256 不一致")
    if payload.get("history_logs_format") != "1.0" or payload.get("task_id") != task or payload.get("run_id") != run_id:
        raise ValueError("日志补证格式或 task/run 不一致")
    result = {}
    total = 0
    for item in payload["files"]:
        path = Path(item["path"])
        if len(path.parts) != 3 or path.parts[0] != "cases" or path.parts[1] in (".", "..") or path.name not in ("stdout.log", "stderr.log"):
            raise ValueError("日志补证只允许 cases/CASE/stdout.log 或 stderr.log")
        if str(path) in result:
            raise ValueError("日志补证路径重复")
        size = item["bytes"]
        if type(size) is not int or not 0 < size <= LIMIT or item.get("encoding") != "zlib+base64":
            raise ValueError("日志大小或编码无效")
        decoder = zlib.decompressobj()
        data = decoder.decompress(base64.b64decode("".join(item["data"]), validate=True), size + 1)
        total += len(data)
        if total > LIMIT or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or len(data) != size or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("日志解压长度、数据流或 SHA256 无效")
        data.decode("utf-8")
        result[str(path)] = data
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(RUNS))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        run = args.run_dir or Path(f"/data/z50063656/tmp/{args.task.lower().replace('-', '')}-reference-results/{RUNS[args.task]}")
        output = args.output.resolve()
        if output.is_relative_to(run.resolve()):
            raise ValueError("补证输出必须在原始 run 之外")
        payload = build_logs(args.task, run)
        data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(data)
        if len(data) > 64 * 1024:
            split_dir = output.with_suffix("")
            write_split_payload(data, split_dir, payload_sha256=payload["payload_sha256"], part_bytes=32 * 1024)
            print(f"history_logs_upload={split_dir}/manifest.json 和全部 part 文件")
        else:
            print(f"history_logs_upload={output}")
        print(f"nonempty_logs={len(payload['files'])} empty_logs_not_transferred={len(payload['empty_logs'])} bytes={len(data)}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, zlib.error) as error:
        print(f"历史日志导出失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
