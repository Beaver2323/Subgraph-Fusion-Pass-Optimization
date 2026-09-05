#!/usr/bin/env python3
"""发布同一次 GPU 运行的 latest 与文本入口；失败也发布明确状态，不回退旧成功。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import uuid


def atomic_link(link: Path, target: str) -> None:
    if link.exists() and not link.is_symlink():
        raise ValueError(f"拒绝覆盖非软链接入口：{link}")
    temporary = link.with_name(f".{link.name}.{uuid.uuid4().hex}")
    try:
        temporary.symlink_to(target)
        os.replace(temporary, link)
    finally:
        temporary.unlink(missing_ok=True)


def publish(
    result_root: Path,
    run_dir: Path | None,
    runner_status: int,
    export_status: int,
) -> Path:
    result_root = result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    for name in ("latest", "latest-text-handoff.json"):
        path = result_root / name
        if path.exists() and not path.is_symlink():
            raise ValueError(f"拒绝覆盖非软链接入口：{path}")
    if run_dir is None:
        run_dir = Path(tempfile.mkdtemp(prefix="failed-launch-", dir=result_root))
        runner_status = runner_status or 1
        export_status = export_status or 1
    run_dir = run_dir.resolve()
    if run_dir.parent != result_root or not run_dir.is_dir():
        raise ValueError("run 必须是 result-root 的直接子目录")
    handoff = run_dir / "text-handoff.json"
    if handoff.is_symlink():
        raise ValueError("handoff 不得是软链接")
    if export_status == 0:
        payload = json.loads(handoff.read_text(encoding="utf-8"))
        if payload.get("reference_summary", {}).get("run_id") != run_dir.name:
            raise ValueError("本轮 handoff run_id 与运行目录不一致")
    else:
        # 不覆盖半写入文件，保留供诊断；旧 timestamp 目录也保持不变。
        if handoff.is_symlink():
            raise ValueError("handoff 不得是软链接")
        if handoff.exists():
            handoff.rename(run_dir / f"text-handoff.incomplete-{uuid.uuid4().hex}.json")
        payload = {
            "handoff_status": "export-failed",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_id": run_dir.name,
            "runner_status": runner_status,
            "export_status": export_status,
            "reference_valid": False,
            "reason": "本轮未得到完整文本证据；请回传本文件及启动日志，不得使用上轮成功结果。",
        }
        handoff.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 文本入口固定通过 latest 寻址，之后只需要原子替换一个运行指针。
    atomic_link(result_root / "latest-text-handoff.json", "latest/text-handoff.json")
    atomic_link(result_root / "latest", run_dir.name)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--runner-status", type=int, required=True)
    parser.add_argument("--export-status", type=int, required=True)
    args = parser.parse_args()
    run_dir = publish(args.result_root, args.run_dir, args.runner_status, args.export_status)
    print(f"latest_run={args.result_root / 'latest'}")
    print(f"latest_text_handoff={args.result_root / 'latest-text-handoff.json'}")
    print(f"published_run_dir={run_dir}")


if __name__ == "__main__":
    main()
