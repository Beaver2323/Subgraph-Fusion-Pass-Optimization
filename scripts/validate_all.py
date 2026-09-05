#!/usr/bin/env python3
"""统一零设备提交检查；严格模式另外阻止历史证据待复核时宣称再认证完成。"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from audit_history import ROOT, build, parse_gpu_runs, write_new
from publish_reference_latest import atomic_link


def history_exit(status, strict):
    if status not in {"passed", "pending", "failed"}:
        raise ValueError("未知历史复核状态")
    if status == "failed":
        return 2
    if status == "pending" and strict:
        return 3
    return 0


def commands(root, pytorch_root):
    python = sys.executable
    yield "unit_tests", [python, "-m", "unittest", "discover", "-s", str(root / "tests")]
    for script in ("validate_tracker_data.py", "validate_comparison_data.py", "validate_prepared_tasks.py"):
        yield script, [python, str(root / "scripts" / script)]
    yield "task_backlog", [python, str(root / "scripts/build_task_backlog.py"), "--check"]
    for task in ("", "t077_", "t078_", "t079_", "t080_"):
        yield f"{task}reference_plan", ["bash", str(root / f"scripts/run_{task}reference_all.sh"), "--pytorch-root", str(pytorch_root), "--validate-only"]
    yield "whitespace", ["git", "-C", str(root), "diff", "--check"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-root", type=Path, default=Path("/home/z50063656/Pass/src/pytorch"))
    parser.add_argument("--require-history-complete", action="store_true", help="历史再认证门禁：待复核返回 3，不得当成 PASS")
    parser.add_argument("--write-audit", action="store_true", help="追加独立审计文件并更新 results/audits/latest.json 入口")
    parser.add_argument("--gpu-run", action="append", default=[], help="只读复核已复制到本机的原始 GPU run，TASK=/path")
    args = parser.parse_args()
    work_dir = Path("/home/z50063656/tmp").resolve()
    if Path.cwd().resolve() != work_dir:
        print(f"错误：先 cd {work_dir} 再运行统一检查。", file=sys.stderr)
        return 2
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PASS_TRACKER_WORK_DIR=str(work_dir), PYTHON=sys.executable)
    try:
        for directory in ("scripts", "runners", "tests"):
            for path in sorted((ROOT / directory).glob("*.py")):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for path in sorted((ROOT / directory).glob("*.sh")):
                subprocess.run(["bash", "-n", str(path)], check=True, cwd=work_dir, env=env)
        for directory in ("schemas", "upstream"):
            for path in sorted((ROOT / directory).iterdir()):
                if path.suffix == ".json" or (path.suffix == ".yaml" and path.read_text().lstrip().startswith("{")):
                    json.loads(path.read_text())
        for label, command in commands(ROOT, args.pytorch_root):
            print(f"START {label}", flush=True)
            subprocess.run(command, check=True, cwd=work_dir, env=env)
        payload = build(gpu_runs=parse_gpu_runs(args.gpu_run))
        if args.write_audit:
            audit_root = ROOT / "results/audits"
            audit_root.mkdir(parents=True, exist_ok=True)
            latest = audit_root / "latest.json"
            if latest.exists() and not latest.is_symlink():
                raise ValueError("拒绝覆盖非软链接 audit latest")
            prefix = datetime.now().astimezone().strftime("audit-%Y%m%dT%H%M%S%z-")
            directory = Path(tempfile.mkdtemp(prefix=prefix, dir=audit_root))
            write_new(directory / "audit.json", payload)
            atomic_link(latest, f"{directory.name}/audit.json")
            print(f"history_audit={latest}")
        print("tooling_gate=passed")
        print(f"history_recertification={payload['status']} counts={json.dumps(payload['status_counts'], sort_keys=True)}")
        status = history_exit(payload["status"], args.require_history_complete)
        print(f"gate_scope={'history-recertification' if args.require_history_complete else 'tooling-only'} exit_code={status}")
        return status
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, subprocess.CalledProcessError) as error:
        print(f"统一门禁失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
