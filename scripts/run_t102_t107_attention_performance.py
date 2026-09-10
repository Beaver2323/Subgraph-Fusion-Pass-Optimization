#!/usr/bin/env python3
"""T-102～T-107 attention功能预检及签门禁后的六臂性能入口。"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "runners/t102_t107_attention_performance_worker.py"
TASK_PATTERNS = {
    "T-102": [1, 2, 3, 4, 5],
    "T-103": [6, 7, 8, 9, 10],
    "T-104": [11, 12, 13, 14, 15],
    "T-105": [16, 17, 18, 19, 20],
    "T-106": [21, 22, 23, 24],
    "T-107": [28, 29, 30],
}
ORDER = (("off", 1), ("on", 1), ("on", 2), ("off", 2), ("off", 3), ("on", 3))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_PATTERNS, required=True)
    parser.add_argument("--pattern", type=int, action="append")
    parser.add_argument("--device", choices=("cuda", "npu"), default="npu")
    parser.add_argument("--phase", choices=("functional", "benchmark"), default="functional")
    parser.add_argument("--gate-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    work = Path(os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")).resolve()
    if Path.cwd().resolve() != work:
        parser.error(f"必须先 cd {work}")
    patterns = args.pattern or TASK_PATTERNS[args.task]
    unknown = sorted(set(patterns) - set(TASK_PATTERNS[args.task]))
    if unknown:
        parser.error(f"pattern不属于{args.task}: {unknown}")
    if args.phase == "benchmark" and args.gate_root is None:
        parser.error("benchmark阶段必须提供--gate-root")
    if args.validate_only:
        compile(WORKER.read_text(encoding="utf-8"), str(WORKER), "exec")
        print(f"prepared_performance_validation=OK task={args.task} patterns={len(patterns)}")
        return 0

    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    root = (args.output_root or work / f"{args.task.lower().replace('-', '')}-npu-results").resolve()
    run = root / f"{args.phase}-{timestamp}"
    run.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        PASS_TRACKER_WORK_DIR=str(work),
        TORCHINDUCTOR_NPU_BACKEND="triton_experimental",
    )
    arms = ORDER if args.phase == "benchmark" else (("off", 0), ("on", 0))
    for pattern in patterns:
        for mode, round_number in arms:
            name = f"{mode}{round_number}" if round_number else mode
            command = [
                sys.executable,
                str(WORKER),
                "--pattern", str(pattern),
                "--mode", mode,
                "--device", args.device,
                "--phase", args.phase,
                "--output", str(run / f"pattern-{pattern}" / name),
                "--warmup", str(args.warmup),
                "--runs", str(args.runs),
            ]
            if args.phase == "benchmark":
                command.extend(
                    ["--gate", str(args.gate_root / args.task / f"pattern-{pattern}.json")]
                )
            print(f"START task={args.task} pattern={pattern} arm={name}", flush=True)
            subprocess.run(command, cwd=work, env=env, check=True)
    print(f"task_run=passed artifacts={run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
