#!/usr/bin/env python3
"""T-112 双卡NCCL/HCCL功能预检及签门禁后的六臂性能入口。"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "runners/t112_dedup_reduce_scatter_worker.py"
ORDER = (("off", 1), ("on", 1), ("on", 2), ("off", 2), ("off", 3), ("on", 3))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "npu"), default="npu")
    parser.add_argument("--phase", choices=("functional", "benchmark"), default="functional")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    work = Path(os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")).resolve()
    if Path.cwd().resolve() != work:
        parser.error(f"必须先 cd {work}")
    if args.phase == "benchmark" and args.gate is None:
        parser.error("benchmark阶段必须提供--gate")
    if args.validate_only:
        compile(WORKER.read_text(encoding="utf-8"), str(WORKER), "exec")
        print("prepared_performance_validation=OK task=T-112 world_size=2")
        return 0

    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    root = (args.output_root or work / "t112-npu-results").resolve()
    run = root / f"{args.phase}-{timestamp}"
    run.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        PASS_TRACKER_WORK_DIR=str(work),
        TORCHINDUCTOR_NPU_BACKEND="triton_experimental",
    )
    arms = ORDER if args.phase == "benchmark" else (("off", 0), ("on", 0))
    for mode, round_number in arms:
        name = f"{mode}{round_number}" if round_number else mode
        command = [
            sys.executable,
            "-m", "torch.distributed.run",
            "--standalone",
            "--nproc-per-node", "2",
            str(WORKER),
            "--mode", mode,
            "--device", args.device,
            "--phase", args.phase,
            "--output", str(run / name),
            "--warmup", str(args.warmup),
            "--runs", str(args.runs),
            "--rows", str(args.rows),
            "--hidden", str(args.hidden),
        ]
        if args.phase == "benchmark":
            command.extend(["--gate", str(args.gate)])
        print(f"START task=T-112 arm={name} world_size=2", flush=True)
        subprocess.run(command, cwd=work, env=env, check=True)
    print(f"task_run=passed artifacts={run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
