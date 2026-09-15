#!/usr/bin/env python3
"""T-091/T-098/T-100 NPU功能预检及签门禁后的六臂性能入口。"""

from __future__ import annotations

import argparse
import fcntl
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "runners/t091_t100_performance_worker.py"
TASK_UNITS = {
    "T-091": ["stack-normalization"],
    "T-098": ["efficient-conv-bn"],
    "T-100": ["linear-binary-folding"],
}
ORDER = (("off", 1), ("on", 1), ("on", 2), ("off", 2), ("off", 3), ("on", 3))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_UNITS, required=True)
    parser.add_argument("--unit", action="append")
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
    units = args.unit or TASK_UNITS[args.task]
    unknown = sorted(set(units) - set(TASK_UNITS[args.task]))
    if unknown:
        parser.error(f"单元不属于{args.task}: {unknown}")
    if args.phase == "benchmark" and args.gate_root is None:
        parser.error("benchmark阶段必须提供--gate-root")
    if args.validate_only:
        compile(WORKER.read_text(encoding="utf-8"), str(WORKER), "exec")
        print(f"prepared_performance_validation=OK task={args.task} units={len(units)}")
        return 0
    lock = None
    if args.phase == "benchmark":
        lock = (work / "pass-tracker-npu-performance.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("已有 tracker 性能测量运行；拒绝并发污染 OFF/ON")
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    root = (
        args.output_root
        or work / f"{args.task.lower().replace('-', '')}-npu-results"
    ).resolve()
    run = root / f"{args.phase}-{timestamp}"
    run.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        PASS_TRACKER_WORK_DIR=str(work),
        TORCHINDUCTOR_NPU_BACKEND="triton_experimental",
    )
    arms = ORDER if args.phase == "benchmark" else (("off", 0), ("on", 0))
    for unit in units:
        for mode, round_number in arms:
            name = f"{mode}{round_number}" if round_number else mode
            command = [
                sys.executable,
                str(WORKER),
                "--unit",
                unit,
                "--mode",
                mode,
                "--device",
                args.device,
                "--phase",
                args.phase,
                "--output",
                str(run / unit / name),
                "--warmup",
                str(args.warmup),
                "--runs",
                str(args.runs),
            ]
            if args.phase == "benchmark":
                command.extend(
                    ["--gate", str(args.gate_root / args.task / f"{unit}.json")]
                )
            print(f"START task={args.task} unit={unit} arm={name}", flush=True)
            arm = run/unit/name
            arm.mkdir(parents=True,exist_ok=False)
            with (arm/'stdout.log').open('w') as stdout, (arm/'stderr.log').open('w') as stderr:
                result = subprocess.run(command,cwd=work,env=env,stdout=stdout,stderr=stderr)
            (arm/'execution.json').write_text(json.dumps(dict(command=command,return_code=result.returncode,
                cwd=str(work),backend=env['TORCHINDUCTOR_NPU_BACKEND'],
                physical_npu=env.get('ASCEND_RT_VISIBLE_DEVICES'),
                generated_at=datetime.now().astimezone().isoformat()),ensure_ascii=False,indent=2)+'\n')
            print(f"END task={args.task} unit={unit} arm={name} return_code={result.returncode} artifacts={arm}",flush=True)
            if result.returncode:
                print((arm/'stderr.log').read_text(errors='replace')[-5000:])
                return result.returncode
    print(f"task_run=passed artifacts={run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
