#!/usr/bin/env python3
"""T-102～T-107 attention功能预检及签门禁后的六臂性能入口。"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from run_prepared_performance import run_arm


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


def runnable_patterns(task, requested=None):
    """去重别名只保留历史记录，不构造独立OFF/ON。"""
    candidates = TASK_PATTERNS[task]
    patterns = candidates if requested is None else requested
    if set(patterns) - set(candidates):
        raise ValueError(f"pattern不属于{task}")
    aliases = {17: ("T-104", 15)}
    if requested is not None and set(patterns) & aliases.keys():
        raise ValueError("17为T-104 pattern15的去重别名，不支持独立功能/性能臂")
    return [number for number in patterns if number not in aliases]


def frozen_benchmark_worker(gate_path, root=ROOT):
    """执行人工gate绑定的本仓库快照，公共worker后续编辑不污染已签输入。"""
    gate = json.loads(gate_path.read_text())
    selected = None
    for key, filename in (
        ('worker_snapshot', WORKER.name),
        ('target_control_snapshot', 'target_entry_control.py'),
        ('observer_snapshot', 'native_contract_observer.py'),
    ):
        item = gate[key]
        path = Path(item['path'])
        if (not path.is_absolute() or path.is_symlink()
                or not path.resolve().is_relative_to((root/'issues').resolve())
                or path.name != filename or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']):
            raise ValueError(f'计时快照路径或哈希无效：{key}')
        if selected is None:
            selected = path.resolve()
        elif path.resolve().parent != selected.parent:
            raise ValueError('三份执行快照必须位于同一归档目录')
    return selected


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
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    work = Path(os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")).resolve()
    if Path.cwd().resolve() != work:
        parser.error(f"必须先 cd {work}")
    try:
        patterns = runnable_patterns(args.task, args.pattern)
    except ValueError as error:
        parser.error(str(error))
    if args.phase == "benchmark" and args.gate_root is None:
        parser.error("benchmark阶段必须提供--gate-root")
    if min(args.timeout,args.runs,args.warmup)<1:
        parser.error('超时、预热和采样次数必须为正')
    if args.validate_only:
        compile(WORKER.read_text(encoding="utf-8"), str(WORKER), "exec")
        print(f"prepared_performance_validation=OK task={args.task} patterns={len(patterns)}")
        return 0

    lock = None
    if args.phase == 'benchmark':
        lock = (work/'pass-tracker-npu-performance.lock').open('a+')
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('已有tracker性能任务持锁；不与其并发计时')

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
        selected_worker = (frozen_benchmark_worker(args.gate_root/args.task/f'pattern-{pattern}.json')
                           if args.phase == 'benchmark' else WORKER)
        for mode, round_number in arms:
            name = f"{mode}{round_number}" if round_number else mode
            command = [
                sys.executable,
                str(selected_worker),
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
            arm_dir = run/f'pattern-{pattern}'/name
            arm_dir.mkdir(parents=True,exist_ok=False)
            with (arm_dir/'stdout.log').open('w') as out, (arm_dir/'stderr.log').open('w') as err:
                try:
                    code = run_arm(command,work,out,err,timeout=args.timeout,env=env)
                except subprocess.TimeoutExpired:
                    code = 124
            (arm_dir/'execution.json').write_text(json.dumps(dict(command=command,return_code=code,
                generated_at=datetime.now().astimezone().isoformat(),
                cooperative_performance_lock=lock is not None),ensure_ascii=False,indent=2)+'\n')
            print(f'END task={args.task} pattern={pattern} arm={name} return_code={code}',flush=True)
            if code:
                print(f'task_run=failed artifacts={arm_dir}',flush=True)
                return 1
    print(f"task_run=passed artifacts={run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
