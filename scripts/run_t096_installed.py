#!/usr/bin/env python3
"""T-096 独立进程功能近邻或带门禁的六臂性能运行。"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['functional', 'benchmark'], required=True)
    p.add_argument('--npu', type=int, required=True)
    p.add_argument('--gate', type=Path)
    p.add_argument('--timeout', type=int, default=480)
    args = p.parse_args()
    if Path.cwd() != Path('/home/z50063656/tmp') or args.npu < 0 or args.timeout < 1:
        p.error('必须从 tmp 启动，卡号非负，超时为正')
    if args.phase == 'benchmark' and not args.gate:
        p.error('性能必须提供已签门禁')
    base = Path('/home/z50063656/tmp/t096-npu-results')
    base.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=args.phase+'-'+datetime.now().strftime('%Y%m%dT%H%M%S-'), dir=base))
    print(f't096_run={run}', flush=True)
    arms = [('off', 'ordinary', 'off'), ('on', 'ordinary', 'on')]
    arms += [(c,c,'on') for c in ['fp16-guard', 'bf16-guard', 'cpu-guard', 'strided', 'exponent-sweep']]
    if args.phase == 'benchmark':
        arms = [(a,'ordinary',a.rstrip('123')) for a in ('off1','on1','on2','off2','off3','on3')]
    lock = open('/home/z50063656/tmp/pass-tracker-npu-performance.lock', 'a')
    try:
        if args.phase == 'benchmark':
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for arm, case, mode in arms:
            folder = run/arm
            folder.mkdir()
            command = [sys.executable, str(ROOT/'runners/t096_installed_worker.py'), '--case', case,
                       '--mode', mode, '--phase', args.phase, '--artifact-dir', str(folder/'artifacts')]
            if args.gate:
                command += ['--gate', str(args.gate.resolve())]
            env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(args.npu), SET_NPU_DEVICE='0',
                       TORCHINDUCTOR_NPU_BACKEND='triton_experimental')
            print(f'START {arm}', flush=True)
            with (folder/'stdout.log').open('w') as out, (folder/'stderr.log').open('w') as err:
                proc = subprocess.Popen(command, cwd='/home/z50063656/tmp', env=env, stdout=out, stderr=err, start_new_session=True)
                try:
                    rc = proc.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    rc = 124
            (folder/'execution.json').write_text(json.dumps(dict(generated_at=datetime.now().astimezone().isoformat(),
                command=command, pid=proc.pid, return_code=rc, timed_out=rc==124, physical_npu=args.npu),
                ensure_ascii=False, indent=2)+'\n')
            print(f'END {arm} rc={rc}', flush=True)
            if rc:
                return 1
    finally:
        lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
