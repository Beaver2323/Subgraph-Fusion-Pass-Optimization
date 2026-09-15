#!/usr/bin/env python3
"""功能测试共享性能互斥锁、独占指定设备；两臂串行并保留完整日志。"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from run_prepared_performance import run_arm

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/home/z50063656/tmp')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npu', type=int, required=True)
    p.add_argument('--wait-timeout', type=int, default=7200)
    p.add_argument('--installed', action='store_true', help='仅运行不加载候选的安装态边界')
    args = p.parse_args()
    if Path.cwd().resolve() != WORK or args.npu < 0 or args.wait_timeout < 0:
        p.error('从规定tmp执行，设备和等待时间不能为负')
    output = Path(tempfile.mkdtemp(prefix='attention-slice-boundary-', dir=WORK))
    state = dict(generated_at=datetime.now().astimezone().isoformat(), status='waiting-for-device-lock',
                 physical_npu=args.npu, device_execution=False, arms=[])
    def save():
        (output/'run_result.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    save()
    print(f'boundary_output={output}',flush=True)
    start = time.monotonic()
    with (WORK/'pass-tracker-npu-performance.lock').open('a') as lock, \
            (WORK/f'pass-tracker-npu-{args.npu}.lock').open('a') as device_lock:
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(device_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                if time.monotonic()-start >= args.wait_timeout:
                    state['status'] = 'not-run-lock-timeout'
                    save()
                    return 2
                time.sleep(1)
        state['status'] = 'running-not-a-verdict'
        save()
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(args.npu), SET_NPU_DEVICE='0',
                   TORCHINDUCTOR_NPU_BACKEND='triton_experimental')
        for arm in (('installed',) if args.installed else ('baseline', 'candidate')):
            command = [sys.executable,str(ROOT/'runners/attention_select_slice_boundary.py'),
                       '--arm',arm,'--output',str(output/arm)]
            state['device_execution'] = True
            print(f'boundary_arm={arm} START',flush=True)
            with (output/f'{arm}.stdout.log').open('w') as out, (output/f'{arm}.stderr.log').open('w') as err:
                try:
                    rc = run_arm(command,WORK,out,err,timeout=1800,env=env)
                except subprocess.TimeoutExpired:
                    rc = 124
            state['arms'].append(dict(arm=arm,command=command,return_code=rc))
            save()
            print(f'boundary_arm={arm} return_code={rc}',flush=True)
        state['status'] = 'finished-awaiting-evidence-review'
        save()
    return 0 if state['arms'][-1]['return_code']==0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
