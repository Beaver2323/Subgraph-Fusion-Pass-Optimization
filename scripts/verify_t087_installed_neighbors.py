#!/usr/bin/env python3
"""T-087 安装态近邻串行运行器；不部署修复、不修改已保存证据。"""
import argparse
from datetime import datetime
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
    p.add_argument('--devices', required=True, help='两张已确认可用的物理卡，例如 5,6')
    p.add_argument('--timeout', type=int, default=480)
    args = p.parse_args()
    if Path.cwd() != Path('/home/z50063656/tmp'):
        p.error('必须从 /home/z50063656/tmp 启动')
    cards = args.devices.split(',')
    if len(cards) != 2 or any(not c.isdecimal() for c in cards) or cards[0] == cards[1] or args.timeout <= 0:
        p.error('需要两个不同的非负物理卡号及正超时')
    base = ROOT/'issues/REF-respecialize-current-device-native/installed_neighbor_runs'
    base.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%dT%H%M%S-'), dir=base))
    print(f'installed_neighbor_run={run}', flush=True)
    records = []
    for case in ['ordinary-compile', 'cpp-wrapper-rejected', 'fx-wrapper-rejected', 'two-device-code-identical']:
        artifact = Path(tempfile.mkdtemp(prefix=f't087-{case}-', dir='/home/z50063656/tmp'))/'artifacts'
        command = [sys.executable, str(ROOT/'runners/t087_installed_neighbors.py'), '--case', case, '--artifact-dir', str(artifact)]
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=args.devices if case == 'two-device-code-identical' else cards[0],
                   SET_NPU_DEVICE='0', TORCHINDUCTOR_NPU_BACKEND='triton_experimental')
        print(f'START {case}', flush=True)
        with (run/f'{case}.stdout.log').open('w') as out, (run/f'{case}.stderr.log').open('w') as err:
            process = subprocess.Popen(command, cwd='/home/z50063656/tmp', env=env, stdout=out, stderr=err, start_new_session=True)
            try:
                rc = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                rc = 124
        row = dict(case=case, command=command, return_code=rc, artifact_dir=str(artifact), pid=process.pid,
                   generated_at=datetime.now().astimezone().isoformat(), visible_devices=env['ASCEND_RT_VISIBLE_DEVICES'])
        records.append(row)
        (run/'execution.json').write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
        print(f'END {case} rc={rc}', flush=True)
        if rc:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
