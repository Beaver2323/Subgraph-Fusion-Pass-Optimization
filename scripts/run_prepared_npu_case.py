#!/usr/bin/env python3
"""逐例运行已准备批次的原生入口，保留零测试/跳过/异常，不自动签发功能门禁。"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/home/z50063656/tmp')
PYTORCH = Path('/home/z50063656/Pass/src/pytorch')


def installed_product_snapshot():
    """无需导入 torch，绑定本解释器安装包的相关 Python 文件。"""
    package = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/torch_npu'
    paths = ['_inductor/triton_experimental/device.py',
             '_inductor/triton_experimental/codegen/triton.py',
             '_inductor/triton_experimental/__init__.py',
             '_inductor/triton_experimental/config.py',
             '_inductor/triton_experimental/e8m0.py']
    return {name: {'path': str(package / name),
                   'sha256': hashlib.sha256((package / name).read_bytes()).hexdigest()}
            for name in paths if (package / name).is_file()}


def classify(return_code: int, log: str) -> tuple[str, int, int]:
    counts = re.findall(r'Ran (\d+) tests? in ', log)
    tests = sum(map(int, counts))
    skipped = sum(map(int, re.findall(r'skipped=(\d+)', log)))
    if return_code == 124:
        return 'timed-out', tests, skipped
    if return_code != 0 or re.search(r'^FAILED\b', log, re.M):
        return 'failed', tests, skipped
    if skipped or re.search(r'expected failures=|unexpected successes=', log):
        return 'skipped-or-xfail', tests, skipped
    if tests == 0:
        return 'no-tests', 0, skipped
    if not re.search(r'^OK\s*$', log, re.M):
        return 'missing-success-summary', tests, skipped
    return 'native-test-passed-awaiting-contract-review', tests, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--npu', type=int, required=True)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--adapter', action='store_true')
    parser.add_argument('--candidate-device', type=Path, help='T-087 独立源码 device.py，只在子进程加载，不修改安装态')
    parser.add_argument('--e8m0-candidate', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != WORK:
        parser.error(f'必须从 {WORK} 启动')
    if not re.fullmatch(r'T-\d{3}', args.task) or args.npu < 0 or args.timeout < 1:
        parser.error('任务编号、设备编号或超时非法')
    if args.candidate_device and (not args.adapter or args.task != 'T-087'
            or args.case != 'REF-respecialize-current-device-native'):
        parser.error('候选只允许用于 T-087 设备解析的原 adapter')
    if args.e8m0_candidate and (not args.adapter or args.task != 'T-096'):
        parser.error('E8M0 候选只允许用于 T-096 原 adapter')
    suffix = args.task.lower().replace('-', '')
    plan = json.loads((ROOT / f'upstream/{suffix}_reference_plan.yaml').read_text())
    selected = [c for c in plan['cases'] if c['case_id'] == args.case]
    if len(selected) != 1:
        parser.error('case 不属于该任务')
    source_name, method = selected[0]['source_test'].split('::', 1)
    source = (PYTORCH / source_name).resolve()
    if not source.is_relative_to(PYTORCH) or not source.is_file():
        parser.error('源文件不在冻结源码目录')
    expected_commit = plan['manifest']['pytorch_commit']
    actual_commit = subprocess.check_output(['git', '-C', str(PYTORCH), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_commit != expected_commit:
        parser.error('实际 PyTorch 源码 HEAD 与计划冻结 commit 不符')
    timestamp = datetime.now().astimezone().isoformat()
    case_root = ROOT / 'issues' / args.case
    adapter = case_root / 'npu_adapter.py'
    if args.adapter:
        blockers = list((case_root/'native_runs').glob('*/run_result.json'))
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        matching_blockers = [json.loads(path.read_text()) for path in blockers]
        matching_blockers = [record for record in matching_blockers
                             if record.get('source_sha256') == source_hash
                             and record.get('test_body_execution_proven') is False
                             and record.get('status') in {'failed', 'skipped-or-xfail', 'no-tests'}]
        if not adapter.is_file() or not matching_blockers:
            parser.error('最小适配前必须有本 case 原生记录及 case adapter')
    base = case_root / ('candidate_runs' if args.candidate_device or args.e8m0_candidate else ('adapter_runs' if args.adapter else 'native_runs'))
    base.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='native-' + datetime.now().strftime('%Y%m%dT%H%M%S') + '-', dir=base))
    cache = Path(tempfile.mkdtemp(prefix=suffix + '-native-', dir=WORK))
    command = [sys.executable, str(source), '-v', method]
    if args.adapter:
        command = [sys.executable, str(adapter), '--artifact-dir', str(cache/'adapter')]
        if args.candidate_device:
            command.extend(['--candidate-device', str(args.candidate_device.resolve(strict=True))])
        if args.e8m0_candidate:
            command.append('--e8m0-candidate')
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(args.npu), SET_NPU_DEVICE='0',
               TORCHINDUCTOR_NPU_BACKEND='triton_experimental', TORCH_DEVICE_BACKEND_AUTOLOAD='1',
               TORCHINDUCTOR_FORCE_DISABLE_CACHES='1', TORCHINDUCTOR_COMPILE_THREADS='1',
               TORCH_COMPILE_DEBUG='1', TORCH_COMPILE_DEBUG_DIR=str(cache/'debug'),
               TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor-cache'), TRITON_CACHE_DIR=str(cache/'triton-cache'))
    report = run / '复现报告.md'
    report.write_text(f'# 原生 NPU 测试入口复现\n\n> 时间：{timestamp}\n\n'
                      f'## 用例\n\n{args.task} / {args.case}\n\n'
                      f'## 命令\n\n```text\n{" ".join(command)}\n```\n\n'
                      '## 环境\n\nPass 环境；后端在导入前选择 triton_experimental；cwd 为临时工作目录。\n\n'
                      '## 结果\n\n执行中，未判定 PASS。\n\n## 下游处理\n\n等待日志与合同复核。\n', encoding='utf-8')
    installed_before = installed_product_snapshot()
    with (run/'stdout.log').open('w') as out, (run/'stderr.log').open('w') as err:
        try:
            code = subprocess.run(command, cwd=WORK, env=env, stdout=out, stderr=err, timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            code = 124
    log = (run/'stdout.log').read_text(errors='replace') + '\n' + (run/'stderr.log').read_text(errors='replace')
    status, tests, skips = classify(code, log)
    if args.adapter and status == 'native-test-passed-awaiting-contract-review':
        status = 'adapted-test-passed-awaiting-contract-review'
    result = dict(task_id=args.task, case_id=args.case, generated_at=timestamp,
                  status=status, tests_ran=tests, tests_skipped=skips, return_code=code,
                  mode='adapter' if args.adapter else 'native',
                  product_candidate=str(args.candidate_device) if args.candidate_device else None,
                  installed_product_before=installed_before,
                  installed_product_after=installed_product_snapshot(),
                  e8m0_candidate=args.e8m0_candidate,
                  test_body_execution_proven=tests > skips and '_FailedTest' not in log and code == 0,
                  command=command, working_directory=str(WORK), physical_npu=args.npu,
                  backend_requested='triton_experimental', backend_verified=False,
                  contract_review_required=True, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  raw_artifact_dir=str(cache), product_gate_bypassed=False, body_or_assertions_modified=False)
    (run/'run_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with report.open('a', encoding='utf-8') as f:
        f.write(f'\n## 执行完成\n\n状态 `{status}`；返回码 {code}；tests={tests}；skip={skips}。\n'
                '\n完整输出见同目录 stdout.log / stderr.log。原生入口结果不等于 NPU 合同/性能通过。\n')
    print(json.dumps(dict(result, report=str(report)), ensure_ascii=False), flush=True)
    return 0 if status.endswith('test-passed-awaiting-contract-review') else 1


if __name__ == '__main__':
    raise SystemExit(main())
