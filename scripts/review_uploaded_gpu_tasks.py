#!/usr/bin/env python3
"""校验本轮已上传 GPU 包并记录人工复核边界；不自动冻结单元或启动设备。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from import_reference_text import load_input, validate_payload

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
TASKS = (87, 88, 89, 90, 91, 96, 102, 103, 104, 105, 106, 107, 112)


def review(number: int, timestamp: str) -> dict:
    task = f'T-{number:03}'
    incoming = ROOT / 'results/incoming' / task
    source = incoming / 'manifest.json'
    if not source.is_file():
        source = incoming / 'text-handoff.json'
    payload = load_input(source)
    run, files = validate_payload(payload)
    summary = payload['reference_summary']
    if not summary['suite_valid'] or not summary['suite_complete']:
        raise ValueError(f'{task}: 不是完整有效套件')
    plan = json.loads((ROOT/f'upstream/t{number:03}_reference_plan.yaml').read_text())
    expected = {c['case_id'] for c in plan['cases']}
    case_plan = {c['case_id']: c for c in plan['cases']}
    actual = [c['case_id'] for c in payload['case_audit']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f'{task}: case 集合不符')
    env = payload['environment']
    src = env['source']
    runtime = env['runtime']
    if (src['actual_commit'] != COMMIT or src['expected_commit'] != COMMIT
            or runtime['torch_git_version'] != COMMIT or src['working_tree_state'] != 'clean'
            or runtime['cuda_available'] is not True):
        raise ValueError(f'{task}: GPU 冻结环境不符')
    cases = []
    for c in payload['case_audit']:
        path = f"cases/{c['case_id']}/reference_result.json"
        result = json.loads(files[path])
        execution = result['execution']
        if (execution['tests_ran'] != 1 or execution['tests_expected'] != 1
                or any(execution[k] != 0 for k in ('return_code', 'tests_skipped', 'tests_expected_failures', 'tests_unexpected_successes'))
                or c['reference_valid'] is not True):
            raise ValueError(f'{task}/{c["case_id"]}: 无法确认原生成功')
        cases.append(dict(case_id=c['case_id'], acceptance_unit_id=c['acceptance_unit_id'],
                          source_test=case_plan[c['case_id']]['source_test'], command=execution['command'],
                          tests_ran=1, tests_skipped=0, reference_result_sha256=hashlib.sha256(files[path]).hexdigest()))
    if number == 112:
        status = 'gpu-validated-duplicate-contract'
        boundary = '与 T-084 同一开关、调用链、社区测例及线性 RS 合同；保留收件，不增加独立能力分母。仅 world_size=1。'
    elif number == 91 or 102 <= number <= 107:
        status = 'native-passed-target-attribution-pending'
        boundary = ('原生仅数值断言，缺 pre-grad normalize_stack_default 精确命中/边界图。' if number == 91 else
                    '通用 fuse_attention/晚于融合的 FX 不足以逐编号归因；12 training 有原生编号断言；dropout 数值范围按源码参数确认。')
    else:
        status = 'gpu-contract-reviewed-awaiting-npu'
        boundary = ('设备解析例只验证单设备运行时代码，不证明跨卡复用/数值对照。' if number == 87 else
                    'GPU 合同通过，不代表 NPU 功能、产品合入或性能通过。')
    return dict(schema_version='1.0', task_id=task, reviewed_at=timestamp, run_id=run,
                status=status, reference_backend='inductor-default', expected_pytorch_commit=COMMIT,
                input=str(source.relative_to(ROOT)), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                payload_sha256=payload['payload_sha256'], environment_fingerprint=env['fingerprint_sha256'],
                native_cases_passed=len(cases), tests_skipped=0, restored_text_files=len(files),
                acceptance_units=sorted({c['acceptance_unit_id'] for c in cases}), cases=cases,
                review_boundary=boundary, npu_execution_in_this_review=False,
                denominator_frozen_by_this_review=False, independent_unit_contribution=0 if number==112 else len({c['acceptance_unit_id'] for c in cases}),
                report='report/gpu_incoming_review_20260911.md')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write', action='store_true')
    args = p.parse_args()
    timestamp = datetime.now().astimezone().isoformat()
    for n in TASKS:
        data = review(n, timestamp)
        if args.write:
            dest = ROOT/'results/current'/data['task_id']/'gpu_reference_review.json'
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                old = json.loads(dest.read_text())
                if old.get('payload_sha256') != data['payload_sha256']:
                    raise ValueError(f'已有不同 run 的复核结果，须显式归档：{dest}')
            dest.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
        print(data['task_id'], data['status'], data['native_cases_passed'], flush=True)


if __name__ == '__main__':
    main()
