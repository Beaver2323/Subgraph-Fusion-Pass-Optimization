#!/usr/bin/env python3
"""校验本轮已上传 GPU 包并记录人工复核边界；不自动冻结单元或启动设备。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

from import_reference_text import load_input, validate_payload

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
TASKS = (87, 88, 89, 90, 91, 96, 98, 100, 102, 103, 104, 105, 106, 107, 112)


def select_input(incoming: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        source = explicit.resolve(strict=True)
        if not source.is_relative_to(incoming.resolve()):
            raise ValueError('显式输入必须位于本任务 incoming 目录')
        return source
    choices = [p for p in (incoming/'manifest.json', incoming/'text-handoff.json',
                           incoming/'text-handoff-parts/manifest.json') if p.is_file()]
    if len(choices) != 1:
        raise ValueError(f'{incoming}: 收件入口不唯一或不存在，请用 --input 显式选择本轮包')
    return choices[0]


def observe_case(case_id: str, files: dict) -> dict:
    match = re.fullmatch(r'REF-sfdp-pattern-(\d+)-native', case_id)
    expected = f'_sfdp_pattern_{match[1]}' if match else 'normalize_stack_default'
    if not match and case_id != 'REF-stack-axis-normalization-native':
        raise ValueError(f'未审核的观察合同：{case_id}')
    observations = []
    for name, raw in sorted(files.items()):
        if not (name.startswith(f'cases/{case_id}/debug/native_contract_observer/')
                and name.endswith('/contract_observation.json')):
            continue
        row = json.loads(raw)
        for key, value in {'capture_scope':'pattern-entry-apply-after-extra-check',
                           'handler_returned':True, 'test_body_modified':False,
                           'device_modified':False, 'assertions_modified':False,
                           'product_gate_bypassed':False,
                           'numerical_correctness_proven_by_observer':False}.items():
            if row.get(key) != value or type(row.get(key)) is not type(value):
                raise ValueError(f'{name}: 观察器边界不符 {key}')
        base = name.rsplit('/', 1)[0]
        paths = [name, base+'/fx_graph_readable.py', base+'/fx_graph_transformed.py']
        if any(p not in files for p in paths):
            raise ValueError(f'{name}: 缺少目标边界 FX 正文')
        changed = files[paths[1]] != files[paths[2]]
        if type(row.get('graph_changed')) is not bool or row['graph_changed'] != changed:
            raise ValueError(f'{name}: 图变化标记与正文不符')
        target = row['target']
        exact = (target == expected if not match else
                 re.fullmatch(re.escape(expected)+r'_(?:half_)?(?:bs1_)?(?:training|inference)', target) is not None)
        observations.append({'target':target, 'expected_target':exact, 'graph_changed':changed,
                             'files':[{'path':p,'sha256':hashlib.sha256(files[p]).hexdigest()} for p in paths]})
    found = [r for r in observations if r['expected_target'] and r['graph_changed']]
    status = ('gpu-contract-reviewed-awaiting-npu' if found else
              'native-passed-different-target-observed' if observations and not any(r['expected_target'] for r in observations) else
              'native-passed-target-attribution-pending')
    return {'status':status, 'expected_target':expected, 'exact_target_observations':len(found),
            'observed_targets':sorted({r['target'] for r in observations}), 'observations':observations,
            'semantic_boundary':('handler 重建节点；边界前后均为 dim=1，不宣称本 handler 首次把 axis 改为 dim。'
                                 if not match else '仅所观测编号/训练推理分支；数值与梯度范围仍以社区原断言为准。')}


def review(number: int, timestamp: str, source: Path | None = None, require_observer: bool = False) -> dict:
    task = f'T-{number:03}'
    incoming = ROOT / 'results/incoming' / task
    source = select_input(incoming, source)
    payload = load_input(source)
    run, files = validate_payload(payload)
    summary = payload['reference_summary']
    if not summary['suite_valid'] or not summary['suite_complete']:
        raise ValueError(f'{task}: 不是完整有效套件')
    plan = json.loads((ROOT/f'upstream/t{number:03}_reference_plan.yaml').read_text())
    # 本复核器只认证原生批次；后加的具名目标补测独立回传/评审，不能使旧原件失效。
    expected = {c['case_id'] for c in plan['cases'] if c['tracking_mode'] == 'direct'}
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
        if result['case']['source_test'] != case_plan[c['case_id']]['source_test']:
            raise ValueError(f'{task}: source_test 与计划不符')
        cases.append(dict(case_id=c['case_id'], acceptance_unit_id=c['acceptance_unit_id'],
                          source_test=case_plan[c['case_id']]['source_test'], command=execution['command'],
                          tests_ran=1, tests_skipped=0, reference_result_sha256=hashlib.sha256(files[path]).hexdigest()))
        if require_observer:
            if not any(str(x).endswith('/native_contract_observer.py') for x in execution['command']):
                raise ValueError(f'{task}: 非原生观察器入口')
            cases[-1]['target_review'] = observe_case(c['case_id'], files)
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
    if require_observer:
        accepted = sum(c['target_review']['status'] == 'gpu-contract-reviewed-awaiting-npu' for c in cases)
        status = 'gpu-contract-reviewed-awaiting-npu' if accepted == len(cases) else 'native-passed-target-review-mixed'
        boundary = '逐 case 记录真实编号；不同目标不能充当本编号命中；handler/FX 观察不额外证明数值或梯度。'
    return dict(schema_version='1.0', task_id=task, reviewed_at=timestamp, run_id=run,
                status=status, reference_backend='inductor-default', expected_pytorch_commit=COMMIT,
                input=str(source.relative_to(ROOT)), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                payload_sha256=payload['payload_sha256'], environment_fingerprint=env['fingerprint_sha256'],
                native_cases_passed=len(cases), tests_skipped=0, restored_text_files=len(files),
                acceptance_units=sorted({c['acceptance_unit_id'] for c in cases}), cases=cases,
                review_boundary=boundary, npu_execution_in_this_review=False,
                denominator_frozen_by_this_review=False, independent_unit_contribution=0 if number==112 else len({c['acceptance_unit_id'] for c in cases}),
                report=('report/unblocked_work_20260914.md' if number in (98,100) else
                        'report/gpu_observer_review_20260914.md' if require_observer else 'report/gpu_incoming_review_20260911.md'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write', action='store_true')
    p.add_argument('--task', type=int, choices=TASKS)
    p.add_argument('--input', type=Path)
    p.add_argument('--require-observer', action='store_true')
    p.add_argument('--archive-previous', action='store_true', help='显式保留被新run取代的旧复核记录；原收件仍由Git历史保存')
    p.add_argument('--previous-input-revision', help='旧收件的Git revision；与本轮run分离，不能把当前文件冒充旧原件')
    p.add_argument('--check-current', action='store_true', help='按已复核记录绑定的精确输入重验，不靠目录文件优先级猜测')
    args = p.parse_args()
    if args.check_current:
        if args.write or args.input or args.archive_previous:
            p.error('--check-current 不与写入/更换输入并用')
        for n in ((args.task,) if args.task is not None else TASKS):
            path = ROOT/f'results/current/T-{n:03}/gpu_reference_review.json'
            old = json.loads(path.read_text())
            observed = any('target_review' in c for c in old['cases'])
            current = review(n, old['reviewed_at'], ROOT/old['input'], observed)
            for key in ('input_sha256','payload_sha256','run_id','cases','status','environment_fingerprint'):
                if current[key] != old[key]:
                    raise ValueError(f'{path}: 已复核证据漂移 {key}')
            print(current['task_id'], current['status'], current['native_cases_passed'])
        return
    if args.input and args.task is None:
        p.error('--input 必须与单个 --task 一起使用')
    if args.archive_previous and not args.previous_input_revision:
        p.error('归档旧复核必须声明 --previous-input-revision')
    timestamp = datetime.now().astimezone().isoformat()
    for n in ((args.task,) if args.task is not None else TASKS):
        data = review(n, timestamp, args.input, args.require_observer)
        if args.write:
            dest = ROOT/'results/current'/data['task_id']/'gpu_reference_review.json'
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                old = json.loads(dest.read_text())
                if old.get('payload_sha256') != data['payload_sha256']:
                    if not args.archive_previous:
                        raise ValueError(f'已有不同 run 的复核结果，须显式归档：{dest}')
                    if not re.fullmatch(r'[0-9a-f]{7,40}', args.previous_input_revision):
                        raise ValueError('旧收件 revision 必须是明确的 Git commit')
                    old_input = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                        f'{args.previous_input_revision}:{old["input"]}'])
                    if hashlib.sha256(old_input).hexdigest() != old['input_sha256']:
                        raise ValueError('声明的 Git revision 不包含匹配的旧收件原件')
                    history = ROOT/'results/history'/data['task_id']/old['run_id']/'gpu_reference_review.json'
                    history.parent.mkdir(parents=True, exist_ok=True)
                    if history.exists() and history.read_bytes() != dest.read_bytes():
                        raise ValueError(f'拒绝覆盖不同历史记录：{history}')
                    history.write_bytes(dest.read_bytes())
                    data['previous_review'] = str(history.relative_to(ROOT))
                    data['previous_input_git_revision'] = args.previous_input_revision
            dest.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
        print(data['task_id'], data['status'], data['native_cases_passed'], flush=True)


if __name__ == '__main__':
    main()
