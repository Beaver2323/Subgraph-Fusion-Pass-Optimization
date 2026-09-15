#!/usr/bin/env python3
"""人工逐图审查后归档 attention 功能臂并签性能门禁；不计时、不部署产品。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

from inspect_npu_codegen import inspect

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/home/z50063656/tmp')
WORKER = ROOT/'runners/t102_t107_attention_performance_worker.py'
spec = importlib.util.spec_from_file_location('attention_gate_worker', WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
NAMES = {'result.json', 'execution.json', 'stdout.log', 'stderr.log',
         'fx_graph_readable.py', 'fx_graph_transformed.py', 'ir_pre_fusion.txt',
         'ir_post_fusion.txt', 'output_code.py', 'contract_observation.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item(path):
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def selected_files(source):
    """保留实际调试工件及目标边界，不归档可执行二进制与重复cache。"""
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(p in {'inductor-cache', 'triton-cache', 'trace', '__pycache__'} for p in relative.parts):
            continue
        if path.name in NAMES or path.name.startswith(('target-', 'snapshot-')):
            require(not path.is_symlink(), '证据文件不得是软链接')
            yield path


def archive(source, destination, *, failed_precheck=False):
    require(not destination.exists(), '拒绝覆盖既有运行归档')
    paths = list(selected_files(source))
    if failed_precheck:
        executions=[p for p in paths if p.name=='execution.json']
        require(bool(executions) and all(read(p)['return_code'] != 0 for p in executions)
                and not any(p.name=='result.json' for p in paths), '失败预检归档不可夹带成功记录')
    else:
        require(any(p.name == 'output_code.py' for p in paths), '未找到真实生成代码')
    inventory = []
    for path in paths:
        target = destination/path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        inventory.append({'path': str(target.relative_to(ROOT)), 'sha256': sha(target)})
    (destination/'inventory.json').write_text(json.dumps(
        {'source_run_dir': str(source), 'failed_precheck':failed_precheck, 'files': inventory}, ensure_ascii=False, indent=2)+'\n')


def review_arm(path, pattern, mode):
    record = read(path/'result.json')
    execution = read(path/'execution.json')
    require(execution['return_code'] == 0, '功能臂未正常结束')
    require(record['mode'] == mode and record['pattern'] == pattern
            and record['phase'] == 'functional', '功能臂归属不符')
    require(record['physical_device'] == '5', '本次审查只接收明确的物理 NPU 5')
    require(record['worker_sha256'] == sha(WORKER), '功能执行器已变化，先复验')
    require(record['target_control_sha256'] == sha(WORKER.with_name('target_entry_control.py')),
            'OFF控制源码已变化')
    require(record['observer_sha256'] == sha(WORKER.with_name('native_contract_observer.py')),
            '精确目标观察器已变化')
    for name, digest in record['source_snapshots'].items():
        require(Path(name).name == name and sha(path/('snapshot-' + name)) == digest,
                '执行时源码快照不符')
    require(record['target_control'].get('whole_pass_disabled') is False, '不得关闭整轮joint')
    code = inspect(path)
    require(code['output_code_count'] > 0 and not code['suspicious_cpu_lines'],
            '缺真实codegen或存在CPU转移线索，需先诊断')
    return record, {k:v for k,v in code.items() if k != 'files'}


def community_for_gate(stage, pattern, explicit=None):
    """历史失败不覆盖；部署后的同卡完整原件必须显式指定并关联已验签部署。"""
    if explicit is None:
        path = ROOT/stage['baseline']['path']
        require(sha(path) == stage['baseline']['sha256'], '社区原件哈希不符')
        return path
    from validate_attention_slice_deployment import verify as verify_deployment
    require(pattern == 22 and stage.get('deployment'), '仅已登记部署支持显式安装态复验')
    require(verify_deployment(ROOT, stage['deployment'])['installed_passed'], '安装态部署回归未完成')
    path = explicit.resolve(strict=True)
    issue = ROOT/'issues'/f'REF-sfdp-pattern-{pattern}-native'
    require(path.is_relative_to(issue/'evidence') and path.name == 'result.json', '原例必须为对应issue已归档证据')
    raw = read(path)
    require(raw.get('status') == 'community-contract-passed'
            and raw.get('isolated_codegen_candidate') is False
            and raw.get('isolated_registration_candidate') is False
            and raw.get('numerical_assertions_modified') is False
            and len(raw.get('tensor_assertions', [])) == 12
            and all(t.get('passed') is True for t in raw['tensor_assertions'])
            and raw.get('exact_target_observations') == 4, '缺完整无候选原方法覆盖')
    deployment = read(ROOT/stage['deployment']['path'])
    require(raw.get('loaded_source_sha256', {}).get(deployment['target']) == deployment['after_sha256'],
            '原例未加载已验证的部署文件')
    parents = [p for p in (issue/'adapter_runs').glob('*/run_result.json')
               if Path(read(p)['raw_artifact_dir']).name == path.parent.parent.name]
    require(len(parents) == 1, '原例父运行不唯一或缺失')
    parent = read(parents[0])
    require(parent['return_code'] == 0 and parent['installed_product_before'] == parent['installed_product_after']
            and parent['attention_select_slice_candidate'] is False
            and parent['attention_registration_candidate'] is False, '父运行失败、候选污染或安装态变动')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', required=True, type=int)
    parser.add_argument('--functional-run', required=True, type=Path,
                        help='包含 pattern-N/off 和 on 的单次 functional-* 根目录')
    parser.add_argument('--gate-root', required=True, type=Path)
    parser.add_argument('--review-note', required=True, help='本编号实际FX/IR/codegen的人工复核结论')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--community-run', type=Path, help='已验证部署之后的同卡完整原例归档；不覆盖旧baseline')
    args = parser.parse_args()
    require(Path.cwd().resolve() == WORK, '从 /home/z50063656/tmp 执行')
    require(args.pattern in set(range(1,25)) | {28,30}
            and args.pattern not in (16,17), '本编号GPU目标尚不合法')
    require(len(args.review_note.strip()) >= 20, '需要具体人工复核说明，不能仅写PASS')
    task = worker.task_for_pattern(args.pattern)
    au = f'AU-fuse-attention-sfdp-pattern-{args.pattern}'
    current = ROOT/'results/current'/task
    stage = read(current/'npu_stage_reviews.json')['units'][au]
    community = community_for_gate(stage, args.pattern, args.community_run)
    require(read(community).get('isolated_codegen_candidate', False) is False,
            '隔离codegen候选不能签安装态性能gate')
    source = args.functional_run.resolve(strict=True)/f'pattern-{args.pattern}'
    require(source.is_relative_to(WORK), '本次功能原件必须在工作临时目录')
    records, reviews = {}, {}
    for mode in ('off','on'):
        records[mode], reviews[mode] = review_arm(source/mode, args.pattern, mode)
    require(records['off']['physical_device'] == records['on']['physical_device'], 'OFF/ON跨物理设备')
    require(str(read(community)['physical_npu']) == records['on']['physical_device'], '原社区与功能/性能必须同卡')
    fields = ('task_id','acceptance_unit_id','backend','pytorch_commit','correctness',
              'numerical_execution','target_rewrite','graph_breaks','fallbacks','product_disabled',
              'measurement_workload','worker_sha256','target_control_sha256','observer_sha256','input_spec',
              'registration_name','dropout_p')
    gate = {key:records['on'][key] for key in fields}
    gate.update(reviewed_at=datetime.now().astimezone().isoformat(),
                reviewer='Codex：原社区合同及目标OFF/ON生成代码逐图复核',
                manual_codegen_review=args.review_note,
                community_oracle_scope=('原社区不比较输出；本gate数值仅覆盖注册派生微图'
                    if args.pattern in (19,20) else '原社区已执行的数值断言及独立性能输入比较'),
                codegen_review=reviews,
                gpu_reference=item(current/'gpu_reference_review.json'),
                community_functional=item(community),
                off_functional=item(source/'off/result.json'),
                target_functional=item(source/'on/result.json'))
    if args.community_run:
        gate['installed_deployment'] = item(ROOT/stage['deployment']['path'])
    # 完整验证在写入归档前完成；标准库导入worker不会初始化torch或设备。
    with tempfile.TemporaryDirectory(prefix='attention-gate-check-', dir=WORK) as temp:
        probe = Path(temp)/'gate.json'
        probe.write_text(json.dumps(gate, ensure_ascii=False, indent=2)+'\n')
        worker.read_gate(probe, args.pattern, 'npu')
    destination = ROOT/'issues'/f'REF-sfdp-pattern-{args.pattern}-native'/'evidence'/args.functional_run.name
    gate_path = args.gate_root.resolve()/task/f'pattern-{args.pattern}.json'
    if args.write:
        require(not destination.exists() and not gate_path.exists(), '拒绝覆盖既有证据或已签gate')
        archive(source, destination)
        for filename in (WORKER.name, 'target_entry_control.py', 'native_contract_observer.py'):
            origin = WORKER.with_name(filename)
            shutil.copy2(origin, destination/filename)
        gate['off_functional'] = item(destination/'off/result.json')
        gate['target_functional'] = item(destination/'on/result.json')
        gate['worker_snapshot'] = item(destination/WORKER.name)
        gate['target_control_snapshot'] = item(destination/'target_entry_control.py')
        gate['observer_snapshot'] = item(destination/'native_contract_observer.py')
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        with gate_path.open('x') as stream:
            stream.write(json.dumps(gate, ensure_ascii=False, indent=2)+'\n')
        worker.read_gate(gate_path, args.pattern, 'npu')
    print(json.dumps({'task':task, 'pattern':args.pattern, 'gate':str(gate_path),
                      'written':args.write, 'device_execution':False,
                      'performance_measured':False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
