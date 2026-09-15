#!/usr/bin/env python3
"""验签attention独立功能/六臂计时，归档后可离线重算；不运行设备或部署产品。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shutil

from review_attention_functional_gate import ROOT, archive, inspect, read, require, sha, worker
from review_t091_t100_completion import ORDER, aggregate


def item(path):
    return {'path': str(path.resolve().relative_to(ROOT)), 'sha256': sha(path)}


def bound(record):
    path = (ROOT/record['path']).resolve()
    require(path.is_relative_to(ROOT) and path.is_file() and sha(path) == record['sha256'],
            '证据路径越界、缺失或哈希不符')
    return path


def validate_samples(record):
    require(set(record['samples']) == {'host_ms','event_ms'}, '缺计时时钟')
    for clock, values in record['samples'].items():
        require(len(values) == 100 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values),
                '样本不足或非法')
        for quantile,q in (('p50',0.5),('p99',0.99)):
            require(math.isclose(worker.percentile(values,q),record['timing'][clock][quantile],rel_tol=1e-12),
                    '分位数与原样本不一致')


def verify_bundle(bundle):
    """只读取仓库文本，原服务器tmp及当前安装环境不参与离线复核。"""
    gate = read(bound(bundle['executed_gate']))
    community = read(bound(bundle['community_functional']))
    gpu = read(bound(bundle['gpu_reference']))
    au = gate['acceptance_unit_id']
    pattern = bundle['pattern']
    require(gate['task_id'] == worker.task_for_pattern(pattern)
            and au == f'AU-fuse-attention-sfdp-pattern-{pattern}'
            and gate['backend'] == 'triton_experimental', 'gate归属错误')
    if pattern == 22:
        from validate_attention_slice_deployment import verify as verify_deployment
        require('installed_deployment' in bundle, '22号缺已验证部署引用')
        require(verify_deployment(ROOT, bundle['installed_deployment'])['installed_passed'], '22号部署未通过')
        require(gate['installed_deployment']['sha256'] == bundle['installed_deployment']['sha256'], '部署绑定变更')
    if pattern in range(1,6):
        from validate_t102_training_deployment import verify as verify_training_deployment
        require('installed_deployment' in bundle, 'T-102 缺少注册窄部署引用')
        require(verify_training_deployment(ROOT, bundle['installed_deployment'])['installed_passed'], '训练注册部署未验证')
        require(gate['installed_deployment']['sha256'] == bundle['installed_deployment']['sha256'], '部署绑定变更')
    require(community['status']=='community-contract-passed' and community['native_assertions_passed'] is True
            and community['tests_ran']==1 and community['tests_skipped']==0
            and community['isolated_registration_candidate'] is False
            and community['product_gate_bypassed'] is False
            and community.get('isolated_codegen_candidate', False) is False
            and community['backend']=='triton_experimental'
            and community['acceptance_unit_id']==au
            and community['pytorch_commit']==worker.COMMIT, '完整安装态社区合同缺失')
    cases = [c for c in gpu['cases'] if c['acceptance_unit_id']==au]
    require(gpu['expected_pytorch_commit']==worker.COMMIT and len(cases)==1
            and cases[0]['tests_ran']==1 and cases[0]['tests_skipped']==0
            and cases[0]['target_review']['expected_target']==f'_sfdp_pattern_{pattern}'
            and cases[0]['target_review']['exact_target_observations']>0, 'GPU精确目标来源缺失')
    for name in ('gpu_reference','community_functional'):
        require(gate[name]['sha256']==bundle[name]['sha256'], 'gate原始绑定变化')
    for field,name in (('worker_sha256','worker_snapshot'), ('target_control_sha256','target_control_snapshot'),
                       ('observer_sha256','observer_snapshot')):
        require(sha(bound(bundle[name]))==gate[field], '执行辅助源码快照不符')
    for reference in bundle['inventories']:
        for entry in read(bound(reference))['files']:
            bound(entry)
    raws = {name:read(bound(reference)) for name,reference in bundle['raw_results'].items()}
    require(set(raws)==set(ORDER)|{'off','on'}, '需要独立功能两臂及六臂计时')
    sources = dict(community['loaded_source_sha256'])
    require(bool(sources), '缺少社区安装态来源')
    pids = {community['pid']}
    reviews = {}
    for name,r in raws.items():
        mode = 'off' if name.startswith('off') else 'on'
        for key,value in dict(backend='triton_experimental',correctness='passed',numerical_execution=True,
                              product_disabled=False,graph_breaks=0,fallbacks=0,mode=mode,
                              pytorch_commit=worker.COMMIT,pytorch_worktree_status='',
                              backend_selected_before_import=True,acceptance_unit_id=au).items():
            require(r.get(key)==value and type(r.get(key)) is type(value), f'{name}: {key}错误')
        require(r['pid'] not in pids and r['python_executable']==community['python_executable'], '复用进程或环境不同')
        pids.add(r['pid'])
        require(r['physical_device']==str(community['physical_npu']), '功能/性能跨物理设备')
        for key in ('worker_sha256','target_control_sha256','observer_sha256','measurement_workload',
                    'input_spec','registration_name','dropout_p'):
            require(r[key]==gate[key], f'{name}: 输入或执行源码与gate不同')
        require((r['exact_pattern_counter']>0 and r['general_fuse_attention_counter']>0) if mode=='on'
                else r['exact_pattern_counter']==r['general_fuse_attention_counter']==0, f'{name}: 精确OFF/ON不成立')
        require(r['target_control'].get('whole_pass_disabled') is False, '整轮pass关闭不能用于归因')
        require(bool(r['loaded_source_sha256']), '缺实际加载源码')
        for path,digest in r['loaded_source_sha256'].items():
            require(path not in sources or sources[path]==digest, '各功能/性能臂安装态来源不同')
            sources[path]=digest
        base = bound(bundle['raw_results'][name]).parent
        for source,digest in r['source_snapshots'].items():
            require(Path(source).name==source and sha(base/('snapshot-'+source))==digest, '启动源码快照不符')
        execution = read(base/'execution.json')
        require(execution['return_code']==0, '臂进程失败')
        review = inspect(base)
        require(review['output_code_count']>0 and not review['suspicious_cpu_lines'], '缺生成代码或存在CPU线索')
        reviews[name]={k:v for k,v in review.items() if k!='files'}
        if name in ORDER:
            require(r['phase']=='benchmark' and r['gate_sha256']==bundle['executed_gate']['sha256'], '计时未绑定原gate')
            require(execution['cooperative_performance_lock'] is True, '未独占本项目性能锁')
            command=execution['command']
            require(command[command.index('--warmup')+1]=='10' and command[command.index('--runs')+1]=='100',
                    '采样参数不符')
            validate_samples(r)
        else:
            require(r['phase']=='functional', '非独立功能记录')
            key='target_functional' if mode=='on' else 'off_functional'
            require(gate[key]['sha256']==bundle['raw_results'][name]['sha256'], '功能gate绑定改变')
    timing,changes,spread,verdict=aggregate({name:raws[name] for name in ORDER})
    return dict(timing=timing,improvement_percent=changes,round_p50_spread_ratio=spread,verdict=verdict),raws,reviews


def merge_summary(path, task, key, row, status):
    payload=read(path) if path.exists() else dict(schema_version='1.0',task_id=task,backend='triton_experimental',status=status,**{key:[]})
    require(not any(r['acceptance_unit_id']==row['acceptance_unit_id'] for r in payload[key]), '拒绝覆盖已验收单元')
    payload[key].append(row)
    payload['generated_at']=datetime.now().astimezone().isoformat()
    return payload


def completed_manifest(manifest, au, now):
    """只冻结完成逐图及性能审核的本单元，不升级同批其他编号。"""
    rows=[u for u in manifest['acceptance_units'] if u['acceptance_unit_id']==au]
    require(len(rows)==1, 'manifest单元不存在或重复')
    unit=rows[0]
    unit.update(review_status='frozen',denominator_eligible='yes-frozen',coverage_phase='formally-closed')
    for variant in unit['variants']:
        variant.update(reference_status='valid-reference',npu_status='passed')
    policy=manifest['counting_policy']
    policy['current_frozen_denominator_units']=sum(u['denominator_eligible']=='yes-frozen' for u in manifest['acceptance_units'])
    policy['current_formally_closed_units']=sum(u.get('coverage_phase')=='formally-closed' for u in manifest['acceptance_units'])
    complete=policy['current_formally_closed_units']==len(manifest['acceptance_units'])
    manifest.update(generated_at=now,status='completed' if complete else 'partially-completed')
    manifest['reference_contract']['suite_status']='valid-reference-suite' if complete else 'partially-frozen-reference'
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pattern',type=int)
    p.add_argument('--benchmark-run',type=Path)
    p.add_argument('--executed-gate',type=Path)
    p.add_argument('--review-note')
    p.add_argument('--write',action='store_true')
    p.add_argument('--check-current',action='store_true')
    args=p.parse_args()
    if args.check_current:
        paths=sorted((ROOT/'results/current').glob('T-*/attention_bundles/pattern-*.json'))
        for path in paths:
            bundle=read(path)
            if args.pattern and bundle['pattern']!=args.pattern: continue
            computed,_,_=verify_bundle(bundle)
            require(computed==bundle['aggregate'], '归档汇总与原始六臂不符')
            print(f'attention_archive=OK pattern={bundle["pattern"]} verdict={computed["verdict"]} device_execution=false')
        return
    require(args.pattern and args.benchmark_run and args.executed_gate and args.review_note,
            '需要编号、单次benchmark根目录、原执行gate和逐图人工复核说明')
    pattern=args.pattern; task=worker.task_for_pattern(pattern); unit=f'pattern-{pattern}'
    current=ROOT/'results/current'/task
    gate=read(args.executed_gate)
    require(gate['acceptance_unit_id']==f'AU-fuse-attention-sfdp-pattern-{pattern}', 'gate编号错误')
    source=args.benchmark_run.resolve(strict=True)/unit
    destination=ROOT/'issues'/f'REF-sfdp-pattern-{pattern}-native'/'evidence'/args.benchmark_run.name
    # 只归档显式选中的已成功六臂，失败臂不生成正式结论。
    for name in ORDER:
        require(read(source/name/'execution.json')['return_code']==0, '至少一臂未正常完成')
        validate_samples(read(source/name/'result.json'))
    if not args.write:
        print('precheck=OK; --write归档后执行完整验签，尚无正式结论')
        return
    gate_path=current/'performance_gates'/f'{unit}-executed.json'
    bundle_path=current/'attention_bundles'/f'{unit}.json'
    function_path=current/'functional'/f'{unit}.json'
    require(not any(path.exists() for path in (destination,gate_path,bundle_path,function_path)), '拒绝覆盖历史验收')
    archive(source,destination)
    gate_path.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(args.executed_gate,gate_path)
    bundle=dict(pattern=pattern,task_id=task,executed_gate=item(gate_path),
                reviewer='Codex：逐图及原样本复核',review_note=args.review_note,
                generated_at=datetime.now().astimezone().isoformat(),raw_results={})
    for name in ('gpu_reference','community_functional','worker_snapshot','target_control_snapshot','observer_snapshot'):
        path=Path(gate[name]['path'])
        require(sha(path)==gate[name]['sha256'], '原gate绑定变更')
        bundle[name]=item(path)
    if pattern == 22 or pattern in range(1,6):
        path = Path(gate['installed_deployment']['path'])
        require(sha(path) == gate['installed_deployment']['sha256'], '部署原件变化')
        bundle['installed_deployment'] = item(path)
    for name,key in (('off','off_functional'),('on','target_functional')):
        bundle['raw_results'][name]=item(Path(gate[key]['path']))
    bundle['raw_results'].update({name:item(destination/name/'result.json') for name in ORDER})
    functional_root=Path(gate['off_functional']['path']).parent.parent
    bundle['inventories']=[item(functional_root/'inventory.json'),item(destination/'inventory.json')]
    computed,raws,reviews=verify_bundle(bundle)
    bundle['aggregate']=computed
    au=gate['acceptance_unit_id']
    community=read(bound(bundle['community_functional']))
    function={k:v for k,v in raws['on'].items() if k not in ('loaded_source_sha256','samples','timing','memory')}
    function.update(unit=unit,source_evidence={m:bundle['raw_results'][m] for m in ('off','on')},
                    community_evidence=bundle['community_functional'],codegen_review={m:reviews[m] for m in ('off','on')},
                    reviewer=bundle['reviewer'],comparison_verdict='BEHAVIOR_UNCHANGED',repair_status='not-needed-product-unchanged',
                    community_alignment=dict(status='PARTIAL_ALIGNED' if pattern in (3,4,5,19,20,21,22,24) else 'FULL_ALIGNED',
                        aligned_scope=[f'原社区方法完整执行；记录{len(community["tensor_assertions"])}个Tensor比较；独立微图数值和精确OFF/ON通过'],
                        divergent_scope=['NPU数学展开与CUDA融合kernel路径不同'] if pattern in (5,21,22,24) else [],
                        open_scope=['原社区无数值oracle，派生微图不能回填'] if pattern in (3,4,19,20) else ['未覆盖的dtype/形状/梯度不外推'],
                        disposition='保留当前产品配置；局部时延不等于模型端到端收益'))
    if pattern == 22:
        function.update(repair_status='installed-original-neighbors-boundary-verified',
                        installed_deployment=bundle['installed_deployment'])
    if pattern in range(1,6):
        function.update(repair_status='installed-original-neighbors-boundary-verified',
                        comparison_verdict='NEWLY_SUPPORTED',
                        installed_deployment=bundle['installed_deployment'])
    if pattern == 19:
        function['community_alignment']['divergent_scope'].append(
            '本次FP32微图ON仍为两次NPU bmm与safe_softmax数学展开，不能记为融合FA内核收益')
        function['community_alignment']['open_scope'].append(
            '额外FP16 query与FP32 mask派生域已发生ON编译dtype错误；本FP32计时不能销项，CUDA同域行为尚未验证')
        function['community_alignment']['disposition'] += (
            '；half域缺口见issues/REF-sfdp-pattern-19-native/性能精度域缺口.md，需独立修复评审')
    functional=merge_summary(current/'npu_functional_summary.json',task,'units',
        dict(unit=unit,acceptance_unit_id=au,status='functional-passed-performance-gate-signed',
             functional_evidence=str(function_path.relative_to(ROOT)),gate=str(gate_path.relative_to(ROOT))),
        'functional-passed-performance-gates-signed')
    performance=merge_summary(current/'performance_summary.json',task,'acceptance_units',
        dict(unit=unit,acceptance_unit_id=au,performance_status='measured',**computed,
             product_action='不改变默认开关；非完整模型',raw_results={name:bundle['raw_results'][name] for name in ORDER},
             measurement_workload=gate['measurement_workload'],input_spec=gate['input_spec'],
             memory={name:raws[name]['memory'] for name in ORDER},
             compile_ms={name:raws[name]['compile_ms'] for name in ORDER},
             codegen_review={name:reviews[name] for name in ORDER},
             measurement_contract=dict(order=list(ORDER),fresh_processes=6,warmup=10,runs_per_arm=100,model_e2e=False,
                scope=raws['on']['measurement_stage'],aggregation='三OFF/三ON各臂p50/p99中位数；高波动或时钟冲突记MIXED'),
             bundle=str(bundle_path.relative_to(ROOT))),
        'performance-disposition-complete-for-listed-units')
    for path,payload in ((bundle_path,bundle),(function_path,function),(current/'npu_functional_summary.json',functional),
                         (current/'performance_summary.json',performance)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    manifest_path=ROOT/'upstream'/f'{task.lower().replace("-", "")}_manifest.yaml'
    manifest=completed_manifest(read(manifest_path),au,bundle['generated_at'])
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    plan_path=ROOT/'upstream'/f'{task.lower().replace("-", "")}_performance_plan.yaml'
    plan=read(plan_path)
    for row in plan['acceptance_units']:
        if row['acceptance_unit_id']==au:
            row.update(performance_status='measured',verdict=computed['verdict'],
                       evidence_bundle=str(bundle_path.relative_to(ROOT)))
    plan['generated_at']=bundle['generated_at']
    if task == 'T-102' and len(plan['acceptance_units']) == 5 and all(
            row['performance_status'] == 'measured' for row in plan['acceptance_units']):
        plan['status']='performance-disposition-complete'
        plan['implementation']['status']='implemented-runtime-validated'
    plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'task':task,'pattern':pattern,**computed},ensure_ascii=False))


if __name__=='__main__':
    main()
