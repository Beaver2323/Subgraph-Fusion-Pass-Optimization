#!/usr/bin/env python3
"""在人工代码复核后验签 T-091/T-100 原合同及六臂证据；不运行设备、不部署产品。"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics

from inspect_npu_codegen import inspect

ROOT = Path(__file__).resolve().parents[1]
ORDER = ('off1','on1','on2','off2','off3','on3')
TASKS = {'T-091':('stack-normalization','AU-split-cat-normalize-stack-default'),
         'T-100':('linear-binary-folding','AU-binary-folding-folded-op')}


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item(path):
    return dict(path=str(path.resolve().relative_to(ROOT)),sha256=digest(path))


def require(value,message):
    if not value:
        raise ValueError(message)


def aggregate(records):
    timing, changes, spread = {}, {}, {}
    for clock in ('host_ms','event_ms'):
        timing[clock] = {mode:{q:statistics.median(records[f'{mode}{i}']['timing'][clock][q]
                            for i in (1,2,3)) for q in ('p50','p99')} for mode in ('off','on')}
        changes[clock] = {q:100*(1-timing[clock]['on'][q]/timing[clock]['off'][q]) for q in ('p50','p99')}
        for mode in ('off','on'):
            values = [records[f'{mode}{i}']['timing'][clock]['p50'] for i in (1,2,3)]
            spread[f'{clock}_{mode}'] = max(values)/min(values)
    flat = [v for metric in changes.values() for v in metric.values()]
    if max(spread.values()) > 1.2 or (min(flat) <= -5 and max(flat) >= 5):
        verdict = 'PERF_MIXED'
    elif min(flat) <= -5:
        verdict = 'PERF_REGRESSED'
    elif changes['host_ms']['p50'] >= 5 and changes['event_ms']['p50'] >= 5:
        verdict = 'PERF_IMPROVED'
    else:
        verdict = 'PERF_NEUTRAL'
    return timing,changes,spread,verdict


def check_current(task):
    """只读取仓库归档，不依赖原机器 tmp 路径；重算六臂并检查移植 gate。"""
    unit, au = TASKS[task]
    current = ROOT/'results/current'/task
    gate_path = current/'performance_gates'/f'{unit}.json'
    gate = load(gate_path)

    def bound(base, record):
        path = (base/record['path']).resolve()
        require(path.is_relative_to(ROOT.resolve()) and path.is_file(), '归档路径缺失或越界')
        require(digest(path) == record['sha256'], f'归档哈希改变: {path}')
        return path

    original = bound(gate_path.parent, gate['executed_gate'])
    executed = load(original)
    for name in ('gpu_reference','target_functional','off_functional','community_functional'):
        bound(gate_path.parent, gate[name])
        require(gate[name]['sha256'] == executed[name]['sha256'], f'移植gate改变原绑定: {name}')
    community = load(bound(gate_path.parent, gate['community_functional']))
    require(community['status']=='community-contract-passed' and community['tests_ran']==1
            and community['tests_skipped']==0, '原社区完整合同未通过')
    summary = load(current/'performance_summary.json')
    rows = [x for x in summary['acceptance_units'] if x['acceptance_unit_id']==au]
    require(len(rows)==1, '性能单元缺失或重复')
    row = rows[0]
    require(digest(bound(ROOT,row['worker_source'])) == executed['worker_sha256'], '运行worker快照不符')
    for inventory in row['inventories']:
        payload = load(bound(ROOT,inventory))
        for entry in payload['files']:
            bound(ROOT,entry)
    records = {arm:load(bound(ROOT,row['raw_results'][arm])) for arm in ORDER}
    functional = {mode:load(bound(gate_path.parent,gate[name])) for mode,name in
                  (('off','off_functional'),('on','target_functional'))}
    loaded_sources = {}
    for arm, raw in [*functional.items(),*records.items()]:
        require(raw['backend']=='triton_experimental' and raw['correctness']=='passed'
                and raw['numerical_execution'] is True and raw['acceptance_unit_id']==au
                and raw['graph_breaks']==raw['fallbacks']==0 and raw['product_disabled'] is False,
                f'{arm}: 功能合同不符')
        for key in ('worker_sha256','target_control_sha256','pytorch_commit','input_spec','measurement_workload'):
            require(raw[key]==executed[key], f'{arm}: 原gate与执行合同不一致')
        state = raw['state']
        require((state['handler_calls']>0 and state['graph_changes']>0 and state['target_counter']>0)
                if arm.startswith('on') else state['handler_calls']==state['target_counter']==0,
                f'{arm}: 精确目标ON/OFF不符')
        for path, sha in raw['loaded_source_sha256'].items():
            require(path not in loaded_sources or loaded_sources[path]==sha, '各臂实际源码变化')
            loaded_sources[path]=sha
        if arm in records:
            require(raw['gate_sha256']==digest(original), '六臂未绑定原执行gate')
            require(set(raw['samples'])=={'host_ms','event_ms'}, '缺少任一计时时钟')
            for clock,values in raw['samples'].items():
                require(len(values)==100 and all(math.isfinite(x) and x>0 for x in values), '样本不足/非法')
                ordered=sorted(values)
                for q,p in (('p50',0.5),('p99',0.99)):
                    position=99*p; index=int(position)
                    value=ordered[index]+(ordered[min(index+1,99)]-ordered[index])*(position-index)
                    require(math.isclose(value,raw['timing'][clock][q],rel_tol=1e-12), '分位数与样本不符')
    require(len({r['pid'] for r in [*functional.values(),*records.values()]})==8, '进程不独立')
    require(len({r['physical_device'] for r in records.values()})==1, '性能跨物理设备')
    timing,changes,spread,verdict=aggregate(records)
    require((timing,changes,spread,verdict)==(row['timing'],row['improvement_percent'],
            row['round_p50_spread_ratio'],row['verdict']), '汇总与六臂原样本计算不符')
    print(f'current_archive_validation=OK task={task} verdict={verdict} device_execution=false')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-current',action='store_true')
    p.add_argument('--task',choices=TASKS)
    p.add_argument('--functional-evidence',type=Path)
    p.add_argument('--performance-evidence',type=Path)
    p.add_argument('--executed-gate',type=Path)
    p.add_argument('--worker-snapshot',type=Path)
    p.add_argument('--review-note')
    p.add_argument('--write',action='store_true')
    args=p.parse_args()
    if args.check_current:
        if args.write:
            p.error('--check-current不写入结果')
        for task in ([args.task] if args.task else TASKS):
            check_current(task)
        return
    if not all((args.task,args.functional_evidence,args.performance_evidence,args.executed_gate,
                args.worker_snapshot,args.review_note)):
        p.error('初次验收必须提供task、功能/性能归档、执行gate、worker快照和人工代码复核说明')
    task=args.task; unit,au=TASKS[task]
    root=ROOT/'results/current'/task
    gate=load(args.executed_gate)
    for name in ('gpu_reference','target_functional','off_functional','community_functional'):
        bound=Path(gate[name]['path'])
        if not bound.is_absolute():bound=args.executed_gate.parent/bound
        require(bound.is_file() and digest(bound)==gate[name]['sha256'],f'gate来源失效: {name}')
    community=load(Path(gate['community_functional']['path']))
    assertions_passed = community.get('native_assertions_passed') is True
    if task=='T-091' and 'native_assertions_passed' not in community:
        # 旧 stack 适配器的 status 仅在 unittest.wasSuccessful 后生成；它只有一次原数值比较。
        assertions_passed = (community.get('exact_handler')=='normalize_stack_default'
            and community.get('body_or_assertions_modified') is False
            and len(community.get('tensor_assertions',[]))==1
            and community['tensor_assertions'][0].get('passed') is True)
    require(community['status']=='community-contract-passed' and community['tests_ran']==1
            and community['tests_skipped']==0 and assertions_passed,
            '完整社区原断言未通过')
    require(community['backend']=='triton_experimental' and community['acceptance_unit_id']==au
            and community['product_gate_bypassed'] is False,'社区后端/归属/gate错误')
    functional={m:load(args.functional_evidence/unit/m/'result.json') for m in ('off','on')}
    results={arm:load(args.performance_evidence/unit/arm/'result.json') for arm in ORDER}
    require(digest(args.worker_snapshot)==gate['worker_sha256'],'worker源码快照不符')
    reviews={}
    loaded_sources={}
    for arm,raw in [*functional.items(),*results.items()]:
        mode='off' if arm.startswith('off') else 'on'
        for key,expected in dict(task_id=task,acceptance_unit_id=au,backend='triton_experimental',
                                 correctness='passed',numerical_execution=True,graph_breaks=0,fallbacks=0,
                                 product_disabled=False,mode=mode).items():
            require(raw.get(key)==expected and type(raw.get(key)) is type(expected),f'{arm}: {key}错误')
        for key in ('worker_sha256','target_control_sha256','pytorch_commit','input_spec','measurement_workload'):
            require(raw[key]==gate[key],f'{arm}: 与gate合同不符: {key}')
        for path,sha in raw['loaded_source_sha256'].items():
            require(path not in loaded_sources or loaded_sources[path]==sha,f'{arm}: OFF/ON实际源码变化: {path}')
            loaded_sources[path]=sha
        state=raw['state']
        require((state['handler_calls']>0 and state['graph_changes']>0 and state['target_counter']>0)
                if mode=='on' else state['handler_calls']==0 and state['target_counter']==0,
                f'{arm}: 精确OFF/ON不符')
        base=(args.functional_evidence if arm in functional else args.performance_evidence)/unit/arm
        execution=load(base/'execution.json')
        require(execution['return_code']==0,f'{arm}: 运行未成功')
        review=inspect(base)
        require(review['output_code_count']>0 and not review['suspicious_cpu_lines'],f'{arm}: 缺codegen或存在CPU线索')
        reviews[arm]={k:v for k,v in review.items() if k!='files'}
        reviews[arm]['manual_review']=args.review_note
        if arm in results:
            require(raw['gate_sha256']==digest(args.executed_gate),f'{arm}: 非本次gate')
            require(raw['phase']=='benchmark',f'{arm}: 非benchmark')
            command=execution['command']
            require(command[command.index('--warmup')+1]=='10' and command[command.index('--runs')+1]=='100',
                    f'{arm}: 实际采样命令不符')
            for clock,values in raw['samples'].items():
                require(len(values)==100 and all(type(v) in (float,int) and math.isfinite(v) and v>0 for v in values),
                        f'{arm}: 样本不足或非法')
                ordered=sorted(values)
                for key,q in (('p50',0.5),('p99',0.99)):
                    pos=(len(ordered)-1)*q; lower=int(pos); upper=min(lower+1,len(ordered)-1)
                    actual=ordered[lower]+(ordered[upper]-ordered[lower])*(pos-lower)
                    require(math.isclose(actual,raw['timing'][clock][key],rel_tol=1e-12,abs_tol=1e-12),
                            f'{arm}: 分位数与原始样本不符')
    require(len({r['pid'] for r in results.values()})==6,'不是六个新进程')
    require(len({r['pid'] for r in (*functional.values(),*results.values())})==8,'功能与性能臂进程不独立')
    require(len({r['physical_device'] for r in results.values()})==1,'跨卡比较')
    timing,changes,spread,verdict=aggregate(results)
    now=datetime.now().astimezone().isoformat()
    functional_path=root/'functional'/f'{unit}.json'
    gate_path=root/'performance_gates'/f'{unit}.json'
    original_gate_path=root/'performance_gates'/f'{unit}-executed.json'
    f={k:v for k,v in functional['on'].items() if k not in ('loaded_source_sha256','samples','timing','memory')}
    f.update(generated_at=now,reviewer='Codex：原合同与生成代码人工复核',comparison_verdict='BEHAVIOR_UNCHANGED',
             repair_status='not-needed',source_evidence={m:item(args.functional_evidence/unit/m/'result.json') for m in functional},
             community_evidence=item(Path(gate['community_functional']['path'])),codegen_review={m:reviews[m] for m in functional},
             community_alignment=dict(status='FULL_ALIGNED',aligned_scope=['原社区测例的输入、数值与精确目标改写合同'],
                divergent_scope=[],open_scope=['社区未覆盖的其他shape/dtype不外推'],disposition='保留当前产品配置；性能只解释本社区派生子图'))
    # 可移植复核gate另存，六臂真正使用的gate按原字节保留，不能混淆两个哈希。
    portable=dict(gate)
    for key,path in dict(target_functional=args.functional_evidence/unit/'on/result.json',
                         off_functional=args.functional_evidence/unit/'off/result.json',
                         community_functional=Path(gate['community_functional']['path']),
                         gpu_reference=root/'gpu_reference_review.json').items():
        portable[key]=dict(path=os.path.relpath(path,gate_path.parent),sha256=digest(path))
    portable['executed_gate']=dict(path=original_gate_path.name,sha256=digest(args.executed_gate))
    summary=dict(schema_version='1.0',task_id=task,generated_at=now,backend='triton_experimental',
        status='functional-passed-performance-gates-signed',units=[dict(unit=unit,acceptance_unit_id=au,
        status='functional-passed-performance-gate-signed',functional_evidence=str(functional_path.relative_to(ROOT)),
        gate=str(gate_path.relative_to(ROOT)))])
    performance=dict(schema_version='1.0',task_id=task,generated_at=now,backend='triton_experimental',
        status='performance-disposition-complete',acceptance_units=[dict(acceptance_unit_id=au,unit=unit,
        performance_status='measured',verdict=verdict,product_action='不改变默认开关；局部收益不外推到完整模型',
        measurement_workload=gate['measurement_workload'],timing=timing,improvement_percent=changes,
        round_p50_spread_ratio=spread,memory={a:r['memory'] for a,r in results.items()},
        compile_ms={a:r['compile_ms'] for a,r in results.items()},codegen_review={a:reviews[a] for a in results},
        raw_results={a:item(args.performance_evidence/unit/a/'result.json') for a in ORDER},
        inventories=[item(base/'inventory.json') for base in (args.functional_evidence,args.performance_evidence)],
        worker_source=item(args.worker_snapshot),
        measurement_contract=dict(order=ORDER,fresh_processes=6,warmup=10,runs_per_arm=100,
            scope='社区功能图派生的compiled子图，不含编译/数据加载',model_e2e=False,
            aggregation='三OFF/三ON各臂p50/p99的中位数；高波动或时钟方向冲突记MIXED'))])
    if args.write:
        files={functional_path:f,gate_path:portable,root/'npu_functional_summary.json':summary,
               root/'performance_summary.json':performance}
        require(not any(x.exists() for x in (*files,original_gate_path)),'拒绝覆盖既有验收记录')
        for path,data in files.items():
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        shutil.copy2(args.executed_gate,original_gate_path)
    print(json.dumps(dict(task=task,verdict=verdict,improvement_percent=changes,written=args.write),ensure_ascii=False))


if __name__=='__main__':
    main()
