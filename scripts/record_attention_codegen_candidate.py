#!/usr/bin/env python3
"""归档原例及三个邻接的隔离候选评审；不部署、不冻结分母、不签性能 gate。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from npu_stage_review import verify
from record_attention_baseline_review import ROOT, WORK, read, sha, require, preserve_previous_stage


def check_run(path, number):
    path=path.resolve(strict=True)
    require(path.is_relative_to(ROOT/'issues'/f'REF-sfdp-pattern-{number}-native/adapter_runs'), '运行记录归属不符')
    parent=read(path)
    require(parent['return_code']==0 and parent['attention_select_slice_candidate'] is True
            and parent['attention_registration_candidate'] is False
            and parent['installed_product_before']==parent['installed_product_after'], '不是安装态不变的隔离候选')
    raw_name=Path(parent['raw_artifact_dir']).name
    raw=ROOT/'issues'/f'REF-sfdp-pattern-{number}-native/evidence'/raw_name/'adapter/result.json'
    r=read(raw)
    require(r['case_id']==f'REF-sfdp-pattern-{number}-native' and r['task_id']=='T-106'
            and r['status']=='community-contract-passed' and r['tests_ran']==1 and r['tests_skipped']==0
            and r['native_assertions_passed'] is True and r['numerical_assertions_modified'] is False
            and r['isolated_codegen_candidate'] is True and r['isolated_registration_candidate'] is False
            and r['product_gate_bypassed'] is False and r['backend']=='triton_experimental'
            and r['pytorch_commit']=='8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b', '候选原合同或后端不符')
    require(r['exact_target_observations']=={21:2,22:4,23:2,24:1}[number]
            and len(r['tensor_assertions'])=={21:2,22:12,23:6,24:1}[number]
            and all(t['passed'] is True for t in r['tensor_assertions']), '原合同分支不完整')
    meta=r['codegen_candidate']
    require(sha(raw.parent/'codegen-candidate/candidate_source.py')==meta['candidate_sha256']
            and sha(raw.parent/'codegen-candidate/installed_triton_source.py')==meta['installed_source_sha256'],
            '候选及安装态快照不符')
    key='_inductor/triton_experimental/codegen/triton.py'
    require(parent['installed_product_before'][key]['sha256']==meta['installed_source_sha256'], '安装态前后来源不一致')
    return dict(path=str(raw.relative_to(ROOT)),sha256=sha(raw),
        parent_run=dict(path=str(path.relative_to(ROOT)),sha256=sha(path)),
        tensor_comparisons=len(r['tensor_assertions']),exact_rewrites=r['exact_target_observations']),meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for number in (21,22,23,24):p.add_argument(f'--pattern-{number}-run',type=Path)
    p.add_argument('--write',action='store_true')
    p.add_argument('--check-current',action='store_true')
    a=p.parse_args()
    require(Path.cwd().resolve()==WORK, '从工作临时目录执行')
    path=ROOT/'results/current/T-106/npu_stage_reviews.json'
    stages=read(path);au='AU-fuse-attention-sfdp-pattern-22';previous=stages['units'][au]
    if a.check_current:
        require(not a.write and previous.get('candidate_kind')=='codegen', '缺少已登记codegen候选或参数冲突')
        entries=dict(previous['neighbor_candidates'], **{'22':previous['candidate']})
        require(set(entries)=={'21','22','23','24'}, '邻接证据不完整')
        metas=[]
        for number,item in entries.items():
            checked,meta=check_run(ROOT/item['parent_run']['path'],int(number));metas.append(meta)
            require(checked==item, '归档原件或父运行哈希不符')
        require(len({m['candidate_sha256'] for m in metas})==1
                and len({m['installed_source_sha256'] for m in metas})==1, '邻接候选来源变化')
        state = verify(ROOT,'T-106',au,previous)
        require(state['baseline_passed'] is False and state['candidate_passed'] is True,
                '安装态/候选边界不符')
        print('codegen_candidate_archive=OK original=1 neighbors=3 '
              f'installed_repair={str(state.get("installed_passed", False)).lower()} device_execution=false')
        return
    require(all(getattr(a,f'pattern_{n}_run') is not None for n in (21,22,23,24)), '需要原例和三个邻接的明确运行记录')
    runs={};metas=[]
    for n in (21,22,23,24):
        runs[str(n)],meta=check_run(getattr(a,f'pattern_{n}_run'),n);metas.append(meta)
    require(len({m['candidate_sha256'] for m in metas})==1
            and len({m['installed_source_sha256'] for m in metas})==1, '邻接与原例不是同候选/安装态')
    current=dict(previous,generated_at=datetime.now().astimezone().isoformat(),candidate=runs['22'],
        candidate_kind='codegen',candidate_verified=True,neighbor_candidates={k:v for k,v in runs.items() if k!='22'},
        repair_status='isolated-codegen-candidate-verified-not-deployed',
        reason='原例4组/12个Tensor及近邻21/23/24全部通过；仅进程内候选，安装态原数值失败保留。',
        next_action='补独立设备边界回归并评审部署；部署后重新验证原例和邻接，之前不得签安装态性能gate。',
        report='issues/REF-sfdp-pattern-22-native/修复候选验证报告.md')
    state=verify(ROOT,'T-106',au,current)
    require(state==dict(baseline_passed=False,candidate_passed=True), '原安装态失败必须继续保留')
    if a.write:
        preserve_previous_stage(ROOT,'T-106',au,previous,current,True)
        stages['units'][au]=current;stages['generated_at']=current['generated_at']
        path.write_text(json.dumps(stages,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(candidate_verified=True,installed_repair=False,performance_gate_issued=False,
        device_execution=False,written=a.write,runs=runs),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
