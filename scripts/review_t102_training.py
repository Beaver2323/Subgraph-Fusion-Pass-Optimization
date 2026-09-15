#!/usr/bin/env python3
"""归档 T-102 注册修复证据；prepare 只备份，不修改安装包。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from review_attention_functional_gate import archive
from validate_t102_training_deployment import verify, COUNTS

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/home/z50063656/tmp')
PRODUCT = Path('/home/z50063656/envs/Pass/lib/python3.11/site-packages/torch_npu/_inductor/triton_experimental')
DEST = ROOT/'issues/REF-sfdp-pattern-1-native/deployment-20260915-training'
CANDIDATE = ROOT/'issues/REF-sfdp-pattern-1-native/candidate/sfdp_training.py'


def read(path):
    return json.loads(path.read_text())


def item(path):
    return {'path':str(path.resolve().relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def collect(parents, mode):
    result = {}
    for parent in parents:
        r = read(parent)
        number = r['case_id'].split('-')[-2]
        assert int(number) in COUNTS and number not in result
        assert r['return_code']==0 and r['installed_product_before']==r['installed_product_after']
        assert r['attention_registration_candidate'] is (mode=='candidate')
        raw = Path(r['raw_artifact_dir'])/'adapter'
        record = read(raw/'result.json')
        assert record['status']=='community-contract-passed'
        assert (len(record['tensor_assertions']), record['exact_target_observations'])==COUNTS[int(number)]
        subprocess.run([sys.executable, str(ROOT/'scripts/archive_prepared_npu_case.py'),
                        '--run-dir',str(raw),'--evidence-only','--keep-reports'],check=True,cwd=WORK)
        archived = ROOT/'issues'/r['case_id']/'evidence'/raw.parent.name/'adapter/result.json'
        result[number] = item(archived)
    assert set(result)=={'1','2','3','4','5'}
    return result


def preserve(path, data):
    text = json.dumps(data, ensure_ascii=False, indent=2)+'\n'
    if path.exists():
        assert path.read_text()==text, f'拒绝覆盖 {path}'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('prepare','finalize'),required=True)
    p.add_argument('--parent', type=Path, action='append', required=True)
    p.add_argument('--boundary', type=Path, required=True)
    args = p.parse_args()
    assert Path.cwd()==WORK
    mode = 'candidate' if args.phase=='prepare' else 'installed'
    cases = collect(args.parent, mode)
    boundary = read(args.boundary/'result.json')
    assert boundary['status']=='passed' and boundary['candidate'] is (mode=='candidate')
    boundary_dest = DEST/(mode+'-boundary')
    if not boundary_dest.exists():
        archive(args.boundary,boundary_dest)
        log = args.boundary.with_suffix('.log')
        if log.is_file():
            shutil.copy2(log,boundary_dest/'run.log')
    if args.phase=='prepare':
        assert not (PRODUCT/'sfdp_training.py').exists(), '已有部署文件，拒绝覆盖'
        assert not (DEST/'preparation.json').exists(), '已有备份，拒绝重建'
        shutil.copy2(PRODUCT/'__init__.py', DEST/'before-__init__.py')
        shutil.copy2(CANDIDATE, DEST/'candidate-sfdp_training.py')
        data = dict(generated_at=datetime.now().astimezone().isoformat(), task_id='T-102',
                    candidate_cases=cases, candidate_boundary=item(boundary_dest/'result.json'),
                    before_initializer=item(DEST/'before-__init__.py'),
                    candidate_sha256=item(CANDIDATE)['sha256'], deployed=False)
        preserve(DEST/'preparation.json', data)
        print('candidate_and_boundary_archived=OK installed_modified=false')
        return
    pre = read(DEST/'preparation.json')
    for filename in ('__init__.py','sfdp_training.py'):
        shutil.copy2(PRODUCT/filename, DEST/('after-'+filename))
    d = dict(generated_at=datetime.now().astimezone().isoformat(), task_id='T-102', backend='triton_experimental',
             deployed=True, pytorch_modified=False, product_disable_bypassed=False,
             candidate_sha256=pre['candidate_sha256'], candidate_cases=pre['candidate_cases'],
             candidate_boundary=pre['candidate_boundary'], installed_cases=cases,
             installed_boundary=item(boundary_dest/'result.json'),
             source_files={name:dict(target=str(PRODUCT/name),
                before=pre['before_initializer'] if name=='__init__.py' else None,
                after=item(DEST/('after-'+name))) for name in ('__init__.py','sfdp_training.py')})
    preserve(DEST/'deployment.json', d)
    reference = item(DEST/'deployment.json')
    verify(ROOT, reference)
    path = ROOT/'results/current/T-102/npu_stage_reviews.json'
    stages = read(path)
    for number in range(1,6):
        unit = f'AU-fuse-attention-sfdp-pattern-{number}'
        old = stages['units'][unit]
        digest = hashlib.sha256((json.dumps(old,ensure_ascii=False,indent=2)+'\n').encode()).hexdigest()
        preserve(ROOT/f'results/history/T-102/{unit}-stage-{digest[:16]}.json', old)
        new = dict(old, generated_at=d['generated_at'], deployment=reference, candidate=pre['candidate_cases'][str(number)],
                   candidate_verified=True, candidate_kind='registration',
                   repair_status='installed-original-and-boundaries-verified-performance-pending',
                   community_alignment_status='PARTIAL_ALIGNED',
                   next_action='安装态原合同及六项边界通过；独立OFF/ON与六臂性能待处置',
                   correctness_scope=f'安装态原社区原方法通过，Tensor比较{COUNTS[number][0]}次、本编号改写{COUNTS[number][1]}次；数值域与CUDA代码生成差异单列',
                   report=f'issues/REF-sfdp-pattern-{number}-native/训练注册修复报告.md')
        stages['units'][unit] = new
    stages['updated_at'] = d['generated_at']
    path.write_text(json.dumps(stages,ensure_ascii=False,indent=2)+'\n')
    print('training_deployment=verified performance_gate_issued=false')


if __name__ == '__main__':
    main()
