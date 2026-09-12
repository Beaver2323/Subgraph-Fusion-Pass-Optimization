#!/usr/bin/env python3
"""把安装态失败与进程局部候选分列；只复核和归档，不部署、不签正式 PASS。"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def checked(path, status):
    row = load(path)
    if (row['status'] != status or row['backend'] != 'triton_experimental'
            or row['pytorch_commit'] != COMMIT or row['tests_ran'] != 1 or row['tests_skipped'] != 0):
        raise ValueError(f'不符合同：{path}')
    return row


def ref(path):
    return {'path':str(path.relative_to(ROOT)), 'sha256':digest(path)}


def candidate(case, label):
    runs = sorted((ROOT/'issues'/case/'candidate_runs').glob('*/run_result.json'))
    if not runs:
        raise ValueError(f'没有候选运行：{case}')
    launch = load(runs[-1])
    if launch['return_code'] != 0 or launch['tests_ran'] != 1 or launch['tests_skipped'] != 0:
        raise ValueError(f'最新候选未通过：{case}')
    base = Path(launch['raw_artifact_dir']).resolve(strict=True)
    row = checked(base/'adapter/result.json', 'community-contract-passed')
    proposal = row['product_candidate']
    if not proposal or proposal['installed_environment_modified'] is not False or row['numerical_execution'] is not True:
        raise ValueError('候选没有明确隔离或设备数值证明')
    if digest(Path(proposal['path'])) != proposal['sha256']:
        raise ValueError('候选源码已漂移')
    for field in ('codegen',):
        if field in proposal and digest(Path(proposal[field]['path'])) != proposal[field]['sha256']:
            raise ValueError('codegen源码已漂移')
    codes = list((base/'adapter/debug').rglob('output_code.py'))
    if not codes:
        raise ValueError('候选没有生成代码')
    for code in codes:
        text = code.read_text()
        if 'torch.npu' not in text or '.cpu(' in text:
            raise ValueError('候选设备路径不符')
        if case.startswith('REF-e8m0'):
            # NPU view(dtype)/rshift 保留图内 extern，不要求 CUDA 的 tl.bitcast 写法。
            required = ('torch.ops.aten.view.dtype(', 'torch.ops.aten.__rshift__.Scalar(',
                        '8388607, tl.int32', '255, tl.int32', '254, tl.int32', 'tmp0 & tmp1')
            if 'libdevice.log2(' in text or 'torch.ops.aten.ceil.default(' in text or not all(s in text for s in required):
                raise ValueError('位运算替换未进入实际 NPU extern/生成核')
    destination = ROOT/'issues'/case/'evidence'/label
    if not destination.exists():
        subprocess.run([sys.executable, str(ROOT/'scripts/archive_issue_evidence.py'),
            '--issue',case,'--run',str(base),'--label',label], check=True)
    archived = destination/'adapter/result.json'
    if digest(archived) != digest(base/'adapter/result.json'):
        raise ValueError('归档与原件不一致，禁止覆盖旧候选')
    return dict(ref(archived), launch=ref(runs[-1]), status='original-contract-passed-in-isolated-source-candidate',
        installed_environment_modified=False, product_merged=False, functional_closure=False,
        product_candidate=proposal)


def main():
    if (ROOT/'results/current/T-087/functional/respecialize-current-device.json').exists():
        raise ValueError('已有安装态修复结果，禁止旧候选收束器覆盖当前状态；--check 仍可核验历史')
    now = datetime.now().astimezone().isoformat()
    t087 = ROOT/'results/current/T-087/npu_blocker_review.json'
    blocker = load(t087)
    blocker.update(generated_at=now, repair_status='isolated-source-candidate-passed-not-deployed',
        candidate=candidate('REF-respecialize-current-device-native', 'candidate-device-codegen-pass-20260911'),
        next_action='两文件隔离候选通过原单设备合同；待安装态同步授权及部署回归，不外推多rank复用。')
    dump(t087, blocker)
    cases = ['REF-e8m0-log2-pattern-native','REF-e8m0-log2-one-ulp-native','REF-e8m0-log2-gh178045-native']
    before = {}
    for case in cases:
        path = ROOT/'issues'/case/'evidence/baseline-20260911/adapter/result.json'
        row = checked(path, 'failed-contract')
        before[case] = dict(ref(path), original_test_passed=row['original_test_passed'],
            target_rewrite=row['target_rewrite'], numerical_execution=row['numerical_execution'])
    after = {case:candidate(case, 'candidate-bitwise-pass-20260911') for case in cases}
    au = 'AU-misc-patterns-e8m0-rceil-log2'
    dump(ROOT/'results/current/T-096/npu_blocker_review.json', dict(task_id='T-096', acceptance_unit_id=au,
        generated_at=now, backend='triton_experimental', result=before[cases[1]], baseline_cases=before,
        candidate_cases=after, correctness='ordinary-pass-boundary-failed-installed',
        current_phase='npu-precision-failed-awaiting-product-fix', repair_status='isolated-source-candidate-passed-not-deployed',
        reason='安装态没有NPU位运算pattern：普通输入正确但未改写；one-ULP/gh178045精度失败。',
        next_action='候选3/3原合同及实际位运算核通过；待产品边界/安装态部署评审。数学oracle与eager软件log2差异明确保留；不对错误OFF计时。',
        report='issues/REF-e8m0-log2-pattern-native/根因分析.md'))
    path = ROOT/'upstream/t096_manifest.yaml'
    manifest = load(path)
    gpu = load(ROOT/'results/current/T-096/gpu_reference_review.json')
    if gpu['status'] != 'gpu-contract-reviewed-awaiting-npu' or gpu['acceptance_units'] != [au]:
        raise ValueError('T-096 GPU合同尚未接受')
    manifest.update(generated_at=now, status='npu-product-fix-pending')
    manifest['reference_contract']['suite_status'] = 'valid-reference-suite'
    manifest['counting_policy'].update(current_frozen_denominator_units=1, current_formally_closed_units=0)
    for unit in manifest['acceptance_units']:
        unit.update(review_status='frozen', denominator_eligible='yes-frozen', coverage_phase='npu-precision-failed-awaiting-product-fix')
        for variant in unit['variants']:
            variant.update(reference_status='valid-reference', npu_status='installed-contract-failed-candidate-only-pass')
    dump(path, manifest)
    print('candidate_review=T087:1/1,T096:3/3 source-only installed_environment_modified=false formal_closure=false')


def validate_archived():
    count = 0
    for task in ('T-087', 'T-096'):
        path = ROOT/'results/current'/task/'npu_blocker_review.json'
        record = load(path)
        before = list(record.get('baseline_cases', {}).values()) or [record['result']]
        after = list(record.get('candidate_cases', {}).values()) or [record['candidate']]
        for group, status in ((before, 'failed-contract'), (after, 'community-contract-passed')):
            for item in group:
                source = (ROOT/item['path']).resolve()
                if not source.is_relative_to(ROOT) or digest(source) != item['sha256']:
                    raise ValueError('候选/基线原件哈希或路径错误')
                raw = checked(source, status)
                if status == 'community-contract-passed':
                    if raw['numerical_execution'] is not True or raw['product_candidate']['installed_environment_modified'] is not False:
                        raise ValueError('候选隔离/设备事实不符')
                    if item['functional_closure'] is not False:
                        raise ValueError('未部署候选不可冒充正式关闭')
                inventory = load(source.parent.parent/'inventory.json')
                for artifact in inventory['files']:
                    target = (ROOT/artifact['path']).resolve()
                    if not target.is_relative_to(ROOT) or digest(target) != artifact['sha256']:
                        raise ValueError('编译产物归档哈希错误')
                count += 1
    print(f'pending_candidate_archive_validation=OK records={count} device_execution=false candidate_records_deployed=false installed_repairs_tracked_separately=true')


if __name__ == '__main__':
    if sys.argv[1:] == ['--check']:
        validate_archived()
    elif sys.argv[1:]:
        raise SystemExit('仅支持 --check 或无参数显式收束本次候选')
    else:
        main()
