#!/usr/bin/env python3
"""复核 T-087 安装态原例/近邻与部署哈希，归档并单独签发性能免测门禁。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ISSUE = ROOT/'issues/REF-respecialize-current-device-native'
AU = 'AU-post-grad-respecialize-current-device'
COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def ref(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path))


def archive(source, label):
    dest = ISSUE/'evidence'/label
    if not dest.exists():
        subprocess.run([sys.executable, str(ROOT/'scripts/archive_issue_evidence.py'),
                        '--issue', ISSUE.name, '--run', str(source), '--label', label], check=True)
    return dest


def validate_record(row):
    if (row.get('backend') != 'triton_experimental' or row.get('pytorch_commit') != COMMIT
            or row.get('tests_ran') != 1 or row.get('tests_skipped') != 0
            or row.get('product_candidate') is not None or row.get('product_gate_bypassed') is not False):
        raise ValueError('安装态合同身份、执行或隔离条件不符')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch', type=Path, required=True)
    p.add_argument('--neighbors', type=Path, required=True)
    args = p.parse_args()
    launch_path = args.launch.resolve(strict=True)
    neighbors_path = args.neighbors.resolve(strict=True)
    if not launch_path.is_relative_to(ISSUE) or not neighbors_path.is_relative_to(ISSUE):
        p.error('只接收本 issue 的已保存执行记录')
    launch = load(launch_path)
    if launch['return_code'] != 0 or launch.get('product_candidate') or launch['e8m0_candidate']:
        raise ValueError('原用例安装态执行未通过')
    snapshot = launch['installed_product_before']
    if snapshot != launch['installed_product_after']:
        raise ValueError('执行期间安装包变更')
    deployment_path = ISSUE/'installed_deployment_20260911.json'
    deployment = load(deployment_path)
    if not deployment['deployed']:
        raise ValueError('候选未部署')
    for item in deployment['files']:
        relative = item['installed'].split('/torch_npu/', 1)[1]
        if snapshot[relative]['sha256'] != item['after_sha256'] or digest(Path(item['installed'])) != item['after_sha256']:
            raise ValueError('安装包与部署/执行哈希不符')
        backup = Path(deployment['backup_directory'])/item['backup']
        if digest(backup) != item['before_sha256']:
            raise ValueError('修复前备份哈希不符')
    raw = Path(launch['raw_artifact_dir'])
    original = load(raw/'adapter/result.json')
    validate_record(original)
    if (original['status'] != 'community-contract-passed' or original['numerical_execution'] is not True
            or not any(x['changed'] for x in original['state']['graphs'])):
        raise ValueError('原例数值/改图未通过')
    original_dest = archive(raw, 'installed-pass-20260911T181204')
    if digest(original_dest/'adapter/result.json') != digest(raw/'adapter/result.json'):
        raise ValueError('原件与归档不符')
    neighbor_launches = load(neighbors_path)
    expected = {'ordinary-compile', 'cpp-wrapper-rejected', 'fx-wrapper-rejected', 'two-device-code-identical'}
    if {r['case'] for r in neighbor_launches} != expected or len(neighbor_launches) != 4:
        raise ValueError('近邻不完整')
    neighbors = []
    inventory = load(original_dest/'inventory.json')['files']
    pids = {original['pid']}
    for entry in neighbor_launches:
        base = Path(entry['artifact_dir'])
        row = load(base/'result.json')
        validate_record(row)
        if entry['return_code'] or row['status'] != 'passed' or row['pid'] in pids:
            raise ValueError('近邻失败或未用独立进程')
        pids.add(row['pid'])
        for name, sha in row['installed_files'].items():
            if snapshot[name]['sha256'] != sha:
                raise ValueError('近邻与原例产品文件不同')
        target = archive(base, 'installed-neighbor-' + entry['case'] + '-20260911')
        if digest(target/'result.json') != digest(base/'result.json'):
            raise ValueError('近邻归档不符')
        inventory += load(target/'inventory.json')['files']
        neighbors.append(ref(target/'result.json'))
    codes = list(original_dest.rglob('output_code.py'))
    if not codes or any('DeviceProperties(type=\'npu\', index=None' not in c.read_text()
                        or 'torch.npu.current_device()' not in c.read_text() for c in codes):
        raise ValueError('未找到安装态运行时设备代码')
    now = datetime.now().astimezone().isoformat()
    deployment.update(verified_at=now, verification_status='passed-original-and-four-neighbors')
    dump(deployment_path, deployment)
    functional = dict(schema_version='1.0', task_id='T-087', acceptance_unit_id=AU, generated_at=now,
        backend='triton_experimental', pytorch_commit=COMMIT, correctness='passed', numerical_execution=True,
        target_rewrite='confirmed', graph_breaks=0, fallbacks=0,
        comparison_verdict='NEWLY_SUPPORTED', repair_status='installed-fix-verified-not-upstream-merged',
        original=ref(original_dest/'adapter/result.json'), launch=ref(launch_path), neighbors=neighbors,
        neighbor_launch=ref(neighbors_path), deployment=ref(deployment_path), artifact_inventory=inventory,
        installed_product=snapshot, product_candidate=None,
        community_alignment=dict(status='FULL_ALIGNED',
            aligned_scope=['冻结单设备原合同：真实编译/数值与无固化设备编号', '两设备代码一致性与普通编译及非法wrapper拒绝近邻'],
            divergent_scope=[], open_scope=['未验证多rank通信和跨卡kernel handle复用；不计入当前单设备合同'],
            disposition='原安装态失败已修复并回归；仅Pass环境部署，未社区合入'))
    directory = ROOT/'results/current/T-087'
    function_path = directory/'functional/respecialize-current-device.json'
    if function_path.exists():
        raise ValueError('正式结果已存在，拒绝覆盖；应使用新 revision 证据')
    dump(function_path, functional)
    gate_path = directory/'performance_gates/respecialize-current-device.json'
    dump(gate_path, dict(schema_version='1.0', generated_at=now, acceptance_unit_id=AU,
        backend='triton_experimental', correctness='passed', target_rewrite='confirmed', graph_breaks=0, fallbacks=0,
        performance_exempt=True, benchmark_allowed=False, functional=ref(function_path),
        reason='lowering前必需设备再特化，无合法可执行OFF；不是性能提升/产品关闭免测'))
    summary_path = directory/'npu_functional_summary.json'
    summary = load(summary_path)
    summary['units'].append(dict(unit='respecialize-current-device', acceptance_unit_id=AU,
        status='functional-passed-performance-gate-signed', functional_evidence=str(function_path.relative_to(ROOT)),
        gate=str(gate_path.relative_to(ROOT)), performance_exempt=True))
    summary.update(generated_at=now, scope='训练单元及设备解析单元均完成；设备解析为已签免测，不计算性能收益')
    dump(summary_path, summary)
    manifest_path = ROOT/'upstream/t087_manifest.yaml'
    manifest = load(manifest_path)
    manifest.update(generated_at=now, status='completed')
    manifest['counting_policy']['current_formally_closed_units'] = 2
    unit = next(u for u in manifest['acceptance_units'] if u['acceptance_unit_id'] == AU)
    unit['coverage_phase'] = 'formally-closed'
    for variant in unit['variants']:
        variant['npu_status'] = 'passed-installed-fix'
    dump(manifest_path, manifest)
    blocked_path = directory/'npu_blocker_review.json'
    blocker = load(blocked_path)
    blocker['installed_resolution'] = dict(generated_at=now, functional=ref(function_path),
        status='resolved-by-installed-fix', note='历史失败及隔离候选字段原样保留，由当前安装态结果接替结论')
    dump(blocked_path, blocker)
    print('installed_repair=T-087 original=1/1 neighbors=4/4 closure=2/2 performance=exempt')


if __name__ == '__main__':
    main()
