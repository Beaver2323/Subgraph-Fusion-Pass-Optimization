#!/usr/bin/env python3
"""离线核验安装态修复证据链；不导入 torch，不要求复核机存在原 Pass 安装路径。"""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_refs(value, root=ROOT):
    """只允许仓库内可恢复原件；绝对部署路径仅作记录，不能冒充归档引用。"""
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            path = (root/value['path']).resolve()
            require(not Path(value['path']).is_absolute() and path.is_relative_to(root.resolve()), '证据越出仓库')
            require(digest(path) == value['sha256'], f'证据哈希不符：{path}')
            if 'bytes' in value:
                require(path.stat().st_size == value['bytes'], '证据大小不符')
        for key, item in value.items():
            # 安装快照有绝对 path；由启动原件哈希和部署前后记录交叉核验。
            if key != 'installed_product':
                check_refs(item, root)
    elif isinstance(value, list):
        for item in value:
            check_refs(item, root)


def check_contract(row):
    require(row.get('backend') == 'triton_experimental' and row.get('pytorch_commit') == COMMIT, '后端/源码合同错误')
    require(row.get('product_candidate', 'missing') is None and row.get('product_gate_bypassed') is False, '隔离候选不能认作安装态')
    require(row.get('tests_ran') == 1 and row.get('tests_skipped') == 0, '原合同必须真实执行且无skip')


def check_common(row):
    require(row.get('backend') == 'triton_experimental' and row.get('pytorch_commit') == COMMIT, '功能后端/版本不符')
    require(row.get('correctness') == 'passed' and row.get('numerical_execution') is True
            and row.get('target_rewrite') == 'confirmed' and row.get('graph_breaks') == row.get('fallbacks') == 0, '功能门禁未通过')
    require(row.get('repair_status') == 'installed-fix-verified-not-upstream-merged'
            and row.get('comparison_verdict') == 'NEWLY_SUPPORTED', '修复状态错误')


def check_timing(row):
    for clock in ('host_ms', 'event_ms'):
        values = row['samples'][clock]
        require(len(values) == 100 and all(math.isfinite(v) and v > 0 for v in values), '性能样本不全或非法')
        ordered = sorted(values)
        for name, quantile in [('p50', .5), ('p99', .99)]:
            position = (len(ordered)-1)*quantile
            lower, upper = int(position), min(int(position)+1, len(ordered)-1)
            measured = ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)
            require(abs(measured-row['timing'][clock][name]) < 1e-12, '逐样本分位数与记录不符')


def check():
    t087 = load(ROOT/'results/current/T-087/functional/respecialize-current-device.json')
    t096 = load(ROOT/'results/current/T-096/functional/e8m0-rceil-log2.json')
    for function in (t087, t096):
        check_common(function)
        check_refs(function)
    original = load(ROOT/t087['original']['path'])
    check_contract(original)
    require(original['status'] == 'community-contract-passed' and original['numerical_execution'], 'T087原例失败')
    launch = load(ROOT/t087['launch']['path'])
    require(launch['installed_product_before'] == launch['installed_product_after'] == t087['installed_product'], 'T087安装快照漂移')
    deployment = load(ROOT/t087['deployment']['path'])
    require(deployment['deployed'] and deployment['verification_status'] == 'passed-original-and-four-neighbors', 'T087未部署')
    for entry in deployment['files']:
        name = entry['installed'].split('/torch_npu/', 1)[1]
        require(t087['installed_product'][name]['sha256'] == entry['after_sha256'], 'T087部署文件不符')
    require(len(t087['neighbors']) == 4, 'T087近邻不完整')
    pids = {original['pid']}
    for reference in t087['neighbors']:
        row = load(ROOT/reference['path'])
        check_contract(row)
        require(row['status'] == 'passed' and row['pid'] not in pids, 'T087近邻失败/复用进程')
        pids.add(row['pid'])
    exempt = load(ROOT/'results/current/T-087/performance_gates/respecialize-current-device.json')
    check_refs(exempt)
    require(exempt['performance_exempt'] and exempt['benchmark_allowed'] is False, 'T087不得造OFF计时')
    gate_path = ROOT/t096['gate']['path']
    gate = load(gate_path)
    check_refs(gate)
    require(gate['benchmark_allowed'] and gate['worker_sha256'] == digest(ROOT/'runners/t096_installed_worker.py'), 'T096门禁失效')
    originals, controls = [], []
    for reference in t096['evidence']:
        row = load(ROOT/reference['path'])
        if reference['path'].endswith('/adapter/result.json'):
            check_contract(row)
            require(row['original_test_passed'] and row['target_rewrite'] and row['numerical_execution'], 'T096原例不通过')
            originals.append(row)
        if row.get('phase') == 'functional':
            require(row['status'] == 'passed' and row['mathematical_equal'] and row['product_candidate'] is None, 'T096功能臂失败')
            controls.append(row)
    require(len(originals) == 3 and len(controls) == len({r['pid'] for r in controls}) == 7, 'T096原例/独立控制臂不全')
    perf = load(ROOT/'results/current/T-096/performance_summary.json')['acceptance_units'][0]
    check_refs(perf)
    rows = [load(ROOT/r['path']) for r in perf['artifact_inventory'] if r['path'].endswith('/artifacts/result.json')]
    require(len(rows) == len({r['pid'] for r in rows}) == 6, '六个独立性能进程不全')
    for row in rows:
        require(row['gate_sha256'] == digest(gate_path) and row['phase'] == 'benchmark'
                and row['status'] == 'passed' and row['mathematical_equal'] and row['eager_equal'], '性能臂合同不符')
        require(row['worker_sha256'] == gate['worker_sha256'] and row['product_candidate'] is None, '性能臂非已审核安装态')
        require(row['state']['calls'] > 0 and row['state']['changes'] > 0 if row['mode'] == 'on' else row['state']['calls'] == 0,
                '性能OFF/ON目标命中错误')
        check_timing(row)
        for path, sha in row['loaded_source_sha256'].items():
            if path in gate['installed_files']:
                require(gate['installed_files'][path] == sha, '性能臂产品版本漂移')
    for clock in ('host_ms', 'event_ms'):
        for quantile in ('p50', 'p99'):
            times = {mode: statistics.median(r['timing'][clock][quantile] for r in rows if r['mode'] == mode) for mode in ('off', 'on')}
            for mode, value in times.items():
                require(abs(perf['timing'][clock][mode][quantile]-value) < 1e-12, '性能汇总中位数错误')
            improvement = (1-times['on']/times['off'])*100
            require(abs(perf['improvement_percent'][clock][quantile]-improvement) < 1e-9, '收益百分比错误')
    require(t096['community_alignment']['status'] == 'PARTIAL_ALIGNED', '额外NPU数值域不能冒充CUDA全对齐')
    require(perf['verdict'] == 'PERF_REGRESSED' and max(perf['improvement_percent']['event_ms'].values()) < -5,
            '本次实测回退不得冒充收益')
    combined = load(ROOT/'issues/REF-respecialize-current-device-native/combined_install_review_20260911.json')
    check_refs(combined)
    check_refs(load(ROOT/combined['inventory']['path'])['files'])
    result = load(ROOT/combined['result']['path'])
    check_contract(result)
    require(result['status'] == 'community-contract-passed' and result['numerical_execution'], '联合安装态原例失败')
    launch = load(ROOT/combined['launch']['path'])
    require(launch['installed_product_before'] == launch['installed_product_after'], '联合安装态执行时产品变化')
    for entry in deployment['files']:
        name = entry['installed'].split('/torch_npu/', 1)[1]
        require(launch['installed_product_before'][name]['sha256'] == entry['after_sha256'], '联合安装态T087文件不符')
    for entry in load(ROOT/combined['t096_deployment']['path'])['files']:
        require(launch['installed_product_before'][entry['relative']]['sha256'] == entry['after_sha256'], '联合安装态T096文件不符')
    return {'installed_repairs': 2, 'original_contracts': 4, 'neighbor_control_arms': 11, 'performance_arms': 6, 'combined_regression': 1}


if __name__ == '__main__':
    print('installed_archive_validation=OK', json.dumps(check(), sort_keys=True))
