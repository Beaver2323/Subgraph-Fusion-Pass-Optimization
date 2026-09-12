#!/usr/bin/env python3
"""T-096 安装态原合同/范围/OFF-ON审查及六臂性能收束，不制造正确性结论。"""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

from review_t087_installed import ROOT, load, dump, digest, ref, validate_record

AU = 'AU-misc-patterns-e8m0-rceil-log2'
COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
CASE = 'REF-e8m0-log2-pattern-native'
DIRECTORY = ROOT/'results/current/T-096'
WORKER = ROOT/'runners/t096_installed_worker.py'
SOURCE = Path('/home/z50063656/Pass/src/pytorch/test/inductor/test_fp8.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def archive(base, case, label):
    dest = ROOT/'issues'/case/'evidence'/label
    if not dest.exists():
        subprocess.run([sys.executable, str(ROOT/'scripts/archive_issue_evidence.py'),
                        '--issue', case, '--run', str(base), '--label', label], check=True)
    inventory = load(dest/'inventory.json')['files']
    for item in inventory:
        require(digest(ROOT/item['path']) == item['sha256'], '归档哈希错误')
    return dest, inventory


def audit_code(base, on):
    codes = list(base.rglob('output_code.py'))
    require(codes, f'无 output_code.py：{base}')
    ops = set()
    for path in codes:
        text = path.read_text()
        require('torch.npu' in text and not re.search(r'\.cpu\(|device=[\x27\x22]cpu|\.to\([\x27\x22]cpu', text), '生成代码含非预期CPU路径')
        called = set(re.findall(r'torch\.ops\.((?:aten|npu)\.[\w.]+)\(', text))
        require(called <= {'aten.view.dtype','aten.__rshift__.Scalar','aten.ceil.default'}, f'未审查NPU extern: {called}')
        require('extern_kernels.' not in text, '未审查 extern_kernels')
        if on:
            require('libdevice.log2(' not in text and 'torch.ops.aten.ceil.default(' not in text,
                    'ON 仍保留 log2/ceil')
            require('8388607, tl.int32' in text and '4194304, tl.int32' in text and 'tl.where(' in text,
                    'ON 位运算/输入域处理未生成')
        ops.update(called)
    return dict(graphs=len(codes), npu_graph_internal_extern=sorted(ops),
                cpu_or_graph_external_fallbacks=0,
                scope='NPU dtype-view、整数右移/ceil 为图内NPU extern；不声称与CUDA kernel完全相同')


def arm(base, case, mode, phase):
    row, launch = load(base/'artifacts/result.json'), load(base/'execution.json')
    expected = dict(backend='triton_experimental', pytorch_commit=COMMIT, status='passed', correctness='passed',
                    case=case, mode=mode, phase=phase, worker_sha256=digest(WORKER), graph_breaks=0,
                    mathematical_equal=True, product_candidate=None, product_gate_bypassed=False,
                    pytorch_worktree_status='')
    require(all(row.get(k) == v for k,v in expected.items()), '臂身份/正确性/版本不符')
    require(launch['return_code'] == 0 and launch['pid'] == row['pid'] and not launch['timed_out'], '臂进程失败')
    is_hit = mode == 'on' and case not in ('fp16-guard','bf16-guard','cpu-guard')
    require((row['state']['calls'] > 0 and row['state']['changes'] > 0) if is_hit else row['state']['calls'] == 0,
            '精确目标/guard不符')
    require(row['numerical_execution'] is (case != 'cpu-guard'), '设备执行记录不符')
    log = (base/'stderr.log').read_text()
    require(not re.search(r'fall back to run on the CPU|fallback to CPU', log, re.I), 'CPU fallback告警')
    sources = {p:h for p,h in row['loaded_source_sha256'].items() if not Path(p).is_relative_to('/home/z50063656/tmp')}
    require(any(p.endswith('/triton_experimental/e8m0.py') for p in sources), '未绑定安装态E8M0模块')
    require(all(digest(Path(p)) == h for p,h in sources.items()), '当前产品/依赖源码与臂不一致')
    audit = audit_code(base/'artifacts/debug', is_hit) if case != 'cpu-guard' else {'scope':'CPU负例，精确目标零调用；不计NPU性能'}
    return row, sources, audit


def functional(args):
    now = datetime.now().astimezone().isoformat()
    gpu_path = DIRECTORY/'gpu_reference_review.json'
    gpu = load(gpu_path)
    require(gpu['status'] == 'gpu-contract-reviewed-awaiting-npu' and gpu['expected_pytorch_commit'] == COMMIT
            and gpu['tests_skipped'] == 0 and gpu['acceptance_units'] == [AU], 'GPU合同未接受')
    case_ids = {CASE, 'REF-e8m0-log2-one-ulp-native', 'REF-e8m0-log2-gh178045-native'}
    require(len(args.launch) == 3, '需要三个原例安装态启动记录')
    evidence, inventory, seen, products, code_reviews = [], [], set(), {}, {}
    for launch_path in args.launch:
        require(launch_path.resolve().is_relative_to(ROOT/'issues'), '启动记录必须在issue内')
        launch = load(launch_path)
        case = launch['case_id']
        require(case in case_ids and case not in seen and launch['return_code'] == 0
                and launch['product_candidate'] is None and not launch['e8m0_candidate'], '原例未以安装态通过')
        require(launch['installed_product_before'] == launch['installed_product_after'], '原例期间安装包改变')
        seen.add(case)
        base = Path(launch['raw_artifact_dir'])
        row = load(base/'adapter/result.json')
        validate_record(row)
        require(row['status'] == 'community-contract-passed' and row['original_test_passed'] and row['target_rewrite']
                and row['numerical_execution'] and row['graph_breaks'] == 0 and row['source_sha256'] == digest(SOURCE), '原例数学/改图未通过')
        code_reviews[case] = audit_code(base/'adapter/debug', True)
        for item in launch['installed_product_before'].values():
            name, sha = item['path'], item['sha256']
            require(name not in products or products[name] == sha, '原例间产品文件改变')
            products[name] = sha
        dest, files = archive(base, case, 'installed-pass-' + launch_path.parent.name)
        inventory += files
        evidence += [ref(dest/'adapter/result.json'), ref(launch_path)]
    before, after = load(args.domain_before/'artifacts/result.json'), load(args.domain_after/'artifacts/result.json')
    require(after['positive_mathematical_contract_passed'] and after['expected_repaired'], '安装态数值域数学检查失败')
    require(before['backend'] == after['backend'] == 'triton_experimental' and before['pytorch_commit'] == after['pytorch_commit'] == COMMIT,
            '边界检查后端/版本不符')
    require([r['bits'] for r in before['rows']] == [r['bits'] for r in after['rows']], '前后边界输入不同')
    for old, new in zip(before['rows'], after['rows']):
        require(new['compiled'] == (new['exact_positive'] if new['exact_positive'] is not None else old['compiled']),
                '数学编码失败或特殊值行为被改变')
    audit_code(args.domain_after/'artifacts/debug', True)
    for base, label in ((args.domain_before,'domain-before-20260911'), (args.domain_after,'domain-installed-20260911')):
        dest, files = archive(base, CASE, label)
        inventory += files
        evidence.append(ref(dest/'artifacts/result.json'))
    arms = [('off','ordinary','off'),('on','ordinary','on')]
    arms += [(c,c,'on') for c in ('fp16-guard','bf16-guard','cpu-guard','strided','exponent-sweep')]
    pids = set()
    for name, case, mode in arms:
        row, sources, audit = arm(args.run/name, case, mode, 'functional')
        require(row['pid'] not in pids, '功能臂复用了进程')
        pids.add(row['pid'])
        if case == 'ordinary':
            require(row['eager_equal'] is True and row['input_spec']['shape'] == [7], '性能测例不是原正确七元素向量')
        for path, sha in sources.items():
            require(path not in products or products[path] == sha, '功能臂间源码变更')
            products[path] = sha
        dest, files = archive(args.run/name, CASE, 'functional-' + args.run.name + '-' + name)
        inventory += files
        evidence += [ref(dest/'artifacts/result.json'), ref(dest/'execution.json')]
        code_reviews[name] = audit
    deployment_path = ROOT/'issues'/CASE/'installed_deployment_20260911.json'
    deployment = load(deployment_path)
    package = Path('/home/z50063656/envs/Pass/lib/python3.11/site-packages/torch_npu')
    for entry in deployment['files']:
        require(products[str(package/entry['relative'])] == entry['after_sha256'], '产品部署hash不一致')
        if entry['backup']:
            require(digest(Path(deployment['backup_directory'])/entry['backup']) == entry['before_sha256'], '备份失效')
    deployment.update(verified_at=now, verification_status='original-3-of-3-domain-and-seven-controls-passed')
    dump(deployment_path, deployment)
    evidence += [ref(deployment_path), ref(gpu_path)]
    gate_path = DIRECTORY/'performance_gates/e8m0-rceil-log2.json'
    require(not gate_path.exists(), '门禁已经存在，拒绝覆盖')
    gate = dict(schema_version='1.0', task_id='T-096', acceptance_unit_id=AU, generated_at=now, reviewed_at=now,
        reviewer='Codex：原数学合同、输入域和实际codegen审核', backend='triton_experimental', pytorch_commit=COMMIT,
        correctness='passed', target_rewrite='confirmed', graph_breaks=0, fallbacks=0, benchmark_allowed=True,
        worker_sha256=digest(WORKER), workload='community-ordinary-seven-fp32', evidence=evidence,
        installed_files=products, codegen_review=code_reviews,
        scope='只允许原社区七元素正确向量计时；one-ULP/gh178045/subnormal数学修复仅功能，不用错误OFF算收益')
    dump(gate_path, gate)
    functional_path = DIRECTORY/'functional/e8m0-rceil-log2.json'
    require(not functional_path.exists(), '功能原件已存在，拒绝覆盖')
    dump(functional_path, dict(schema_version='1.0', generated_at=now, task_id='T-096', acceptance_unit_id=AU,
        backend='triton_experimental', pytorch_commit=COMMIT, correctness='passed', numerical_execution=True,
        target_rewrite='confirmed', graph_breaks=0, fallbacks=0, comparison_verdict='NEWLY_SUPPORTED',
        repair_status='installed-fix-verified-not-upstream-merged', gate=ref(gate_path), evidence=evidence,
        artifact_inventory=inventory, codegen_review=code_reviews,
        community_alignment=dict(status='PARTIAL_ALIGNED',
            aligned_scope=['A100 reference对应三个原社区FP32合同全部通过', '正正规数的指数/尾数算法与社区pre-SM100对齐'],
            divergent_scope=['NPU增加subnormal/负值/NaN边界保护，未声称对应CUDA已实测', 'dtype-view与右移保留图内NPU extern，不同于CUDA融合kernel'],
            open_scope=['新增边界的GPU邻接对照尚无；不增加冻结分母', 'FP16/BF16保持原不命中保护，不代表这两种dtype全范围精度支持'],
            disposition='保留数学oracle和已测NPU特殊值行为；原三例已修复，额外边界明确标为NPU派生扩展')))
    print(f't096_functional=passed original=3/3 controls=7/7 gate={gate_path}')


def benchmark(args):
    gate_path = DIRECTORY/'performance_gates/e8m0-rceil-log2.json'
    gate = load(gate_path)
    require(gate['worker_sha256'] == digest(WORKER) and gate['benchmark_allowed'], '门禁过期')
    rows, inventory, audits, pids = {}, [], {}, set()
    for name in ('off1','on1','on2','off2','off3','on3'):
        row, sources, audit = arm(args.run/name, 'ordinary', name.rstrip('123'), 'benchmark')
        require(row['gate_sha256'] == digest(gate_path) and row['pid'] not in pids, '未绑定门禁或进程重复')
        require(all(len(row['samples'][clock]) == 100 for clock in ('host_ms','event_ms')), '样本数错误')
        require(all(path not in gate['installed_files'] or gate['installed_files'][path] == sha for path,sha in sources.items()), '门禁后源码漂移')
        pids.add(row['pid'])
        rows[name], audits[name] = row, audit
        _, files = archive(args.run/name, CASE, 'performance-' + args.run.name + '-' + name)
        inventory += files
    timing, improvements, spread = {}, {}, {}
    for clock in ('host_ms','event_ms'):
        timing[clock] = {}
        for mode in ('off','on'):
            timing[clock][mode] = {q:statistics.median(row['timing'][clock][q] for name,row in rows.items() if name.startswith(mode)) for q in ('p50','p99')}
            medians = [r['timing'][clock]['p50'] for n,r in rows.items() if n.startswith(mode)]
            spread[clock+'_'+mode] = max(medians)/min(medians)
        improvements[clock] = {q:(1-timing[clock]['on'][q]/timing[clock]['off'][q])*100 for q in ('p50','p99')}
    event, host = improvements['event_ms'], improvements['host_ms']
    if max(spread.values()) > 1.5:
        verdict = 'PERF_MIXED'
    elif min(event.values()) >= 5 and host['p99'] >= -5:
        verdict = 'PERF_IMPROVED'
    elif max(event.values()) <= -5:
        verdict = 'PERF_REGRESSED'
    elif min(event.values()) <= -5 or max(event.values()) >= 5:
        verdict = 'PERF_MIXED'
    else:
        verdict = 'PERF_NEUTRAL'
    now = datetime.now().astimezone().isoformat()
    path = DIRECTORY/'performance_summary.json'
    require(not path.exists(), '性能原件已存在，拒绝覆盖')
    dump(path, dict(schema_version='1.0', task_id='T-096', backend='triton_experimental', generated_at=now,
        status='performance-disposition-complete', acceptance_units=[dict(acceptance_unit_id=AU, unit='e8m0-rceil-log2',
            performance_status='measured', verdict=verdict, improvement_percent=improvements, timing=timing,
            round_p50_spread_ratio=spread, gate=ref(gate_path), source_run_dir=str(args.run), artifact_inventory=inventory,
            compile_ms={n:r['compile_ms'] for n,r in rows.items()}, memory={n:r['memory'] for n,r in rows.items()},
            codegen_review=audits, measurement_contract=dict(order=list(rows), runs_per_arm=100, warmup=10,
                fresh_processes=6, workload='社区原七元素FP32向量', model_e2e=False),
            product_action='新增开关默认开启是正确性修复，不是依据此局部性能把显式关闭优化打开')]))
    function_path = DIRECTORY/'functional/e8m0-rceil-log2.json'
    dump(DIRECTORY/'npu_functional_summary.json', dict(schema_version='1.0', task_id='T-096', generated_at=now,
        backend='triton_experimental', status='functional-passed-performance-gates-signed', units=[dict(
            acceptance_unit_id=AU, unit='e8m0-rceil-log2', status='functional-passed-performance-gate-signed',
            functional_evidence=str(function_path.relative_to(ROOT)), gate=str(gate_path.relative_to(ROOT)))],
        scope='三个原FP32合同；特殊值/guard/strided/sweep为NPU派生扩展，未冒充GPU原例'))
    mpath = ROOT/'upstream/t096_manifest.yaml'
    manifest = load(mpath)
    manifest.update(generated_at=now,status='completed')
    manifest['counting_policy']['current_formally_closed_units'] = 1
    for unit in manifest['acceptance_units']:
        unit['coverage_phase'] = 'formally-closed'
        for variant in unit['variants']:
            variant['npu_status'] = 'passed-installed-fix'
    dump(mpath, manifest)
    ppath = ROOT/'upstream/t096_performance_plan.yaml'
    plan = load(ppath)
    plan.update(generated_at=now, status='performance-disposition-complete')
    plan['implementation'] = dict(status='implemented-runtime-validated', entrypoint='scripts/run_t096_installed.py',
        reason='安装态原合同、输入域和精确OFF/ON门禁通过，仅原七元素正确输入测性能')
    unit = plan['acceptance_units'][0]
    unit.update(worker_unit='e8m0-rceil-log2', performance_status='measured', verdict=verdict,
        functional_gate='原3例及NPU输入域、dtype/device/stride/指数边界通过；独立OFF/ON七元素数学正确且精确改图',
        off_on_control='新进程内设置 experimental.config.enable_e8m0_rceil_log2；仅本目标extra_check读取；不改总pattern_matcher',
        artifacts=['原GPU reference','安装态原例及边界/guard','实际FX/IR/output_code','六臂原始时延/显存/编译时间'])
    unit['workloads'].append(dict(workload_id='community-ordinary-seven-fp32',role='measured',
        shape_contract='[1,2,4,3,1.5,0.5,0.25]，float32[7]→uint8[7]', measurement_scope='目标编码子图一次调用，不含编译，不是模型E2E'))
    plan['hardware_contract'].update(performance_minimum_devices=1, performance_world_size=1)
    dump(ppath, plan)
    bpath = DIRECTORY/'npu_blocker_review.json'
    blocked = load(bpath)
    blocked['installed_resolution'] = dict(generated_at=now,status='resolved-by-installed-fix',functional=ref(function_path),
        note='原失败与隔离候选证据保留；当前以已部署修复及性能处置为准')
    dump(bpath, blocked)
    print(f't096_performance={verdict} improvements={improvements}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['functional','benchmark'], required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--launch', type=Path, action='append', default=[])
    p.add_argument('--domain-before', type=Path)
    p.add_argument('--domain-after', type=Path)
    args = p.parse_args()
    require(args.run.resolve().is_relative_to('/home/z50063656/tmp'), 'run必须在tmp')
    if args.phase == 'functional':
        require(args.domain_before and args.domain_after, '缺少安装前后输入域证据')
        functional(args)
    else:
        benchmark(args)


if __name__ == '__main__':
    main()
