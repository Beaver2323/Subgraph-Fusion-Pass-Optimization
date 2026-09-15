"""T-102 注册窄部署证据核验；只读正文，不导入 torch 或运行归档代码。"""
import ast
import hashlib
import json
from pathlib import Path

COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
# Pattern 5 的全零 bool mask 分支退化为 pattern 1；不把邻接的四次改写算成本编号。
COUNTS = {1:(20,8), 2:(10,4), 3:(0,2), 4:(0,2), 5:(30,8)}
SCENARIOS = {'div-fp32-training', 'mul-fp16-strided-training', 'reused-intermediate-negative',
             'tensor-scale-negative', 'add-mask-fp32-training', 'scalar-mask-negative'}
POSITIVE_TARGETS = {'div-fp32-training':1, 'mul-fp16-strided-training':2, 'add-mask-fp32-training':5}


def require(value, message):
    if not value:
        raise ValueError(message)


def bound(root, item):
    path = (root/item['path']).resolve(strict=True)
    require(path.is_relative_to(root.resolve()) and path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], '部署引用缺失、越界或哈希不符')
    return path


def verify(root, reference):
    d = json.loads(bound(root, reference).read_text())
    require(d['task_id']=='T-102' and d['backend']=='triton_experimental' and d['deployed'] is True
            and d['pytorch_modified'] is False and d['product_disable_bypassed'] is False,
            '注册部署范围不符')
    require(set(d['source_files'])=={'__init__.py','sfdp_training.py'}, '只允许两个 experimental 文件')
    for name, item in d['source_files'].items():
        require(item['target'].endswith('/torch_npu/_inductor/triton_experimental/'+name), '部署目标不属于 experimental')
    initializer = d['source_files']['__init__.py']
    before = ast.parse(bound(root, initializer['before']).read_text())
    after = ast.parse(bound(root, initializer['after']).read_text())
    activation = next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='_activate')
    added = [n for n in activation.body if
             isinstance(n,ast.ImportFrom) and n.module=='sfdp_training' or
             isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
             isinstance(n.value.func,ast.Name) and n.value.func.id=='install_sfdp_training_patterns']
    require(len(added)==2, '缺少唯一惰性激活入口')
    activation.body = [n for n in activation.body if n not in added]
    require(ast.dump(before)==ast.dump(after), '注册激活之外出现无关安装态变更')
    helper = d['source_files']['sfdp_training.py']
    require(helper['before'] is None and helper['after']['sha256']==d['candidate_sha256'], '部署源码不是实际验证候选')
    bound(root, helper['after'])
    pids = set()
    for mode in ('candidate','installed'):
        require(set(d[mode+'_cases'])=={'1','2','3','4','5'}, '缺少五个原合同')
        for number, item in d[mode+'_cases'].items():
            r = json.loads(bound(root,item).read_text())
            tensors, exact = COUNTS[int(number)]
            require(r['status']=='community-contract-passed' and r['tests_ran']==1 and r['tests_skipped']==0
                    and r['native_assertions_passed'] is True and r['numerical_assertions_modified'] is False
                    and r['isolated_registration_candidate'] is (mode=='candidate')
                    and r['isolated_codegen_candidate'] is False and r['product_gate_bypassed'] is False
                    and r['pytorch_commit']==COMMIT and r['backend']=='triton_experimental'
                    and str(r['physical_npu'])=='5' and not r['test_errors'] and not r['test_failures']
                    and r['case_id']==f'REF-sfdp-pattern-{number}-native'
                    and len(r['tensor_assertions'])==tensors and r['exact_target_observations']==exact,
                    '原方法不完整或候选/安装态混淆')
            actual_exact = sum(o['graph_changed'] is True and
                               o['target'].startswith(f'_sfdp_pattern_{number}_') for o in r['observations'])
            require(actual_exact==exact, '精确改写汇总与实际观察不一致')
            require(all(x['passed'] is True and x['actual_device'].startswith('npu:')
                        and x['expected_device'].startswith('npu:') for x in r['tensor_assertions']), '缺少真实 NPU 比较')
            require(r['pid'] not in pids, '原例复用了进程')
            pids.add(r['pid'])
            if mode=='candidate':
                require(r['candidate_snapshot_sha256']==d['candidate_sha256'], '候选源码不符')
            else:
                for item in d['source_files'].values():
                    require(r['loaded_source_sha256'].get(item['target'])==item['after']['sha256'], '实际加载安装源码不符')
        boundary_path = bound(root,d[mode+'_boundary'])
        boundary = json.loads(boundary_path.read_text())
        require(boundary['status']=='passed' and boundary['candidate'] is (mode=='candidate')
                and boundary['backend']=='triton_experimental' and boundary['pytorch_commit']==COMMIT
                and boundary['product_sha256']==d['candidate_sha256']
                and str(boundary['physical_npu'])=='5'
                and boundary['pid'] not in pids
                and len(boundary['scenarios'])==6 and {r['name'] for r in boundary['scenarios']}==SCENARIOS
                and all(r['passed'] is True for r in boundary['scenarios']), '边界实测不完整')
        pids.add(boundary['pid'])
        for filename, digest in (('harness_source.py',boundary['harness_sha256']),
                                 ('candidate_source.py',boundary['product_sha256'])):
            require(hashlib.sha256((boundary_path.parent/filename).read_bytes()).hexdigest()==digest,
                    '边界实际执行源码快照不符')
        for row in boundary['scenarios']:
            require(row['target']==POSITIVE_TARGETS.get(row['name']), '正负例边界身份被更改')
            require((not row['actual_targets']) if row['target'] is None else
                    any(name.startswith(f'_sfdp_pattern_{row["target"]}_') for name in row['actual_targets']),
                    '边界声明与真实目标不符')
    require(json.loads(bound(root,d['candidate_boundary']).read_text())['harness_sha256']==
            json.loads(bound(root,d['installed_boundary']).read_text())['harness_sha256'],
            '部署前后边界执行器发生变化')
    return dict(deployed=True, installed_passed=True)
