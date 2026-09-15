#!/usr/bin/env python3
"""只读核验 select-load 窄部署及无候选复验链；不导入 torch、不执行归档代码。"""
import ast
import importlib.util
import json
from pathlib import Path
import textwrap

_spec = importlib.util.spec_from_file_location(
    'attention_boundary_validation', Path(__file__).with_name('validate_attention_slice_boundary.py'))
_boundary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_boundary)
verify_boundary = _boundary.verify
sha, require, COMMIT, CANDIDATE = _boundary.sha, _boundary.require, _boundary.COMMIT, _boundary.CANDIDATE

METHODS = {'_maybe_record_select_lane_load', '_maybe_rewrite_select_lane_load'}
AFTER = '86f310ce99960eb12ead8b67db7264c068a56f9ecb5e0e7a43a6e48cbcad5a57'
BEFORE = '0e7ff7e98c8a540c89e7ebd6ef0214927a7c3ee7e7f3bdb98bdd248b2f0ff381'


def bound(root, item):
    p = (root / item['path']).resolve(strict=True)
    require(p.is_relative_to(root.resolve()) and sha(p) == item['sha256'], '部署引用缺失、越界或哈希不符')
    return p


def unchanged_remainder(before, after):
    def stripped(code):
        tree = ast.parse(code)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NPUTritonKernel')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
        require(len(methods) == 2, '缺少两个修复方法')
        cls.body = [n for n in cls.body if n not in methods]
        return ast.dump(tree, include_attributes=False)
    require(stripped(before) == stripped(after), '窄部署修改了两个方法以外的代码')


def verify(root, reference):
    root = root.resolve()
    path = bound(root, reference)
    d = json.loads(path.read_text())
    require(d['task_id'] == 'T-106' and d['backend'] == 'triton_experimental'
            and d['acceptance_unit_id'] == 'AU-fuse-attention-sfdp-pattern-22'
            and d['deployed'] is True and d['pytorch_modified'] is False
            and d['product_disable_bypassed'] is False and d['performance_gate_issued'] is False,
            '部署范围不符')
    before = path.parent / d['backup']; after = path.parent / d['installed_snapshot']
    require(before.resolve().is_relative_to(root) and after.resolve().is_relative_to(root)
            and sha(before) == d['before_sha256'] == BEFORE
            and sha(after) == d['after_sha256'] == AFTER
            and d['candidate_sha256'] == CANDIDATE and set(d['changed_methods']) == METHODS,
            '前后源码或候选不符')
    unchanged_remainder(before.read_text(), after.read_text())
    pre = (path.parent / d['candidate_boundary_result']).resolve(strict=True)
    require(pre.is_relative_to(root) and sha(pre) == d['candidate_boundary_result_sha256'], '部署前边界引用错误')
    verify_boundary(pre, BEFORE, d['candidate_boundary_runner_sha256'], 'candidate')
    # 窄部署的方法必须逐字等于实际执行候选，不能只相信部署摘要。
    code = after.read_text(); lines = code.splitlines(keepends=True)
    cls = next(n for n in ast.parse(code).body if isinstance(n, ast.ClassDef) and n.name=='NPUTritonKernel')
    for method in cls.body:
        if isinstance(method, ast.FunctionDef) and method.name in METHODS:
            actual = textwrap.dedent(''.join(lines[method.lineno-1:method.end_lineno]))
            require(actual == (pre.parent/'candidate'/f'after-{method.name}.py').read_text(), '部署方法与实测候选不同')
    if d['verification_status'] == 'pending-fresh-installed-original-neighbors-boundary':
        return dict(deployed=True, installed_passed=False)
    require(d['verification_status'] == 'passed-original-and-three-neighbors-and-boundary', '未知安装态验证状态')
    require(set(d['installed_cases']) == {'21','22','23','24'}, '安装态原例或邻接缺失')
    pids = set()
    for number, reference in d['installed_cases'].items():
        r = json.loads(bound(root, reference).read_text())
        require(r['status']=='community-contract-passed' and r['tests_ran']==1 and r['tests_skipped']==0
                and r['case_id']==f'REF-sfdp-pattern-{number}-native' and r['task_id']=='T-106'
                and r['backend']=='triton_experimental' and r['pytorch_commit']==COMMIT
                and r['native_assertions_passed'] is True and r['numerical_assertions_modified'] is False
                and r['isolated_codegen_candidate'] is False and r['isolated_registration_candidate'] is False
                and r['product_gate_bypassed'] is False, '缺少真实无候选安装态原合同')
        require(r['loaded_source_sha256'][d['target']]==AFTER, '复验没有加载实际部署文件')
        require(len(r['tensor_assertions'])=={'21':2,'22':12,'23':6,'24':1}[number]
                and all(t['passed'] is True and t['actual_device'].startswith('npu:')
                        and t['expected_device'].startswith('npu:') for t in r['tensor_assertions'])
                and r['exact_target_observations']=={'21':2,'22':4,'23':2,'24':1}[number], '原方法数值或目标覆盖不足')
        require(r['pid'] not in pids, '复验复用了进程')
        pids.add(r['pid'])
    post = bound(root, d['installed_boundary_result'])
    verify_boundary(post, AFTER, d['installed_boundary_runner_sha256'], 'installed')
    require(json.loads(post.read_text())['pid'] not in pids, '边界与原例复用了进程')
    return dict(deployed=True, installed_passed=True)
