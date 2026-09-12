#!/usr/bin/env python3
"""T-096：原生方法的设备适配及只读目标观察，不绕过产品 CUDA guard。"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/z50063656/Pass/src/pytorch/test/inductor/test_fp8.py')
COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'


class DeviceAdapter(ast.NodeTransformer):
    def visit_Constant(self, node):
        if node.value == 'cuda':
            return ast.copy_location(ast.Constant('npu'), node)
        return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--e8m0-candidate', action='store_true', help='仅本进程验证已评审 NPU 注册候选')
    args = parser.parse_args()
    if Path.cwd() != Path('/home/z50063656/tmp'):
        parser.error('必须从 tmp 启动')
    plan = json.loads((ROOT/'upstream/t096_reference_plan.yaml').read_text())
    selected = [c for c in plan['cases'] if c['case_id'] == args.case]
    if len(selected) != 1:
        parser.error('未知 T-096 case')
    case = selected[0]
    method = case['source_test'].split('.')[-1]
    out = args.artifact_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TORCHINDUCTOR_NPU_BACKEND='triton_experimental', TORCH_COMPILE_DEBUG='1',
        TORCH_COMPILE_DEBUG_DIR=str(out/'debug'), TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'),
        TRITON_CACHE_DIR=str(out/'triton-cache'), TORCHINDUCTOR_FORCE_DISABLE_CACHES='1',
        TORCHINDUCTOR_COMPILE_THREADS='1')
    import torch
    import torch_npu
    import torch_npu.testing
    from torch.testing._internal.common_utils import TestCase
    from torch._inductor.fx_passes.misc_patterns import _misc_patterns_init
    from torch._inductor.pattern_matcher import ReplacementPatternEntry
    from torch._dynamo.utils import counters
    from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu
    register_inductor_npu()
    assert torch.version.git_version == COMMIT
    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    torch.npu.set_device(0)
    candidate = None
    if args.e8m0_candidate:
        candidate_path = ROOT/'issues/REF-e8m0-log2-pattern-native/e8m0_npu_candidate.py'
        spec = importlib.util.spec_from_file_location('e8m0_npu_candidate', candidate_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.install()
        candidate = {'path':str(candidate_path), 'sha256':hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            'scope':'process-local-source-candidate', 'installed_environment_modified':False}
    original_class = next(n for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == 'TestE8M0Log2PatternBitManip')
    methods = [n for n in original_class.body if isinstance(n, ast.FunctionDef) and n.name == method]
    assert len(methods) == 1
    cls_ast = ast.ClassDef(name=original_class.name, bases=[ast.Name(id='TestCase', ctx=ast.Load())],
        keywords=[], body=methods, decorator_list=[])
    tree = ast.fix_missing_locations(DeviceAdapter().visit(ast.Module(body=[cls_ast], type_ignores=[])))
    (out/'adapted_test.py').write_text(ast.unparse(tree)+'\n')
    scope = {'torch':torch, 'TestCase':TestCase, '_misc_patterns_init':_misc_patterns_init}
    exec(compile(tree, str(SOURCE), 'exec'), scope)
    state = {'target_records':[], 'executions':[]}
    original_apply, original_compile = ReplacementPatternEntry.apply, torch.compile

    @functools.wraps(original_apply)
    def observe(entry, match, graph, node):
        name = getattr(entry, 'pattern_name', '')
        if name != 'e8m0_rceil_log2_pattern':
            return original_apply(entry, match, graph, node)
        index = len(state['target_records'])
        before = graph.python_code('self').src
        (out/f'target-{index}-before.txt').write_text(before)
        result = original_apply(entry, match, graph, node)
        after = graph.python_code('self').src
        (out/f'target-{index}-after.txt').write_text(after)
        state['target_records'].append({'pattern':name, 'changed':before != after})
        return result

    @functools.wraps(original_compile)
    def observe_compile(fn, *pos, **kw):
        compiled = original_compile(fn, *pos, **kw)
        @functools.wraps(fn)
        def run(*inputs, **kwargs):
            result = compiled(*inputs, **kwargs)
            eager = fn(*inputs, **kwargs)
            torch.npu.synchronize()
            state['executions'].append({'input':inputs[0].tolist(), 'compiled':result.tolist(),
                'eager':eager.tolist(), 'device':str(result.device),
                'eager_equal':torch.equal(result, eager)})
            return result
        return run

    with mock.patch.object(ReplacementPatternEntry, 'apply', observe), mock.patch.object(torch, 'compile', observe_compile):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([scope[original_class.name](method)]))
    torch.npu.synchronize()
    test_passed = result.wasSuccessful() and result.testsRun == 1 and not result.skipped
    target = any(row['changed'] for row in state['target_records'])
    breaks = sum(counters['graph_break'].values())
    numerical = bool(state['executions']) and all(row['device'].startswith('npu:') for row in state['executions'])
    valid = test_passed and target and numerical and breaks == 0
    record = dict(task_id='T-096', case_id=args.case, acceptance_unit_id=case['acceptance_unit_id'],
        generated_at=datetime.now().astimezone().isoformat(), pid=os.getpid(),
        status='community-contract-passed' if valid else 'failed-contract',
        original_test_passed=test_passed, tests_ran=result.testsRun, tests_skipped=len(result.skipped),
        backend=_InductorNpuRegistry._loaded_backend, pytorch_commit=torch.version.git_version,
        torch_npu_version=torch_npu.__version__, physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        numerical_execution=numerical, target_rewrite=target, graph_breaks=breaks, state=state,
        product_candidate=candidate,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        product_gate_bypassed=False, body_or_assertions_modified=False,
        adapter_deviation=['仅提取原方法，CUDA专用类装饰器改为真实NPU环境检查', '设备字面量cuda转npu',
            '只读replacement、输入/输出/eager与graph-break观察；不修改产品注册'],
        performance_gate_issued=False)
    (out/'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)
    raise SystemExit(0 if valid else 1)


if __name__ == '__main__':
    main()
