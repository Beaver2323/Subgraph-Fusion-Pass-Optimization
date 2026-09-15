#!/usr/bin/env python3
"""逐例调用冻结的 CUDA attention 社区方法；只适配测试发现和实际设备。"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py')
COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'


def task_for(pattern):
    return f'T-{102 + (pattern - 1) // 5:03}' if pattern < 25 else 'T-107'


def passed_tensor_pairs(a, b, tensor_type, path=()):
    """仅在原assertEqual成功后枚举容器叶子；不新增或改变数值断言。"""
    if isinstance(a, tensor_type) and isinstance(b, tensor_type):
        yield path, a, b
    elif isinstance(a, (list, tuple)) and isinstance(b, type(a)):
        for index, (left, right) in enumerate(zip(a, b)):
            yield from passed_tensor_pairs(left, right, tensor_type, (*path, index))
    elif isinstance(a, dict) and isinstance(b, dict):
        for key in a.keys() & b.keys():
            yield from passed_tensor_pairs(a[key], b[key], tensor_type, (*path, str(key)))


def reviewed_math_codegen(code, observations, pattern):
    """只承接已人工审查的14号SDPA数学展开；不是任意缺失FA符号的通行证。"""
    if pattern != 14:
        return False
    calls = [ast.unparse(node.func) for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)]
    target_changed = any(row.get('target', '').startswith('_sfdp_pattern_14_')
                         and row.get('graph_changed') is True for row in observations)
    return (target_changed and calls.count('extern_kernels.bmm') == 2
            and any('_safe_softmax' in call and call.endswith('.run') for call in calls)
            and 'torch.npu.set_device' in calls
            and not any(call.endswith('.cpu') or call == 'empty_strided_cpu' for call in calls))


def main(pattern):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--registration-candidate', action='store_true')
    parser.add_argument('--allow-reviewed-sdpa-math', action='store_true')
    args = parser.parse_args()
    if pattern not in (*range(1, 16), *range(18, 25), 28, 30):
        parser.error('本入口只执行已取得精确 GPU 归因的编号')
    if args.allow_reviewed_sdpa_math and pattern != 14:
        parser.error('数学展开的代码断言适配仅审查了pattern 14')
    if Path.cwd() != Path('/home/z50063656/tmp') or os.environ.get('TORCHINDUCTOR_NPU_BACKEND') != 'triton_experimental':
        parser.error('必须从 tmp 启动，导入前固定 triton_experimental')
    out = args.artifact_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'harness_source.py').write_bytes(Path(__file__).read_bytes())
    os.environ.update(TORCH_COMPILE_DEBUG='1', TORCH_COMPILE_DEBUG_DIR=str(out/'debug'),
                      TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'), TRITON_CACHE_DIR=str(out/'triton-cache'))
    started = datetime.now().astimezone().isoformat()
    import torch
    import torch_npu
    import torch_npu.testing
    from torch_npu.utils._dynamo import register_inductor_npu, _InductorNpuRegistry
    from torch.testing._internal import inductor_utils, common_cuda
    from torch._inductor import pattern_matcher
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    assert torch.version.git_version == COMMIT
    assert torch.npu.is_available()
    torch.npu.set_device(0)
    from native_contract_observer import install
    install(pattern_matcher.ReplacementPatternEntry, out/'target-observations')
    candidate_records = []
    if args.registration_candidate:
        from attention_registration_candidate import install as install_candidate
        (out/'candidate_source.py').write_bytes(Path(__file__).with_name('attention_registration_candidate.py').read_bytes())
        candidate_records = install_candidate(pattern, out/'registration-candidate')
    spec = importlib.util.spec_from_file_location('attention_community', SOURCE)
    community = importlib.util.module_from_spec(spec)
    # 仅让原 CUDA 类/partialmethod 被定义；不伪装 torch.cuda.is_available，不修改产品 guard。
    with mock.patch.object(inductor_utils, 'HAS_CUDA_AND_TRITON', True), \
         mock.patch.object(inductor_utils, 'GPU_TYPE', 'cuda'), \
         mock.patch.object(common_cuda, 'PLATFORM_SUPPORTS_FUSED_ATTENTION', True):
        spec.loader.exec_module(community)
    # 部分原方法（如8/9/10的checkpoint分支）直接用模块GPU_TYPE创建张量。
    # 类名仍选原_gpu入口，但实际张量设备统一为NPU；不伪装SM80/TF32等CUDA硬件能力。
    community.GPU_TYPE = 'npu'
    checks = []
    integer_checks = []
    codegen_checks = []

    class NpuCase(community.SDPAPatternRewriterGpuTests):
        device = 'npu'

        def assertIn(self, member, container, *pos, **kw):
            if member != 'aten._scaled_dot_product' or not isinstance(container, str):
                return super().assertIn(member, container, *pos, **kw)
            # 原失败已证明CUDA符号断言不适用于NPU；只认实际调用，注释/字符串不能冒充。
            tree = ast.parse(container)
            calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
            accepted = calls & {'torch.ops.npu.npu_fusion_attention_v3.default',
                                'torch.ops.npu.npu_fusion_attention.default'}
            if not accepted and args.allow_reviewed_sdpa_math:
                observed = [json.loads(path.read_text()) for path in
                            (out/'target-observations').glob('*/contract_observation.json')]
                self.assertTrue(reviewed_math_codegen(container, observed, pattern),
                                '必须同时有14号真实改图及已审查的NPU BMM/safe-softmax数学展开')
                codegen_checks.append(dict(original_expected=member,
                    actual_calls=sorted(call for call in calls if call=='extern_kernels.bmm' or call.endswith('.run')),
                    source_sha256=hashlib.sha256(container.encode()).hexdigest(),
                    adapted_expectation='reviewed-npu-sdpa-math-not-fused-attention-kernel'))
                return
            self.assertTrue(accepted, 'CUDA SDPA符号须对应实际NPU fusion-attention调用')
            codegen_checks.append(dict(original_expected=member, actual_calls=sorted(accepted),
                                       source_sha256=hashlib.sha256(container.encode()).hexdigest()))

        def assertEqual(self, a, b, *pos, **kw):
            depth = getattr(self, '_tracker_assert_depth', 0)
            self._tracker_assert_depth = depth + 1
            try:
                result = super().assertEqual(a, b, *pos, **kw)
            finally:
                self._tracker_assert_depth = depth
            if depth:
                return result  # 递归比较仅由最外层成功断言记录，避免重复计数。
            for path, left, right in passed_tensor_pairs(a, b, torch.Tensor):
                checks.append(dict(shape=list(left.shape), dtype=str(left.dtype), actual_device=str(left.device),
                                   expected_device=str(right.device), passed=True, tolerance=kw,
                                   container_path=list(path)))
            if isinstance(a, int) and isinstance(b, int):
                integer_checks.append(dict(actual=a, expected=b))
            return result

    method = f'test_sdpa_rewriter_{pattern}_gpu'
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([NpuCase(method)]))
    torch.npu.synchronize()
    observations = [json.loads(p.read_text()) for p in sorted((out/'target-observations').glob('*/contract_observation.json'))]
    exact = [x for x in observations if re.match(rf'^_sfdp_pattern_{pattern}_', x['target']) and x['graph_changed']]
    native_pass = result.wasSuccessful() and result.testsRun == 1 and not result.skipped
    valid = native_pass and bool(exact)
    sources = {}
    for name, module in list(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if file and name.startswith(('torch._inductor', 'torch_npu._inductor', 'triton')) and Path(file).is_file():
            sources[str(Path(file).resolve())] = hashlib.sha256(Path(file).read_bytes()).hexdigest()
    record = dict(generated_at=datetime.now().astimezone().isoformat(), started_at=started, pid=os.getpid(),
                  task_id=task_for(pattern), case_id=f'REF-sfdp-pattern-{pattern}-native',
                  acceptance_unit_id=f'AU-fuse-attention-sfdp-pattern-{pattern}',
                  source_test=f'test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.{method}',
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  status='community-contract-passed' if valid else 'failed-or-target-missing',
                  tests_ran=result.testsRun, tests_skipped=len(result.skipped),
                  test_errors=[text for _, text in result.errors], test_failures=[text for _, text in result.failures],
                  backend=_InductorNpuRegistry._loaded_backend, native_assertions_passed=native_pass,
                  numerical_execution=bool(checks), tensor_assertions=checks, integer_assertions=integer_checks,
                  tensor_observer_version='2-container-leaves-after-original-assertion',
                  reviewed_math_codegen_adaptation=args.allow_reviewed_sdpa_math,
                  codegen_assertion_adaptations=codegen_checks, numerical_assertions_modified=False,
                  exact_target_observations=len(exact), observations=observations,
                  isolated_registration_candidate=args.registration_candidate,
                  candidate_registration_records=candidate_records,
                  pytorch_commit=torch.version.git_version, torch_version=torch.__version__, torch_file=torch.__file__,
                  torch_npu_version=torch_npu.__version__, torch_npu_file=torch_npu.__file__,
                  python=sys.version, python_executable=sys.executable, loaded_source_sha256=sources,
                  harness_snapshot_sha256=hashlib.sha256((out/'harness_source.py').read_bytes()).hexdigest(),
                  candidate_snapshot_sha256=hashlib.sha256((out/'candidate_source.py').read_bytes()).hexdigest()
                      if args.registration_candidate else None,
                  environment={k:os.environ.get(k) for k in ('PYTHONPATH','CONDA_PREFIX','VIRTUAL_ENV','ASCEND_HOME_PATH',
                      'ASCEND_RT_VISIBLE_DEVICES','TORCHINDUCTOR_NPU_BACKEND')},
                  physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), device_name=torch.npu.get_device_name(0),
                  test_body_modified=False, codegen_assertion_modified=True,
                  body_or_assertions_modified=True, product_gate_bypassed=False, performance_gate_issued=False,
                  adapter_deviation=['仅开启原CUDA测试类定义条件','原类self.device与模块GPU_TYPE均绑定npu','原partialmethod和函数体保留',
                                     '只读replacement边界与已通过原断言记录',
                                     'CUDA生成代码符号断言映射为AST核验的NPU fusion-attention实际调用'],
                  numerical_scope='仅原社区实际比较的分支；dropout未比较输出/梯度时不额外声明正确')
    (out/'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('status','tests_ran','tests_skipped','exact_target_observations')},ensure_ascii=False))
    raise SystemExit(0 if valid else 1)
