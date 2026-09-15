#!/usr/bin/env python3
"""16/29 的显式 GPU 目标补测；复用冻结函数和数值 helper，不是原生测例。"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
SOURCE_SHA256 = '56b2d52db32bf9df7e2cef2a9ad6fc898d0f5882204b8a105fb5c5aa84ff966f'
SOURCE_TESTS = {
    16: 'test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_16_inference_gpu',
    29: 'test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_29_gpu',
}


def extracted_function(source: str, pattern: int) -> tuple[str, str]:
    """锚点不符即拒绝；只改声明的 dropout 概率或 mask 来源。"""
    if pattern not in SOURCE_TESTS:
        raise ValueError('只评审了16/29；17不能强加训练注册')
    module = ast.parse(source)
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef)
               and n.name == 'TestSDPAPatternRewriterTemplate')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == f'_test_sdpa_rewriter_{pattern}')
    original = next(n for n in method.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'dot_prod_attention')
    function = copy.deepcopy(original)
    if pattern == 16:
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == 'torch.nn.functional.dropout']
        if len(calls) != 1:
            raise ValueError('dropout调用数变化')
        probability = next(k for k in calls[0].keywords if k.arg == 'p')
        if not isinstance(probability.value, ast.Constant) or probability.value.value != 0.4:
            raise ValueError('原dropout概率变化')
        probability.value = ast.Constant(1e-12)
    else:
        assignments = [n for n in function.body if isinstance(n, ast.Assign)
                       and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                       and n.targets[0].id == 'attn_mask']
        if (len(assignments) != 1 or not isinstance(assignments[0].value, ast.Call)
                or ast.unparse(assignments[0].value.func) != 'torch.zeros'):
            raise ValueError('原全零mask锚点变化')
        # 与原例相同 [1,1,seq,seq] / dtype；作为闭包张量捕获，不新增 mask 梯度合同。
        assignments[0].value = ast.Name(id='supplement_mask', ctx=ast.Load())
    ast.fix_missing_locations(function)
    return ast.unparse(original) + '\n', ast.unparse(function) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-test', required=True)
    parser.add_argument('--pattern', type=int, choices=tuple(SOURCE_TESTS), required=True)
    args = parser.parse_args()
    if args.source_test != SOURCE_TESTS[args.pattern]:
        parser.error('source-test不属于已审核合同')
    import torch
    if torch.version.git_version != COMMIT or not torch.cuda.is_available() or torch.version.hip:
        parser.error('必须是冻结commit的真实CUDA环境，不允许skip或ROCm替代')
    source_path = Path(torch.__file__).resolve().parents[1] / 'test/inductor/test_fused_attention.py'
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_SHA256:
        parser.error('冻结社区测试源码哈希不匹配')
    original, adapted = extracted_function(source_bytes.decode(), args.pattern)
    root = Path(os.environ['TORCH_COMPILE_DEBUG_DIR'])
    evidence = root / 'target_supplement'
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence/'original_function.py').write_text(original)
    (evidence/'adapted_function.py').write_text(adapted)
    (evidence/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    from torch._inductor import pattern_matcher
    from native_contract_observer import install
    for entry in (pattern_matcher.GraphPatternEntry, pattern_matcher.ReplacementPatternEntry):
        install(entry, root/'native_contract_observer', test_body_modified=True)
    sys.path.insert(0, str(source_path.parent))
    spec = importlib.util.spec_from_file_location('supplement_community_attention', source_path)
    community = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(community)
    checks = []

    class TargetSupplement(community.TestSDPAPatternRewriterTemplate):
        device = 'cuda'

        def test_reviewed_target(self):
            if args.pattern == 16:
                namespace = {'torch': torch}
                exec(compile(adapted, str(evidence/'adapted_function.py'), 'exec'), namespace)
                for batch in (4, 1):
                    torch._dynamo.reset()
                    torch.manual_seed(1234)
                    inputs = [torch.randn(batch, 2, 16, 32, device='cuda') for _ in range(3)]
                    name = '_sfdp_pattern_16' + ('_bs1' if batch == 1 else '') + '_training'
                    self._check_common(namespace['dot_prod_attention'], args1=inputs,
                                       contains=False, has_dropout=True, check_train=True,
                                       override_check_equal=True,
                                       expected_fused_attention_patterns={True: name})
                    checks.append({'batch': batch, 'dtype': 'float32', 'target': name,
                                   'output_checks': 2, 'input_gradient_checks': 3})
            else:
                for dtype in (torch.half, torch.float):
                    for batch in (2, 1):
                        torch._dynamo.reset()
                        torch.manual_seed(1234)
                        inputs = [torch.randn(batch, 8, 4, 16, device='cuda', dtype=dtype)
                                  for _ in range(3)]
                        mask = torch.linspace(-0.25, 0.25, 64, device='cuda', dtype=dtype).reshape(1, 1, 8, 8)
                        namespace = {'torch': torch, 'supplement_mask': mask}
                        exec(compile(adapted, str(evidence/'adapted_function.py'), 'exec'), namespace)
                        name = '_sfdp_pattern_29' + ('_half' if dtype == torch.half else '')
                        name += '_bs1' if batch == 1 else ''
                        names = {False: name + '_inference'}
                        if batch == 2:
                            names[True] = name + '_training'
                        self._check_common(namespace['dot_prod_attention'], args1=inputs,
                                           dtype=dtype, check_train=batch == 2,
                                           expected_fused_attention_patterns=names)
                        checks.append({'batch': batch, 'dtype': str(dtype), 'targets': names,
                                       'output_checks': 2 if batch == 2 else 1,
                                       'input_gradient_checks': 3 if batch == 2 else 0})

    # 只加载新增方法，不自动包含基类上游的其他 test_*。
    suite = unittest.TestSuite([TargetSupplement('test_reviewed_target')])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {
        'status': 'passed' if result.wasSuccessful() and not result.skipped else 'failed',
        'scope': 'reviewed-derived-target-reference-not-original-community-test',
        'pattern': args.pattern, 'source_test': args.source_test, 'pytorch_commit': COMMIT,
        'source_sha256': SOURCE_SHA256, 'checks_completed': checks,
        'product_gate_bypassed': False, 'registrations_modified': False,
        'test_body_modified': True, 'atol': 1e-3, 'rtol': 0.2,
        'oracle': '冻结社区_check_common逐输出与Q/K/V输入梯度；不检查mask梯度',
        'limitation': ('极低非零dropout用于目标归因与数值回归，不证明原p=0.4随机分布等价'
                       if args.pattern == 16 else '非平凡mask扩展，不回填原全零mask运行'),
    }
    (evidence/'supplement_result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
