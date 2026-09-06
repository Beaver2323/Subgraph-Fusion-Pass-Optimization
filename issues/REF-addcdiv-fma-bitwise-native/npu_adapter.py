#!/usr/bin/env python3
"""复用上游原方法，只更换测试设备/入口并采集图；保留产品和 addcdiv CUDA/XPU guard。"""

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import unittest


if os.environ.get('TORCHINDUCTOR_NPU_BACKEND') != 'triton_experimental':
    raise RuntimeError('必须在导入 torch 前指定 triton_experimental')
if Path.cwd().resolve() != Path('/home/z50063656/tmp'):
    raise RuntimeError('必须从 /home/z50063656/tmp 启动')

import torch
import torch_npu
import torch_npu.testing
from torch._dynamo.utils import counters
from torch_npu.utils._dynamo import register_inductor_npu, _InductorNpuRegistry


TORCH_NPU_EXPERIMENTAL_SOURCE = Path(
    '/home/z50063656/Pass/src/torch_npu/torch_npu/_inductor/triton_experimental'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--source-fix', action='store_true')
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    assert torch.version.git_version == '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
    torch.npu.set_device(0)
    source_modules = {}
    if args.source_fix:
        if not getattr(sys, '_t078_formal_source_overlay', None):
            raise RuntimeError('formal source sitecustomize overlay 未激活')
        for short_name in ('overrides', 'fx_passes', 'lowering'):
            module = importlib.import_module(
                f'torch_npu._inductor.triton_experimental.{short_name}'
            )
            source_modules[short_name] = str(Path(module.__file__).resolve())
        expected_modules = {
            name: str((TORCH_NPU_EXPERIMENTAL_SOURCE / f'{name}.py').resolve())
            for name in source_modules
        }
        if source_modules != expected_modules:
            raise RuntimeError(
                'formal source overlay 未加载目标工作树: '
                f'actual={source_modules}, expected={expected_modules}'
            )
    source = Path('/home/z50063656/Pass/src/pytorch/test/inductor/test_torchinductor.py')
    spec = importlib.util.spec_from_file_location('t078_upstream_torchinductor', source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream.GPU_TYPE = 'npu'
    method = inspect.unwrap(upstream.CommonTemplate.test_addcdiv_fma_bitwise_equal)
    observations = []

    class NpuCase(upstream.TestCase):
        device = 'npu'
        test_addcdiv = method

        @classmethod
        def setUpClass(cls):
            super().setUpClass()
            # 上游 TestCase 为 CUDA Triton 调试额外打开 dtype/shape static_assert；
            # triton-ascend 3.2.0 无法在编译期判定其中的 shape 表达式，且会把
            # value=2.0 捕获的标量参数检查成指针 dtype。这里只关闭测试插桩，
            # 不改变社区测试体、数值断言或产品 pattern guard。
            cls._stack.enter_context(
                upstream.config.patch(
                    {
                        'test_configs.runtime_triton_dtype_assert': False,
                        'test_configs.runtime_triton_shape_assert': False,
                    }
                )
            )

        def assertEqual(self, actual, expected, *positional, **kwargs):
            if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
                observations.append({'index': len(observations) + 1, 'shape': list(actual.shape),
                                     'dtype': str(actual.dtype), 'actual_device': str(actual.device),
                                     'expected_device': str(expected.device),
                                     'bitwise_equal': bool(torch.equal(actual, expected)),
                                     'max_abs_error': float((actual - expected).abs().max().item()),
                                     'addcdiv_fma_fused': counters['inductor']['addcdiv_fma_fused']})
            return super().assertEqual(actual, expected, *positional, **kwargs)

    suite = unittest.TestSuite([NpuCase('test_addcdiv')])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {
        'tracking_mode': 'source-fix-verification' if args.source_fix else 'adapter',
        'source_test': str(source) + '::CommonTemplate.test_addcdiv_fma_bitwise_equal',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'adapter_deviation': ['GPU_TYPE=npu', '缺少 GPUTests 时直接绑定 CommonTemplate 原方法，去除 CUDA-only 测试包装',
                              '关闭 CUDA 测试基类额外开启、triton-ascend 3.2.0 无法编译的 runtime_triton_dtype_assert/runtime_triton_shape_assert 插桩',
                              '追加逐分支 bitwise/counter 观察，不改变原方法和 assertEqual 判据'],
        'direct_blocker': 'direct-4002o46c：原生 GPUTests 不存在，未进入测试体',
        'backend': _InductorNpuRegistry._loaded_backend, 'product_gate_bypassed': False,
        'formal_source_overlay': args.source_fix,
        'formal_source_modules': source_modules,
        'upstream_addcdiv_device_guard_bypassed': False,
        'runtime_triton_shape_assert': False,
        'runtime_triton_dtype_assert': False,
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped), 'community_assertions_passed': result.wasSuccessful() and not result.skipped,
        'target_codegen_asserted_by_this_case': False,
        'observations': observations,
        'compatibility_verdict': (
            ('source-fix-functional-pass' if args.source_fix else 'community-functional-pass-codegen-covered-by-separate-case')
            if result.wasSuccessful() and len(observations) == 2
            else 'community-functional-fail'
        ),
    }
    result_name = 'source_fix_result.json' if args.source_fix else 'adapter_result.json'
    (args.artifact_dir / result_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if result.wasSuccessful() and result.testsRun == 1 and not result.skipped and len(observations) == 2 else 1


if __name__ == '__main__':
    raise SystemExit(main())
