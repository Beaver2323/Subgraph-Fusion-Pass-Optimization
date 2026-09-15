#!/usr/bin/env python3
"""T-091：保留原社区 stack 测试和断言，仅适配设备入口并记录精确 handler。"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_passes.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    args = parser.parse_args()
    if Path.cwd() != Path('/home/z50063656/tmp') or os.environ.get('TORCHINDUCTOR_NPU_BACKEND') != 'triton_experimental':
        parser.error('必须从 tmp 启动，在 torch 导入前选择 triton_experimental')
    out = args.artifact_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TORCH_COMPILE_DEBUG='1', TORCH_COMPILE_DEBUG_DIR=str(out/'debug'),
                      TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'), TRITON_CACHE_DIR=str(out/'triton-cache'))
    import torch
    import torch_npu
    import torch_npu.testing
    from torch_npu.utils._dynamo import register_inductor_npu, _InductorNpuRegistry
    from torch.testing._internal import triton_utils
    from torch._inductor import pattern_matcher
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    assert torch.version.git_version == '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
    assert torch.npu.is_available()
    torch.npu.set_device(0)
    sys.path.insert(0, str(ROOT/'runners'))
    from native_contract_observer import install
    install(pattern_matcher.GraphPatternEntry, out/'target-observations')
    spec = importlib.util.spec_from_file_location('t091_community', SOURCE)
    community = importlib.util.module_from_spec(spec)
    with mock.patch.object(triton_utils, 'requires_gpu', lambda f: f):
        spec.loader.exec_module(community)
    community.GPU_TYPE = 'npu'
    checks = []

    class NpuCase(community.TestSplitCatFxPasses):
        def assertEqual(self, a, b, *pos, **kw):
            result = super().assertEqual(a, b, *pos, **kw)
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                checks.append({'shape':list(a.shape), 'dtype':str(a.dtype),
                               'actual_device':str(a.device), 'expected_device':str(b.device),
                               'passed':True})
            return result

    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([
        NpuCase('test_stack_normalization_axis_kwarg')]))
    torch.npu.synchronize()
    observations = [json.loads(p.read_text()) for p in (out/'target-observations').glob('*/contract_observation.json')]
    exact = [x for x in observations if x['target'] == 'normalize_stack_default' and x['graph_changed']]
    numerical = any(c['actual_device'].startswith('npu:') and c['expected_device'].startswith('npu:') for c in checks)
    valid = result.wasSuccessful() and result.testsRun == 1 and not result.skipped and bool(exact) and numerical
    sources = {}
    for name, module in list(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if file and name.startswith(('torch._inductor', 'torch_npu._inductor')) and Path(file).is_file():
            sources[str(Path(file).resolve())] = hashlib.sha256(Path(file).read_bytes()).hexdigest()
    record = dict(generated_at=datetime.now().astimezone().isoformat(), task_id='T-091',
                  case_id='REF-stack-axis-normalization-native', acceptance_unit_id='AU-split-cat-normalize-stack-default',
                  source_test='test/inductor/test_split_cat_fx_passes.py::TestSplitCatFxPasses.test_stack_normalization_axis_kwarg',
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  status='community-contract-passed' if valid else 'failed-or-target-missing',
                  tests_ran=result.testsRun, tests_skipped=len(result.skipped),
                  backend=_InductorNpuRegistry._loaded_backend, numerical_execution=numerical,
                  tensor_assertions=checks, exact_handler='normalize_stack_default', observations=observations,
                  pytorch_commit=torch.version.git_version, torch_version=torch.__version__, torch_file=torch.__file__,
                  torch_npu_version=torch_npu.__version__, torch_npu_file=torch_npu.__file__,
                  python=sys.version, python_executable=sys.executable, loaded_source_sha256=sources,
                  environment={k:os.environ.get(k) for k in ('PYTHONPATH','CONDA_PREFIX','VIRTUAL_ENV','ASCEND_HOME_PATH',
                      'ASCEND_RT_VISIBLE_DEVICES','TORCHINDUCTOR_NPU_BACKEND')},
                  physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), device_name=torch.npu.get_device_name(0),
                  body_or_assertions_modified=False, product_gate_bypassed=False, performance_gate_issued=False,
                  adapter_deviation=['GPU_TYPE=npu','真实NPU检查替代requires_gpu装饰器','原类原方法显式unittest入口','只读handler及数值断言记录'])
    (out/'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('status','tests_ran','tests_skipped','numerical_execution','tensor_assertions')},ensure_ascii=False))
    raise SystemExit(0 if valid else 1)


if __name__ == '__main__':
    main()
