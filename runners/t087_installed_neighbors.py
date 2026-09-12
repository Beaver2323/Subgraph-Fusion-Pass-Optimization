#!/usr/bin/env python3
"""T-087 安装态公共 codegen 近邻：原 wrapper 拒绝、双设备与普通编译。"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/z50063656/Pass/src/pytorch/test/distributed/tensor/test_compile_on_one_rank.py')
CASES = {
    'cpp-wrapper-rejected': 'test_cpp_wrapper_under_coor_rejected',
    'fx-wrapper-rejected': 'test_fx_wrapper_under_coor_rejected',
    'two-device-code-identical': 'test_inductor_code_identical_across_devices',
    'ordinary-compile': 'test_ordinary_compile',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--artifact-dir', type=Path, required=True)
    args = p.parse_args()
    if Path.cwd() != Path('/home/z50063656/tmp'):
        p.error('必须从 /home/z50063656/tmp 启动')
    out = args.artifact_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TORCHINDUCTOR_NPU_BACKEND='triton_experimental',
        TORCH_COMPILE_DEBUG='1', TORCH_COMPILE_DEBUG_DIR=str(out/'debug'),
        TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'), TRITON_CACHE_DIR=str(out/'triton-cache'),
        TORCHINDUCTOR_FORCE_DISABLE_CACHES='1', TORCHINDUCTOR_COMPILE_THREADS='1')
    import torch
    import torch_npu
    import torch_npu.testing
    import torch.compiler.config as compiler_config
    from torch.testing._internal.common_utils import TestCase
    from torch._inductor.utils import run_and_get_code
    from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    assert torch.version.git_version == '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
    torch.npu.set_device(0)
    if args.case == 'two-device-code-identical':
        assert torch.npu.device_count() >= 2, '需要两张实际 NPU，不能把 SKIP 当 PASS'
    adapter_path = ROOT/'issues/REF-respecialize-current-device-native/npu_adapter.py'
    spec = importlib.util.spec_from_file_location('t087_device_adapter', adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    cls_source = next(n for n in ast.parse(SOURCE.read_text()).body
                      if isinstance(n, ast.ClassDef) and n.name == 'TestCompileOnOneRankDeviceAsParameter')
    method = CASES[args.case]
    names = {'_coor_inductor_fn', '_assert_no_baked_device', '_inductor_code_on_device'}
    if args.case != 'ordinary-compile':
        names.add(method)
    selected = [n for n in cls_source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == names
    cls_ast = ast.ClassDef(name=cls_source.name, bases=[ast.Name(id='TestCase', ctx=ast.Load())],
                          keywords=[], body=selected, decorator_list=[])
    tree = ast.fix_missing_locations(adapter.DeviceAdapter().visit(ast.Module(body=[cls_ast], type_ignores=[])))
    (out/'adapted_test.py').write_text(ast.unparse(tree)+'\n')
    scope = dict(torch=torch, unittest=unittest, compiler_config=compiler_config,
                 TestCase=TestCase, difflib=difflib)
    exec(compile(tree, str(SOURCE), 'exec'), scope)
    cls = scope[cls_source.name]
    ordinary = []

    def test_ordinary_compile(self):
        # 源于原 _coor_inductor_fn：只关闭 CooR，检查公共 codegen 默认路径。
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=str(dtype)), compiler_config.patch(compile_on_one_rank=False):
                torch._dynamo.reset()
                inp = torch.randn(2, 8, device='npu', dtype=dtype)
                compiled = torch.compile(cls._coor_inductor_fn, backend='inductor', fullgraph=True)
                actual, codes = run_and_get_code(compiled, inp)
                self.assertEqual(actual, cls._coor_inductor_fn(inp))
                self.assertEqual(actual.device.type, 'npu')
                self.assertIn("DeviceProperties(type='npu', index=0", '\n'.join(codes))
                self.assertNotIn('_coor_device_idx', '\n'.join(codes))
                for i, code in enumerate(codes):
                    (out/f'ordinary-{dtype}-{i}.py').write_text(code)
                ordinary.append(dict(dtype=str(dtype), device=str(actual.device), exact_contract_passed=True))

    if args.case == 'ordinary-compile':
        setattr(cls, method, test_ordinary_compile)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([cls(method)]))
    torch.npu.synchronize()
    success = result.wasSuccessful() and result.testsRun == 1 and not result.skipped
    installed = Path(torch_npu.__file__).parent
    files = ['_inductor/triton_experimental/device.py', '_inductor/triton_experimental/codegen/triton.py']
    record = dict(generated_at=datetime.now().astimezone().isoformat(), case=args.case,
        community_method=None if args.case == 'ordinary-compile' else f'{cls_source.name}.{method}',
        origin='community-derived-coor-off' if args.case == 'ordinary-compile' else 'community-native-device-adapted',
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status='passed' if success else 'failed', tests_ran=result.testsRun, tests_skipped=len(result.skipped),
        backend=_InductorNpuRegistry._loaded_backend, pytorch_commit=torch.version.git_version,
        torch_npu_file=torch_npu.__file__, pid=os.getpid(), visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        installed_files={name: hashlib.sha256((installed/name).read_bytes()).hexdigest() for name in files},
        product_candidate=None, product_gate_bypassed=False, ordinary=ordinary)
    (out/'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
