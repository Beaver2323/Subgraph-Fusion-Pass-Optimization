#!/usr/bin/env python3
"""T-087 原生 CooR 单例的 AST 设备适配；冻结源文件不修改。"""
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

SOURCE = Path('/home/z50063656/Pass/src/pytorch/test/distributed/tensor/test_compile_on_one_rank.py')
COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
METHODS = {'_coor_inductor_fn', 'test_inductor_compiles_under_coor', '_assert_no_baked_device'}


class DeviceAdapter(ast.NodeTransformer):
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if node.attr == 'cuda':
            node.attr = 'npu'
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(node.value.replace('cuda', 'npu').replace('CUDA', 'NPU')), node)
        return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--candidate-device', type=Path, help='只在本进程加载独立源码副本的device.py；不修改安装环境')
    args = parser.parse_args()
    if Path.cwd() != Path('/home/z50063656/tmp'):
        parser.error('必须从 tmp 启动')
    output = args.artifact_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'triton_experimental'
    os.environ['TORCH_COMPILE_DEBUG'] = '1'
    os.environ['TORCH_COMPILE_DEBUG_DIR'] = str(output/'debug')
    os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(output/'inductor-cache'), TRITON_CACHE_DIR=str(output/'triton-cache'),
                      TORCHINDUCTOR_FORCE_DISABLE_CACHES='1', TORCHINDUCTOR_COMPILE_THREADS='1')
    import torch
    import torch_npu
    import torch_npu.testing
    import torch.compiler.config as compiler_config
    from torch.testing._internal.common_utils import TestCase
    from torch._inductor.fx_passes import post_grad
    from torch._inductor import utils
    from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu
    register_inductor_npu()
    if torch.version.git_version != COMMIT or _InductorNpuRegistry._loaded_backend != 'triton_experimental':
        raise RuntimeError('冻结源码或后端不符')
    candidate = None
    if args.candidate_device:
        path = args.candidate_device.resolve(strict=True)
        if not path.is_relative_to(Path('/home/z50063656/Pass/worktrees')) or path.name != 'device.py':
            raise ValueError('候选必须是明确的独立源码副本 device.py')
        spec = importlib.util.spec_from_file_location('torch_npu._inductor.triton_experimental._candidate_device', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.register_device_op_overrides_for_npu()
        candidate = {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'scope':'process-local-source-candidate', 'installed_environment_modified':False}
        codegen = path.parent/'codegen/triton.py'
        codegen_spec = importlib.util.spec_from_file_location('torch_npu._inductor.triton_experimental.codegen._candidate_triton', codegen)
        codegen_module = importlib.util.module_from_spec(codegen_spec)
        codegen_spec.loader.exec_module(codegen_module)
        from torch._inductor.codegen.common import register_backend_for_device
        from torch_npu._inductor.triton_experimental.codegen.wrapper import NPUWrapperCodeGen
        register_backend_for_device('npu', codegen_module.NPUTritonScheduling, NPUWrapperCodeGen)
        candidate['codegen'] = {'path':str(codegen), 'sha256':hashlib.sha256(codegen.read_bytes()).hexdigest()}
    torch.npu.set_device(0)
    tree = ast.parse(SOURCE.read_text())
    original_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TestCompileOnOneRankDeviceAsParameter')
    selected = [n for n in original_class.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in METHODS]
    if {n.name for n in selected} != METHODS:
        raise RuntimeError('冻结社区方法结构变化')
    adapted_class = ast.ClassDef(name=original_class.name, bases=[ast.Name(id='TestCase', ctx=ast.Load())], keywords=[], body=selected, decorator_list=[])
    module = ast.fix_missing_locations(DeviceAdapter().visit(ast.Module(body=[adapted_class], type_ignores=[])))
    (output/'adapted_test.py').write_text(ast.unparse(module)+'\n')
    namespace = {'torch':torch, 'unittest':unittest, 'compiler_config':compiler_config, 'TestCase':TestCase}
    exec(compile(module, str(SOURCE), 'exec'), namespace)
    cls = namespace[original_class.name]
    state = {'handler_calls':0, 'graphs':[], 'numeric_compared':False, 'codes':[]}
    original_handler = post_grad.respecialize_current_device_nodes
    original_run = utils.run_and_get_code

    @functools.wraps(original_handler)
    def observe(graph):
        index = state['handler_calls']
        state['handler_calls'] += 1
        before = graph.python_code('self').src
        (output/f'target-{index}-before.txt').write_text(before)
        value = original_handler(graph)
        after = graph.python_code('self').src
        (output/f'target-{index}-after.txt').write_text(after)
        state['graphs'].append({'changed':before != after})
        return value

    @functools.wraps(original_run)
    def capture(compiled, *inputs, **kwargs):
        result, codes = original_run(compiled, *inputs, **kwargs)
        for index, code in enumerate(codes):
            (output/f'output_code_{index}.py').write_text(code)
            state['codes'].append(f'output_code_{index}.py')
        TestCase().assertEqual(result, cls._coor_inductor_fn(*inputs))
        state['numeric_compared'] = True
        return result, codes

    with mock.patch.object(post_grad, 'respecialize_current_device_nodes', observe), mock.patch.object(utils, 'run_and_get_code', capture):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([cls('test_inductor_compiles_under_coor')]))
    torch.npu.synchronize()
    passed = result.wasSuccessful() and result.testsRun == 1 and not result.skipped and state['numeric_compared'] and any(g['changed'] for g in state['graphs'])
    record = dict(task_id='T-087', case_id='REF-respecialize-current-device-native', acceptance_unit_id='AU-post-grad-respecialize-current-device',
                  generated_at=datetime.now().astimezone().isoformat(), pid=os.getpid(),
                  status='community-contract-passed' if passed else 'failed-contract', tests_ran=result.testsRun, tests_skipped=len(result.skipped),
                  backend=_InductorNpuRegistry._loaded_backend, pytorch_commit=torch.version.git_version,
                  torch_npu_version=torch_npu.__version__, torch_npu_file=torch_npu.__file__,
                  physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), state=state, product_candidate=candidate,
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  product_gate_bypassed=False, numerical_execution=state['numeric_compared'],
                  adapter_deviation=['只抽取当前case及两个原helper，设备API/字面量/正则cuda转npu', '补同输入eager数值检查', '只读目标前后图及生成代码采集'],
                  preserved_contract=['原factory/reduction函数与[2,8]输入', 'compile_on_one_rank=True、fullgraph=True', '输出设备/current_device及三种无固化index断言'],
                  performance_status='exempt-no-legal-off-path')
    (output/'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
