"""Conv-BN/Linear folding：调用原社区完整方法，不缩减参数乘积。"""
from __future__ import annotations
import argparse
from datetime import datetime
import functools
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOURCES = Path('/home/z50063656/Pass/src/pytorch')


def main(task):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--conv-hf32', choices=('default','off'), default='default')
    args = parser.parse_args()
    if task not in ('T-098','T-100') or Path.cwd() != Path('/home/z50063656/tmp'):
        parser.error('只允许指定任务并从tmp启动')
    if args.conv_hf32 != 'default' and task != 'T-098':
        parser.error('HF32诊断适配只属于T-098')
    if os.environ.get('TORCHINDUCTOR_NPU_BACKEND') != 'triton_experimental':
        parser.error('导入前必须固定triton_experimental')
    plan = json.loads((ROOT/f"upstream/{task.lower().replace('-','')}_reference_plan.yaml").read_text())
    case = plan['cases'][0]
    file, method = case['source_test'].split('::')
    clsname, method = method.split('.')
    source = SOURCES/file
    out = args.artifact_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'harness_source.py').write_bytes(Path(__file__).read_bytes())
    os.environ.update(TORCH_COMPILE_DEBUG='1',TORCH_COMPILE_DEBUG_DIR=str(out/'debug'),
                      TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'),TRITON_CACHE_DIR=str(out/'triton-cache'))
    import torch
    import torch_npu
    import torch_npu.testing
    from torch_npu.utils._dynamo import register_inductor_npu, _InductorNpuRegistry
    from torch.testing._internal import inductor_utils
    from torch._inductor.pattern_matcher import GraphPatternEntry
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    assert torch.version.git_version == plan['manifest']['pytorch_commit']
    assert torch.npu.is_available()
    torch.npu.set_device(0)
    initial_conv_hf32 = torch.npu.conv.allow_hf32
    initial_matmul_hf32 = torch.npu.matmul.allow_hf32
    if args.conv_hf32 == 'off':
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False
    handlers = ({'efficient_conv_bn_eval_graph_transform_inlined','efficient_conv_bn_eval_graph_transform_decomposed'}
                if task == 'T-098' else {'folded_op'})
    records, tensors, integers = [], [], []
    current_combination = None
    def checkpoint_progress():
        payload=dict(generated_at=datetime.now().astimezone().isoformat(),status='running-not-a-verdict',
                     task_id=task,passed_tensor_assertions=len(tensors),passed_integer_assertions=len(integers),
                     target_handler_returns=len(records),last_tensor=tensors[-1] if tensors else None,
                     precision_mode=args.conv_hf32,full_contract_passed=False,
                     completed_combinations=len(integers),
                     expected_combinations=112 if task=='T-098' else 176,
                     current_combination=current_combination,
                     pid=os.getpid(),physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
        temporary=out/'progress.json.tmp'
        temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
        os.replace(temporary,out/'progress.json')
    original = GraphPatternEntry.apply

    @functools.wraps(original)
    def observed(entry, match, graph, node):
        name = getattr(getattr(entry,'handler',None),'__name__','')
        if name not in handlers:
            return original(entry,match,graph,node)
        before = graph.python_code('self').src
        ret = original(entry,match,graph,node)
        after = graph.python_code('self').src
        i = len(records)
        (out/f'target-{i}-before.txt').write_text(before)
        (out/f'target-{i}-after.txt').write_text(after)
        records.append(dict(handler=name,changed=before!=after))
        return ret

    GraphPatternEntry.apply = observed
    spec = importlib.util.spec_from_file_location('community_module_folding',source)
    community = importlib.util.module_from_spec(spec)
    with mock.patch.object(inductor_utils,'HAS_GPU',True), mock.patch.object(inductor_utils,'GPU_TYPE','cuda'):
        spec.loader.exec_module(community)

    class NpuCase(getattr(community,clsname)):
        device = 'npu'
        def assertEqual(self,a,b,*pos,**kw):
            nonlocal current_combination
            ret = super().assertEqual(a,b,*pos,**kw)
            # 只读取原社区调用帧，便于长测监控定位当前组合，不改循环或输入。
            frame = inspect.currentframe().f_back
            try:
                if task=='T-098' and frame.f_code.co_filename == str(source):
                    values = frame.f_locals
                    if {'test_class','module','use_bias','sync_bn','decompose_nn_module'} <= values.keys():
                        current_combination = {
                            'test_class':values['test_class'].__name__,
                            'module':values['module'][0].__name__,
                            'use_bias':values['use_bias'],'sync_bn':values['sync_bn'],
                            'decompose_nn_module':values['decompose_nn_module'],
                        }
            finally:
                del frame
            if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
                tensors.append(dict(shape=list(a.shape),dtype=str(a.dtype),actual_device=str(a.device),
                                    expected_device=str(b.device),tolerance=kw,passed=True))
            elif type(a) is int and type(b) is int:
                integers.append(dict(actual=a,expected=b))
            checkpoint_progress()
            return ret

    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([NpuCase(method)]))
    torch.npu.synchronize()
    valid = result.wasSuccessful() and result.testsRun == 1 and not result.skipped and any(x['changed'] for x in records)
    source_hashes = {}
    for name, module in list(sys.modules.items()):
        file = getattr(module,'__file__',None)
        if file and name.startswith(('torch._inductor','torch_npu._inductor','triton')) and Path(file).is_file():
            source_hashes[str(Path(file).resolve())] = hashlib.sha256(Path(file).read_bytes()).hexdigest()
    record = dict(generated_at=datetime.now().astimezone().isoformat(),task_id=task,case_id=case['case_id'],
        acceptance_unit_id=case['acceptance_unit_id'],source_test=case['source_test'],
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        status='community-contract-passed' if valid else 'failed-or-target-missing',
        tests_ran=result.testsRun,tests_skipped=len(result.skipped),
        test_errors=[text for _,text in result.errors],test_failures=[text for _,text in result.failures],
        backend=_InductorNpuRegistry._loaded_backend,numerical_execution=bool(tensors),
        tensor_assertions=tensors,integer_assertions=integers,handler_records=records,
        native_assertions_passed=result.wasSuccessful(),expected_match=case['expected_match'],
        physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),pid=os.getpid(),
        pytorch_commit=torch.version.git_version,torch_npu_version=torch_npu.__version__,
        python_executable=sys.executable,loaded_source_sha256=source_hashes,
        harness_snapshot_sha256=hashlib.sha256((out/'harness_source.py').read_bytes()).hexdigest(),
        adapter_deviation=['仅启用原CUDA复制类定义','原类self.device=npu','只读精确handler和原断言记录'],
        body_or_assertions_modified=False,product_gate_bypassed=False,performance_gate_issued=False,
        precision_contract={'initial_conv_hf32':initial_conv_hf32,
            'initial_matmul_hf32':initial_matmul_hf32,
            'effective_conv_hf32':torch.npu.conv.allow_hf32,'matmul_hf32':torch.npu.matmul.allow_hf32,
            'mode':'cuda-tf32-off-analogue' if args.conv_hf32=='off' else 'original-npu-default',
            'configuration_adapted':args.conv_hf32=='off','native_tolerance_modified':False,
            'cuda_tf32_on_arm_covered':False},
        numerical_scope='保留原社区断言；Conv-BN通过SGD后输出间接检查梯度，不冒充逐参数梯度oracle')
    (out/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('status','tests_ran','tests_skipped')},ensure_ascii=False))
    raise SystemExit(0 if valid else 1)
