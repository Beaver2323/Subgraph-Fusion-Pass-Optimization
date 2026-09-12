#!/usr/bin/env python3
"""保留 split/cat 社区测试方法与断言，仅适配 GPU harness 到 NPU 并观察精确 handler。"""
from __future__ import annotations
import argparse
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
SOURCE = Path('/home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py')
TARGETS = {
    'REF-select-cat-aten-native': ('select_cat_aten_pass', 'merge_select_cat_aten'),
    'REF-split-cat-aten-native': ('split_cat_aten_pass', 'merge_split_cat_aten'),
    'REF-split-cat-aten-singular-native': ('split_cat_aten_pass', 'merge_split_cat_aten'),
    'REF-move-view-after-cat-aten-native': ('move_view_after_cat_aten_pass', 'move_view_after_cat'),
    'REF-normalize-cat-default-aten-native': ('normalization_aten_pass', 'normalize_cat_default_aten'),
}


def rewrite_contract(records, expected_match):
    """正例必须改图；负例允许进入handler后被guard拒绝，不能强迫负例改图。"""
    return any(record['changed'] for record in records) == expected_match


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=('T-088','T-089','T-090'), required=True)
    p.add_argument('--case', choices=TARGETS, required=True)
    p.add_argument('--artifact-dir', type=Path, required=True)
    args = p.parse_args()
    if Path.cwd() != Path('/home/z50063656/tmp') or os.environ.get('TORCHINDUCTOR_NPU_BACKEND') != 'triton_experimental':
        p.error('必须从 tmp 启动并在导入前选择 triton_experimental')
    plan = json.loads((ROOT/f"upstream/{args.task.lower().replace('-','')}_reference_plan.yaml").read_text())
    case = next(c for c in plan['cases'] if c['case_id']==args.case)
    method = case['source_test'].split('::')[1].split('.')[1]
    out = args.artifact_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TORCH_COMPILE_DEBUG='1', TORCH_COMPILE_DEBUG_DIR=str(out/'debug'),
                      TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'), TRITON_CACHE_DIR=str(out/'triton-cache'),
                      TORCHINDUCTOR_FORCE_DISABLE_CACHES='1', TORCHINDUCTOR_COMPILE_THREADS='1')
    import torch
    import torch_npu
    import torch_npu.testing
    from torch_npu.utils._dynamo import register_inductor_npu, _InductorNpuRegistry
    from torch.testing._internal import triton_utils
    from torch._inductor.fx_passes import split_cat
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    assert torch.version.git_version == '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
    assert torch.npu.is_available()
    torch.npu.set_device(0)
    spec = importlib.util.spec_from_file_location('community_split_cat_aten', SOURCE)
    upstream = importlib.util.module_from_spec(spec)
    # 只替换本测试模块导入时的 GPU harness 装饰器；config/数值/计数装饰器和方法不变。
    with mock.patch.object(triton_utils, 'requires_gpu_and_triton', lambda f:f):
        spec.loader.exec_module(upstream)
    upstream.GPU_TYPE = 'npu'
    pass_name, symbol = TARGETS[args.case]
    records, assertions = [], []
    seen = 0
    for entries in split_cat.POST_GRAD_PATTERNS[pass_name].patterns.values():
        for entry in entries:
            original = getattr(entry, 'handler', None)
            if getattr(original, '__name__', '') != symbol:
                continue
            seen += 1
            @functools.wraps(original)
            def observed(*pos, __original=original, **kw):
                graph = pos[0].graph
                before = graph.python_code('self').src
                result = __original(*pos, **kw)
                after = graph.python_code('self').src
                i = len(records)
                (out/f'target-{i}-before.txt').write_text(before)
                (out/f'target-{i}-after.txt').write_text(after)
                records.append({'handler':symbol, 'changed':before!=after})
                return result
            entry.handler = observed
    assert seen == 1, f'目标注册数不符：{seen}'
    class NpuCase(upstream.TestSplitCatAten):
        def assertEqual(self, a, b, *pos, **kw):
            if type(a) is int and type(b) is int:
                assertions.append({'actual':a,'expected':b})
            return super().assertEqual(a,b,*pos,**kw)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([NpuCase(method)]))
    torch.npu.synchronize()
    valid = result.wasSuccessful() and result.testsRun==1 and not result.skipped and rewrite_contract(records, case['expected_match'])
    record = dict(generated_at=datetime.now().astimezone().isoformat(), task_id=args.task, case_id=args.case,
                  acceptance_unit_id=case['acceptance_unit_id'], source_test=case['source_test'],
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  status='community-contract-passed' if valid else 'failed-or-target-missing',
                  tests_ran=result.testsRun, tests_skipped=len(result.skipped),
                  backend=_InductorNpuRegistry._loaded_backend, numerical_execution=True,
                  exact_handler=symbol, handler_records=records, integer_assertions=assertions,
                  expected_match=case['expected_match'],
                  physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                  pytorch_commit=torch.version.git_version, torch_npu_version=torch_npu.__version__,
                  adapter_deviation=['GPU_TYPE=npu','以真实NPU backend可用检查代替GPU专用harness装饰器','只读精确handler观察'],
                  body_or_assertions_modified=False, product_gate_bypassed=False,
                  performance_gate_issued=False)
    (out/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
    raise SystemExit(0 if valid else 1)


if __name__ == '__main__':
    main()
