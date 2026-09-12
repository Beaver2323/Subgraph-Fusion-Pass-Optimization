#!/usr/bin/env python3
"""T-087：复用冻结社区 _train_once，一次进程只执行一个门禁状态。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path

SOURCE = Path('/home/z50063656/Pass/src/pytorch/test/inductor/test_reorder_for_locality_in_training.py')
COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('off', 'on', 'master-off'), required=True)
    p.add_argument('--artifact-dir', type=Path, required=True)
    args = p.parse_args()
    if Path.cwd() != Path('/home/z50063656/tmp'):
        p.error('必须从 /home/z50063656/tmp 启动')
    if os.environ.get('TORCHINDUCTOR_NPU_BACKEND') != 'triton_experimental':
        p.error('导入前必须选择 triton_experimental')
    out = args.artifact_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TORCH_COMPILE_DEBUG='1', TORCH_COMPILE_DEBUG_DIR=str(out/'debug'),
                      TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'), TRITON_CACHE_DIR=str(out/'triton-cache'),
                      TORCHINDUCTOR_FORCE_DISABLE_CACHES='1', TORCHINDUCTOR_COMPILE_THREADS='1')
    import torch
    import torch_npu
    import torch_npu.testing  # 使用官方测试框架适配；不改产品 pass。
    from torch_npu.utils._dynamo import register_inductor_npu, _InductorNpuRegistry
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    assert torch.version.git_version == COMMIT
    torch.npu.set_device(0)
    spec = importlib.util.spec_from_file_location('community_reorder', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    recorder = module._Recorder

    class Observed(recorder):
        def __call__(self, graph):
            before = str(graph)
            super().__call__(graph)
            number = len(records)
            after = str(graph)
            (out/f'target-{number}-before.txt').write_text(before)
            (out/f'target-{number}-after.txt').write_text(after)
            records.append({'before': f'target-{number}-before.txt', 'after': f'target-{number}-after.txt',
                            'changed': before != after})

    module._Recorder = Observed
    params, grads, rec = module._train_once('npu', flag_on=args.mode != 'off', reorder_on=args.mode != 'master-off')
    torch.npu.synchronize()
    if args.mode == 'on':
        assert rec.calls > 0 and rec.moved, 'ON 必须调用且实际移动节点'
    else:
        assert rec.calls == 0, 'OFF/主开关关闭不得调用目标'
    torch.save({'params': {k:v.cpu() for k,v in params.items()}, 'grads': {k:v.cpu() for k,v in grads.items()}}, out/'training_state.pt')
    record = dict(generated_at=datetime.now().astimezone().isoformat(), task_id='T-087',
                  acceptance_unit_id='AU-post-grad-reorder-for-locality', mode=args.mode,
                  backend=_InductorNpuRegistry._loaded_backend, torch_commit=torch.version.git_version,
                  torch_npu_version=torch_npu.__version__, torch_npu_file=torch_npu.__file__,
                  physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                  numerical_execution=True, correctness='awaiting-cross-process-state-comparison',
                  handler_calls=rec.calls, graph_changed=rec.moved, graphs=records,
                  product_gate_bypassed=False, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  adapter_deviation=['设备参数 npu', '原测试内 OFF/ON 拆为独立进程；原 _train_once 完整复用', '只读 Recorder 图采集'],
                  preserved_contract=['模型、[8,16] FP32输入、seed=1234', 'SGD lr=0.1', '前向/反向及 optimizer step', '原门禁设置与目标移动条件'])
    (out/'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
