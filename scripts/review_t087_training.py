#!/usr/bin/env python3
"""对照独立 OFF/ON 社区训练状态并归档文本证据；不签发性能门禁。"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if Path.cwd() != Path('/home/z50063656/tmp'):
        parser.error('必须从 tmp 启动')
    os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'triton_experimental'
    import torch
    from torch.testing._internal.common_utils import TestCase
    check = TestCase()
    results = {m:json.loads((args.run_dir/m/'result.json').read_text()) for m in ('off','on','master-off')}
    assert {r['backend'] for r in results.values()} == {'triton_experimental'}
    assert len({r['source_sha256'] for r in results.values()}) == 1
    assert results['on']['graph_changed'] and results['on']['handler_calls'] > 0
    assert results['off']['handler_calls'] == results['master-off']['handler_calls'] == 0
    states = {m:torch.load(args.run_dir/m/'training_state.pt', map_location='cpu', weights_only=True) for m in results}
    errors = {}
    for mode in ('on','master-off'):
        for kind in ('params','grads'):
            check.assertEqual(set(states['off'][kind]), set(states[mode][kind]))
            for name, tensor in states['off'][kind].items():
                actual = states[mode][kind][name]
                check.assertEqual(actual, tensor)
                errors[f'{mode}/{kind}/{name}'] = (actual-tensor).abs().max().item()
    dest = ROOT/'issues/REF-reorder-locality-training-native/evidence/npu-20260911'
    dest.mkdir(parents=True, exist_ok=True)
    names = {'result.json','fx_graph_readable.py','fx_graph_transformed.py','ir_pre_fusion.txt','ir_post_fusion.txt','output_code.py'}
    inventory = []
    for path in sorted(args.run_dir.rglob('*')):
        if not path.is_file() or not (path.name in names or path.name.startswith('target-') or path.name.endswith('.log')):
            continue
        relative = path.relative_to(args.run_dir)
        target = dest/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        inventory.append({'path':str(target.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    result = dict(task_id='T-087', acceptance_unit_id='AU-post-grad-reorder-for-locality',
                  generated_at=datetime.now().astimezone().isoformat(), backend='triton_experimental',
                  status='community-training-contract-passed-performance-pending',
                  correctness='passed', numerical_execution=True, source_run=str(args.run_dir),
                  arms=results, cross_process_max_abs_errors=errors, artifacts=inventory,
                  comparison_scope='社区 OFF/ON 梯度与 step 后参数，另主开关关闭负例；不是性能结果',
                  product_gate_bypassed=False, performance_gate_issued=False)
    # 不放到 functional/ 下：矩阵不能把初步合同结果自动提升为正式闭环。
    out = ROOT/'results/current/T-087/npu_training_review.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'max_errors':errors,'result':str(out)},ensure_ascii=False))


if __name__ == '__main__':
    main()
