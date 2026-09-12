#!/usr/bin/env python3
"""归档逐例 NPU 合同与阶段进度，不把初步功能通过升级成性能/正式闭环。"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'result.json','fx_graph_readable.py','fx_graph_transformed.py','ir_pre_fusion.txt','ir_post_fusion.txt','output_code.py'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    source=args.run_dir.resolve()
    record=json.loads((source/'result.json').read_text())
    if record['status']!='community-contract-passed' or record['backend']!='triton_experimental' or record['tests_ran']!=1 or record['tests_skipped']!=0:
        raise ValueError('只归档经过原生断言和目标正负判据验证的单例')
    case=record['case_id'];task=record['task_id'];unit=record['acceptance_unit_id']
    case_root=ROOT/'issues'/case
    dest=case_root/'evidence'/source.parent.name/source.name
    inventory=[]
    for path in sorted(source.rglob('*')):
        if not path.is_file() or not (path.name in NAMES or path.name.startswith('target-')):
            continue
        target=dest/path.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        inventory.append({'path':str(target.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    # 首例手动启动时 stdout/stderr 位于 run-dir 旁，仍随证据归档。
    for suffix in ('stdout.log', 'stderr.log'):
        path = source.with_name(source.name + '.' + suffix)
        if path.is_file():
            target = dest / suffix
            shutil.copy2(path, target)
            inventory.append({'path': str(target.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if not any(Path(x['path']).name=='output_code.py' for x in inventory):
        raise ValueError('缺少真实生成代码，不能记录设备合同通过')
    current=datetime.now().astimezone()
    now=current.isoformat()
    display_time=current.strftime('%Y-%m-%d %H:%M')+' CST（UTC+08:00）'
    progress_path=ROOT/'results/current'/task/'npu_contract_progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else dict(task_id=task,units={})
    plan=json.loads((ROOT/f"upstream/{task.lower().replace('-','')}_reference_plan.yaml").read_text())
    data=progress['units'].setdefault(unit,{'cases':{}})
    data['cases'][case]={'result_path':str((dest/'result.json').relative_to(ROOT)),
                         'sha256':hashlib.sha256((dest/'result.json').read_bytes()).hexdigest(),'artifacts':inventory}
    required={c['case_id'] for c in plan['cases'] if c['acceptance_unit_id']==unit}
    data.update(backend='triton_experimental',correctness='passed-for-recorded-cases',
                contract_complete=required<=set(data['cases']),pending_cases=sorted(required-set(data['cases'])),
                fallback_review='manual-review-pending',performance_gate_issued=False)
    progress['updated_at']=now
    progress_path.write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
    report=f'''# {task} {case} 最小适配与复现

> 更新时间：{display_time}
> 单元：{unit}；实际后端：`triton_experimental`。

## 1. 原生阻断

原生预检保存在本目录 native_runs：GPU 专用 requires_gpu_and_triton 不接受 NPU，不能将 skip 当 PASS。当前源码未提供 test_upsteam/disabled_testcases.json；目标未发现 experimental 显式产品 disable。

## 2. 最小适配

```python
# runners/t088_t090_npu_case.py；本目录 npu_adapter.py 固定 case 入口
with mock.patch.object(triton_utils, 'requires_gpu_and_triton', lambda f:f):
    spec.loader.exec_module(upstream)
upstream.GPU_TYPE = 'npu'
# 原方法体、shape/dtype/stride、config patch、counter/数值断言全部保留。
```

原社区入口 `{record['source_test']}`，源码 SHA256 `{record['source_sha256']}`。只替换 GPU harness 的设备可用性检查，实际启动前验证 NPU 可用和冻结 torch commit。只读目标 handler 包装器保存实际调用前后 FX；没有修改 product gate 或 handler 返回值。

## 3. 必要调用链（源码重建）

```text
npu_adapter.py -> 原 TestSplitCatAten 方法 -> torch.compile
 -> post_grad -> {record['exact_handler']}
 -> lowering/codegen -> NPU -> 原数值与计数断言
```

## 4. 结果与边界

tests=1、skip=0；原社区断言通过，目标正负期望和实际改写吻合。整数断言为 `{record['integer_assertions']}`；精确 handler 记录 `{record['handler_records']}`。
实际 torch_npu 版本 `{record['torch_npu_version']}`，物理 NPU `{record['physical_npu']}`。`product_gate_bypassed=false`，没有修改冻结 PyTorch 或 NPU 产品源码。

完整前后 FX、IR、output_code.py 以及 result.json 在 [本例证据目录]({dest.relative_to(case_root).as_posix()}/)；原生和适配 stdout/stderr 保存在 native_runs/adapter_runs（首个手动适配运行的日志随本例证据归档）。此结果只覆盖本 case，不证明未运行的邻接例。对 singular 负例，原社区要求 split counter=0、目标不改图；不得以“所有例必须改图”误判失败。首次收集器曾误用该条件，原失败记录保留，修正收集判据后已经重跑，不是 NPU 产品修复。

## 5. 后续门禁与 Handoff

功能 PASS（当前社区合同）；逐任务进度见 results/current/{task}/npu_contract_progress.json。生成代码已保留，性能前还须完成 fallback/graph-break 人工复核和独立 OFF/ON 测量图门禁。没有签发性能 gate，也没有宣称正式闭环或性能收益。
'''
    (case_root/'适配报告.md').write_text(report,encoding='utf-8')
    (case_root/'复现报告.md').write_text(report,encoding='utf-8')
    print(task,case,'archived',data['contract_complete'],len(inventory))


if __name__=='__main__':
    main()
