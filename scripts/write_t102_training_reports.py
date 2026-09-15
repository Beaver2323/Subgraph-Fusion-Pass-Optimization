#!/usr/bin/env python3
"""从已验签部署生成 T-102 自包含修复说明，不自行宣称性能或社区合入。"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from validate_t102_training_deployment import verify

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS = {
    1:'softmax(QKᵀ / scale) @ V：将除法缩放的显式 attention 子图替换为 SDPA。',
    2:'softmax(QKᵀ * scale) @ V：识别乘法缩放形式，减少显式 attention 中间张量。',
    3:'softmax(QKᵀ / scale) → dropout → V：识别带随机丢弃的训练 attention。',
    4:'softmax(QKᵀ * scale) → dropout → V：识别乘法缩放、dropout 的训练形式。',
    5:'softmax(QKᵀ / scale + mask) @ V：把加性 mask attention 表达为 SDPA，由后端选择具体实现。'}


def read(p):
    return json.loads(p.read_text())


def put(path, content):
    if path.exists() and path.read_text()!=content:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        old=path.with_name(path.stem+'-历史-'+digest+path.suffix)
        if not old.exists():
            old.write_bytes(path.read_bytes())
    path.write_text(content)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--product-commit', required=True)
    p.add_argument('--product-branch', required=True)
    a=p.parse_args()
    assert Path.cwd()==Path('/home/z50063656/tmp')
    dep=ROOT/'issues/REF-sfdp-pattern-1-native/deployment-20260915-training/deployment.json'
    reference={'path':str(dep.relative_to(ROOT)), 'sha256':hashlib.sha256(dep.read_bytes()).hexdigest()}
    assert verify(ROOT,reference)['installed_passed']
    d=read(dep)
    stages=read(ROOT/'results/current/T-102/npu_stage_reviews.json')['units']
    performance_path=ROOT/'results/current/T-102/performance_summary.json'
    performances={row['unit']:row for row in read(performance_path)['acceptance_units']} if performance_path.exists() else {}
    source=(ROOT/d['source_files']['sfdp_training.py']['after']['path']).read_text()
    upstream_source=subprocess.run(['git','-C','/home/z50063656/Pass/src/pytorch','show',
        '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b:torch/_inductor/fx_passes/fuse_attention.py'],
        check=True,capture_output=True,text=True).stdout
    upstream_nodes={node.name:node for node in ast.parse(upstream_source).body if isinstance(node,ast.FunctionDef)}
    now=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M CST（UTC+08:00）')
    for n in range(1,6):
        issue=ROOT/f'issues/REF-sfdp-pattern-{n}-native'
        rawpath=ROOT/d['installed_cases'][str(n)]['path']; raw=read(rawpath)
        candidate=ROOT/d['candidate_cases'][str(n)]['path']
        baseline=ROOT/stages[f'AU-fuse-attention-sfdp-pattern-{n}']['baseline']['path']
        parents=[r for r in (issue/'adapter_runs').glob('*/run_result.json')
                 if Path(read(r)['raw_artifact_dir']).name==rawpath.parent.parent.name]
        assert len(parents)==1
        command=shlex.join(read(parents[0])['command'])
        relative=lambda path: os.path.relpath(path,issue)
        perf=performances.get(f'pattern-{n}')
        perf_text='性能仍需独立同卡 OFF/ON 验证和六臂计时，不能用原方法执行时间替代。'
        next_action='合法 OFF/ON 和六臂性能处置；随后冻结部分交付。'
        if perf:
            rows=[]
            for clock in ('host_ms','event_ms'):
                for percentile in ('p50','p99'):
                    timing=perf['timing'][clock]
                    rows.append(f"| {clock} {percentile} | {timing['off'][percentile]:.6f} | {timing['on'][percentile]:.6f} | {perf['improvement_percent'][clock][percentile]:+.2f}% |")
            perf_text=(f"已完成独立功能门禁和六臂计时，结论 **{perf['verdict']}**。正改善率表示时延下降，负值表示回退。\n\n"
                       '| 时钟/分位 | OFF ms | ON ms | 改善率 |\n|---|---:|---:|---:|\n'+'\n'.join(rows)+
                       '\n\n同 NPU 5、六个新进程，顺序 OFF1/ON1/ON2/OFF2/OFF3/ON3；每臂预热10次、host及Event各100样本。'
                       '汇总取三轮各臂p50/p99的中位数，原样本、波动和compile_ms保留。compile_ms起点位于先行lazy_init之后，不是包含训练注册trace的完整冷启动成本。'
                       f"[功能门禁](../../results/current/T-102/performance_gates/pattern-{n}-executed.json)、"
                       f"[完整验签包](../../results/current/T-102/attention_bundles/pattern-{n}.json)、"
                       '[时延原始记录](../../results/current/T-102/performance_summary.json)。')
            next_action='本单元已闭环；部分交付保留性能结论与未覆盖域，产品 PR/CI 另行处理。'
        dtypes=', '.join(sorted({r['dtype'] for r in raw['tensor_assertions']})) or '原 dropout 方法未比较 Tensor'
        scope=('原方法没有随机输出/梯度数值 oracle。当前仅原执行/改写合同通过，低非零 dropout 性能微图的数值验证另列。'
               if n in (3,4) else f'原方法 {len(raw["tensor_assertions"])} 次输出/梯度比较通过，dtype 范围：{dtypes}。')
        special=('全零 bool mask 的四次改写由 pattern 1 承载，不计成本编号；本编号八次改写，六组合同总计三十次 Tensor 比较。带 mask 路径允许 NPU BMM/safe-softmax 数学展开，不等于 FA kernel 融合。'
                 if n==5 else '推理时 dropout=0 由相邻无 dropout 编号承载；本编号只统计训练的两次改写。'
                 if n in (3,4) else '按本编号实际改写计数，不以总 fuse_attention counter 替代精确归因。')
        text=f'''# T-102 pattern {n}：训练注册修复与学习说明

> 更新时间：{now}。安装态验证通过；性能以 results/current/T-102 的最终记录为准。

## 问题描述与 pattern 意图

{DESCRIPTIONS[n]}

旧安装态训练图使用 NPU matmul_backward、mul/sub softmax 梯度（dropout 例还含 packed-mask NPU dropout），无法匹配 CUDA 预生成训练注册。原例在 `fuse_attention >= 1` 失败，不是把数值失败改成宽容差通过。

## 调用栈与阶段

```text
# 冻结 test/inductor/test_fused_attention.py，源调用链重建；实际异常见原复现报告
test_sdpa_rewriter_{n}_gpu → _check_common → torch.compile(fullgraph=True)
 → AOTAutograd joint graph → joint_graph.lazy_init → fuse_attention._sfdp_init
 → gen_register_replacement → ReplacementPatternEntry → SDPA
 → NPU lowering/codegen → 设备前向/反向 → 原断言
```

注册及图改写发生在 lowering 和内核选择之前。最终是 FA、外部 NPU BMM 还是 Triton 数学展开，需要读取 output_code，不能仅凭 FX 改写判断。

## 修复内容摘要、修改清单与关键代码

- 新增 `torch_npu/_inductor/triton_experimental/sfdp_training.py`，仅已审查 1～5 号实际 NPU training 按当前 decomposition 重建匹配图。
- 同后端 `__init__.py::_activate` 新增导入及调用。原推理、其他编号、CPU/CUDA、mixed-device 调用原入口；幂等激活。
- 保留 search/replacement、extra_check、scalar_workaround、去重及 tracing 参数；不恢复 FMA，不修改上游、默认禁用配置或精度阈值。
- 无 C/C++ 变更，不需重编译二进制；未验证独立 wheel 发布。

```python
# torch_npu/_inductor/triton_experimental/sfdp_training.py，实际部署源码
{source.rstrip()}
```

实际备份及文件 SHA256 见[部署记录](../REF-sfdp-pattern-1-native/deployment-20260915-training/部署记录.md)，机器可读闭环为[deployment.json](../REF-sfdp-pattern-1-native/deployment-20260915-training/deployment.json)。T-106 codegen 修复原样保留。

## 验证结果与 GPU/NPU 对比

| 阶段 | 结论 | 原件 |
|---|---|---|
| GPU/reference | 冻结 8e86e0a 原方法及本编号已复核 | [GPU review](../../results/current/T-102/gpu_reference_review.json) |
| NPU 修复前 | 原训练合同 FAIL | [原 result]({relative(baseline)}) |
| 隔离共同候选 | 原方法 PASS，不算安装态 | [候选 result]({relative(candidate)}) |
| 安装态无候选新进程 | tests=1、skip=0、本编号改写 {raw['exact_target_observations']} 次 | [新 result]({relative(rawpath)}) |
| 部署前/后设备边界 | 各 6 场景、25 次输出/梯度比较；负例不命中 | [全部边界](../REF-sfdp-pattern-1-native/deployment-20260915-training/) |

{scope}

{special}

原方法数值与结构结论不外推 BF16、所有 shape、分布式或完整模型训练。新增 FP16 非连续布局边界属于本项目派生测试，不冒称社区原方法的参数。

## 原用例复验命令与关键输出

```bash
cd /home/z50063656/tmp
source /home/z50063656/Pass/activate_pass.sh
# controller 导入前指定 experimental，ASCEND_RT_VISIBLE_DEVICES=5；下列为实际子进程命令
{command}
```

[完整父命令/哈希/退出码及 stdout/stderr]({relative(parents[0].parent)}/)。返回码 0，`Ran 1 test` / `OK`，不含候选开关。

## 近邻与边界命令

```bash
cd /home/z50063656/tmp
ASCEND_RT_VISIBLE_DEVICES=5 TORCHINDUCTOR_NPU_BACKEND=triton_experimental \\
  /home/z50063656/envs/Pass/bin/python {ROOT/'runners/attention_training_boundary.py'} \\
  --output /home/z50063656/tmp/t102-training-boundary-installed-20260915
```

该输出目录已有结果，复跑必须换新目录。边界覆盖 div FP32、mul FP16 strided、add-mask FP32 三个正例，以及中间结果复用、tensor scale、scalar mask 三个负例；部署前后同一 harness，全部比较前向及输入梯度。

## 性能与代码路径范围

{perf_text}

测例来自冻结注册 search/shape/stride/dtype，经确定性初始化；dropout 3/4 保留极低非零 dropout。五个编号均计训练前向，不含 backward、首次编译或完整模型。1号旧推理OFF被3号推理接替，故1/2/5改测本次修复对应training合同；没有关闭邻接、放宽OFF门禁或把整批原方法运行时间作为样本。详见[本批修订合同](../../upstream/t102_performance_plan.yaml)及[原性能合同说明](../../report/attention_performance_contract_review_20260914.md)。

## 代码审查与残余风险

无 CPU 兜底、无新增禁用项、无冻结 PyTorch 修改，修复与注册层根因一致。首次 lazy 注册增加运行时 tracing 开销；当前稳态微基准不涵盖该成本。边界并非全 dtype/随机性穷尽证明。最终性能回退或部分对齐必须保留，不据此擅改默认配置。

本地产品分支 `{a.product_branch}`，提交 `{a.product_commit}`；没有推送产品仓、发起 PR、通过社区 CI 或发布 wheel。Tracker 提交不等于产品合入。

### Handoff

- 阶段结论：原例及共同注册边界通过，安装态加载两份部署源码的 SHA256 已核验。
- 下一步：{next_action}
- 验证覆盖：仅 Python、原例 PASS、共同候选五原例与六项设备边界均通过。
- 残余风险：important，性能和未覆盖输入域不外推。
'''
        put(issue/'训练注册修复报告.md',text)
        put(issue/'修复验证报告.md',text.replace('训练注册修复与学习说明','训练注册修复验证'))
        pr=f'''# [Fix] Align experimental SFDP pattern {n} training registration with NPU decomposition

> 更新时间：{now}。本地验证材料，未提交社区 PR/CI。

## 【合入来源】

T-102 原社区训练 counter 失败。五个编号属于同一后端注册修复，不建议拆成五份重复补丁。

## 【修改方案】

仅 experimental 激活入口及 sfdp_training.py，按实际 NPU decomposition 生成 1～5 training pattern，保留全部原 guard/replacement。修复与代码见[完整报告](训练注册修复报告.md)。

## 【资料变更】

新增复现、根因、前后生成代码、安装回归与性能范围说明。

## 【接口变更】

无客户 API/C++ ABI 变化，无二进制改动；首次 lazy 注册 tracing 成本需注意。

## 【功能验证】

同一候选五个原方法、安装态五个原方法、部署前后各六项设备边界通过。{scope} {special}

## 【CheckList】

- [x] 不增加 CPU fallback、skip、产品禁用绕过或宽松断言。
- [x] 当前后端、冻结源码、精确目标、文件 SHA256 和新进程证据完整。
- [x] 共享注册入口正负例及布局边界验证通过。
- [ ] 社区 CI、产品 PR 合入及 wheel 发布（未执行）。

分支 `{a.product_branch}`，本地提交 `{a.product_commit}`。性能以最终结果为准，不自动宣称收益。
'''
        put(issue/'代码合入描述.md',pr)
        if perf:
            nodes=[upstream_nodes[f'_sfdp_pattern_{n}'],upstream_nodes[f'_sfdp_replacement_{n}']]
            code='\n\n'.join('\n'.join(upstream_source.splitlines()[node.lineno-1:node.end_lineno]) for node in nodes)
            function=read(ROOT/f'results/current/T-102/functional/pattern-{n}.json')
            spec_table='\n'.join(f"| {row['parameter']} | {row.get('shape',row.get('value'))} | {row.get('dtype',row.get('python_type'))} | {row.get('requires_grad','—')} |" for row in function['input_spec'])
            calls=[]
            for mode in ('off','on'):
                raw_function=ROOT/function['source_evidence'][mode]['path']
                for path in sorted((raw_function.parent/'debug').rglob('output_code.py')):
                    selected=[line.strip() for line in path.read_text().splitlines() if
                              not line.lstrip().startswith('#') and any(token in line for token in
                              ('extern_kernels.bmm(', '= torch.ops.npu.', '= torch.ops.aten.matmul_backward'))]
                    calls.append(f"```python\n# {path.relative_to(ROOT)}\n"+'\n'.join(selected[:12])+'\n```')
            learning=f'''# T-102 pattern {n}：功能、修复与性能对照

> 更新时间：{now}。原社区合同、安装修复、精确OFF/ON与性能处置完成；不是产品PR已合入。

## Pattern 意图与代码

{DESCRIPTIONS[n]} 匹配/替换位于 joint FX 阶段，在 lowering、scheduler、autotune 和最终 kernel 选择之前。

```python
# PyTorch 8e86e0a，torch/_inductor/fx_passes/fuse_attention.py:{nodes[0].lineno}
{code}
```

## 功能测例及 GPU/NPU 对比

原入口为 `test/inductor/test_fused_attention.py::SDPAPatternRewriterGpuTests.test_sdpa_rewriter_{n}_gpu`。
GPU使用用户已回传的冻结原方法/只读精确观察包，本轮未重新跑GPU。NPU仅设备及代码符号适配，不更改数值阈值。

| 项目 | GPU reference | NPU experimental |
|---|---|---|
| 原方法与本编号 | 原生通过，有目标边界FX | 修复前训练未命中；新安装态1方法、0skip、本编号{raw['exact_target_observations']}次改写 |
| 输出/梯度 | 以原方法实际断言为界 | {scope} |
| 修复层 | 不修改CUDA或上游 | 活动NPU decomposition重建1～5训练pattern，原replacement/guard保留 |
| 当前统计 | [GPU审查](gpu_reference_review.json) | NEWLY_SUPPORTED；{function['community_alignment']['status']}；性能{perf['verdict']} |

{special} 对齐标签限本次冻结合同，FULL_ALIGNED不代表所有dtype/shape或代码生成逐指令一致；FX改写不自动等于融合FA或性能收益。

[完整修复说明与调用栈](../../../issues/REF-sfdp-pattern-{n}-native/训练注册修复报告.md)包含实际36行部署代码、五原例和六正负边界、前后原件。
[部署证明](../../../issues/REF-sfdp-pattern-1-native/deployment-20260915-training/deployment.json)验签对应两个安装文件与新进程。

## 性能测例来源、输入与范围

没有将已融合SDPA API benchmark冒充本FX rewrite benchmark。复用冻结 `_get_sfdp_patterns` 同编号half training的search、shape、stride和dtype，固定种子normal std0.25替代未初始化empty；保留requires_grad。
3/4使用社区极低非零dropout设计1e-11，不能置0后测邻接；1/2/5无dropout。

| 输入 | shape/标量 | dtype | requires_grad |
|---|---|---|---|
{spec_table}

{'5号另有冻结注册scalar_workaround：inv_scale=0.66666，作为partial绑定常量，不在输入Tensor表内。' if n==5 else ''}

仅training-forward微图；不含backward、首次编译或完整模型。每臂与同卡eager比较，派生域rtol=0.2、atol=2e-3，不回写原社区方法的容差；OFF只删本编号，不关闭整轮joint或邻接。合法OFF的attention计数为0，ON本编号改写>=1。
1号历史推理OFF被3号接替，故改测与本次修复一致的training合同；[失败与修订记录](../../../issues/REF-sfdp-pattern-1-native/性能合同修订说明.md)保留，不借用T-106的特许豁免。

## 实际生成代码对照

下列摘取实际OFF、ON调用行，完整FX/IR/代码位于注释路径；Triton softmax/转换核与缓存分配未全展开，不是完整可执行脚本。

{chr(10).join(calls)}

## 性能结果

{perf_text.replace('../../results/current/T-102/','')}

## 交付边界

原方法未比较的随机梯度、BF16与其他shape不外推；低dropout微图通过不能补写原高dropout数值oracle。局部收益不代表整个训练或默认开关决策。未改产品默认配置；本地产品提交{a.product_commit}尚未推送产品仓/运行社区CI或发布wheel。
'''
            put(ROOT/f'results/current/T-102/pattern-{n}_讲解.md',learning)
    if len(performances)==5:
        table=[]
        for n in range(1,6):
            row=performances[f'pattern-{n}']; rate=row['improvement_percent']
            table.append(f"| [{n}](pattern-{n}_讲解.md) | {row['verdict']} | {rate['host_ms']['p50']:+.2f}% / {rate['host_ms']['p99']:+.2f}% | {rate['event_ms']['p50']:+.2f}% / {rate['event_ms']['p99']:+.2f}% |")
        put(ROOT/'results/current/T-102/README.md',f'''# T-102 交付索引：attention pattern 1～5

> 更新时间：{now}。5/5 原合同、安装态修复、独立OFF/ON和性能处置完成。

## 结论

五个编号原先训练图不能匹配CUDA预生成注册，现在通过活动NPU decomposition重建匹配图，安装态原例均通过；不是仅有隔离候选，也没有修改上游、容差或显式禁用配置。
共同候选五原方法、安装态五原方法、部署前后各六项设备边界通过；原社区每方法均1test、0skip。
3/4原方法没有随机输出/梯度数值oracle，部分对齐继续保留；5号带mask最终是NPU数学展开，不等于融合FA内核。

## 每个pattern的功能、代码与性能

正改善率表示时延下降，负值表示回退；均为half training-forward微图，不含backward、完整冷启动或模型端到端。

| Pattern及讲解 | 性能结论 | host p50 / p99 改善 | Event p50 / p99 改善 |
|---|---|---|---|
{chr(10).join(table)}

本批5项均实测，没有默认关闭免测或特许归因受限。各臂10次预热、host及Event各100样本，六个新进程独占串行；实际OFF无attention、ON本编号精确改图，且数值通过才计时。收益不用于擅改默认配置。

## 证据阅读顺序

1. 上表逐pattern讲解：源码位置、search/replacement、GPU/NPU、实际生成代码、输入与时延。
2. [GPU复核](gpu_reference_review.json)、[功能汇总](npu_functional_summary.json)、[性能汇总及原样本引用](performance_summary.json)。
3. [机器可读验签包](attention_bundles/)及[执行过的功能门禁](performance_gates/)。
4. [共同修复报告](../../../issues/REF-sfdp-pattern-1-native/训练注册修复报告.md)、[部署备份/原例/边界](../../../issues/REF-sfdp-pattern-1-native/deployment-20260915-training/)、[独立产品补丁](../../../issues/REF-sfdp-pattern-1-native/sfdp_training_registration.patch)。其余四个issue也有自包含修复说明。
5. 性能入口真实问题：[1号推理OFF邻接接替](../../../issues/REF-sfdp-pattern-1-native/性能合同修订说明.md)、[scale占位适配](../../../issues/REF-sfdp-pattern-2-native/性能输入适配报告.md)。旧失败不覆盖。

`npu_stage_reviews.json`保存修复阶段证明，并非最新最终判定；最终以本页及functional/attention_bundles为准。
产品分支`{a.product_branch}`、本地提交`{a.product_commit}`，未推送产品仓/发起社区PR/运行社区CI或发布wheel。T-102新部署不自动重新认证其他批次旧源码指纹。

## 离线核验

```bash
cd /home/z50063656/tmp
/home/z50063656/envs/Pass/bin/python {ROOT/'scripts/review_attention_completion.py'} --check-current
```

此命令只读仓库证据和哈希，不导入torch，也不会执行归档生成代码；输出device_execution=false指此次离线检查不使用设备，不否定原件的真机执行。
整体验收和历史再认证边界见[部分交付说明](../../../report/部分交付说明_20260915.md)。
''')
    print(f't102_training_reports=written patterns=5 measured_records={len(performances)}')


if __name__=='__main__':
    main()
