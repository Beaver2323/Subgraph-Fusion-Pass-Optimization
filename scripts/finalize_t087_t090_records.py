#!/usr/bin/env python3
"""按已验证原件收束 T-087～T-090 计划和中文讲解；未通过的单元不会被关闭。"""
from datetime import datetime
import json
from pathlib import Path
from review_t087_t090_completion import ROOT, CASES, WORKER, dump, load, require, validate_archived

EXPLANATIONS = {
    "reorder-locality": ("训练局部性重排", "使同一反向分支尽早计算和释放；只调整拓扑合法的顺序，不改变算子数学定义。",
        "torch/_inductor/fx_passes/post_grad.py:193", "reorder_for_locality(graph)  # 只在允许的训练/主开关下调用",
        "社区训练 OFF/ON 验证参数、梯度与节点移动，主开关关闭时零调用。",
        "相同训练合同通过；反向 matmul_backward 在 NPU 保留为图内原生 extern，不能声称 CUDA 相同反向 lowering。"),
    "select-cat-aten": ("连续 select + cat 转 view", "完整连续切片又拼回同一维度时，用 view 代替复制；不完整或不连续切片不改。",
        "torch/_inductor/fx_passes/split_cat.py:1907", "view_node = graph.call_function(torch.ops.aten.view.default, args=(node_input, cat_node.meta['val'].shape))",
        "原生四输出合同通过；只有完整连续分支目标命中。",
        "四输出与数值保持，完整连续分支改成 view；部分分支继续 NPU cat extern。"),
    "split-cat-aten": ("split 连续片段收束", "把足够长的连续 split/getitem 序列合并为原输入或 slice，减少拼接输入及中间描述。",
        "torch/_inductor/fx_passes/split_cat.py:1803", "threshold_to_cat = config.post_grad_fusion_options['split_cat_aten_pass'].get('threshold_to_cat', 10)",
        "两组正例原数值/counter 通过，singular 负例 counter=0。",
        "复用两组正例和 singular 负例；性能只复用第一组 [1024,128]/[1024,32]，不冒充全部 shape 性能。"),
    "move-view-after-cat": ("将 view 移到 cat 之后", "把多个切片 view 的拼接改为先拼接再 view，减少重复视图处理；保留 clone 多用户关系。",
        "torch/_inductor/fx_passes/split_cat.py:2935", "# 语义示意：cat([view(p0), ..., view(p6)]) -> view(cat([p0, ..., p6]))",
        "原 [7,8,96]、clone 与嵌套 cat 的社区合同通过。",
        "精确 handler 改图且原数值通过。实际 OFF 为 3 次 NPU cat extern；ON 为 1 个 Triton 切片/复制核加 2 次 NPU cat extern，并非所有 cat 都被消除。"),
    "normalize-cat-aten": ("ATen cat 参数规范化", "把位置参数、关键字与维度写法规范化，帮助后续 matcher 识别；它不是独立计算核。",
        "torch/_inductor/fx_passes/split_cat.py:1959", "tensors = get_arg_value(cat_node, 0, 'tensors')\ncat_dim = get_arg_value(cat_node, 1, 'dim')",
        "两组 split-cat 社区图提供 normalization 精确处理与数值合同。",
        "精确 normalize_cat_default_aten 改图。性能切换 normalization pass、保留下游 split-cat，收益若存在属于前提使能链，不能全归因于单个参数改写。"),
}


def main():
    if (ROOT/'results/current/T-087/functional/respecialize-current-device.json').exists():
        raise ValueError('已有更新的安装态修复结果，禁止旧五单元收束器覆盖；请保留当前结果并使用安装态复核流程')
    validate_archived()
    now = datetime.now().astimezone()
    stamp = now.strftime('%Y-%m-%d %H:%M CST（UTC+08:00）')
    for task in ('T-087','T-088','T-089','T-090'):
        directory = ROOT/'results/current'/task
        gpu = load(directory/'gpu_reference_review.json')
        require(gpu['status']=='gpu-contract-reviewed-awaiting-npu', '缺少已接受 GPU 合同')
        manifest_path = ROOT/'upstream'/f"{task.lower().replace('-','')}_manifest.yaml"
        plan_path = ROOT/'upstream'/f"{task.lower().replace('-','')}_performance_plan.yaml"
        manifest, plan = load(manifest_path), load(plan_path)
        perf_path = directory/'performance_summary.json'
        summary = load(perf_path) if perf_path.exists() else {'schema_version':'1.0','task_id':task,'backend':'triton_experimental','acceptance_units':[]}
        indexed = {row['acceptance_unit_id']:row for row in summary['acceptance_units']}
        if task == 'T-087':
            au = 'AU-post-grad-respecialize-current-device'
            if au not in indexed:
                item = {'acceptance_unit_id':au, 'performance_status':'exempt-no-legal-off-path', 'verdict':'PERF_EXEMPT',
                    'reason':'必需的 lowering 前设备解析没有合法 OFF；免测不代表功能通过，安装态 wrapper 编译失败另列。',
                    'product_action':'不删除必需 pass 制造 OFF；继续处理 NPU wrapper 接口缺口。'}
                summary['acceptance_units'].append(item)
                indexed[au] = item
        closed = 0
        completed_units = []
        for unit in manifest['acceptance_units']:
            au = unit['acceptance_unit_id']
            name = next((k for k,v in WORKER.TARGETS.items() if v[1]==au), None)
            functional = directory/'functional'/f'{name}.json'
            is_closed = functional.exists() and au in indexed and name in CASES
            closed += int(is_closed)
            unit.update(review_status='frozen', denominator_eligible='yes-frozen',
                        coverage_phase='formally-closed' if is_closed else 'npu-compile-failed-awaiting-product-fix')
            for variant in unit['variants']:
                variant['reference_status'] = 'valid-reference'
                variant['npu_status'] = 'passed' if functional.exists() else 'failed-wrapper-compilation'
            if is_closed:
                completed_units.append({'unit':name, 'acceptance_unit_id':au,
                    'status':'functional-passed-performance-gate-signed',
                    'functional_evidence':str(functional.relative_to(ROOT)),
                    'gate':str((directory/'performance_gates'/f'{name}.json').relative_to(ROOT)),
                    'artifacts':len(load(functional)['artifact_inventory'])})
                title,intent,source,code,gpu_behavior,npu_behavior = EXPLANATIONS[name]
                indexed[au]['pattern_explanation'] = {'name':title,'intent':intent,'source':source,'source_excerpt':code,
                    'gpu_behavior':gpu_behavior,'npu_behavior':npu_behavior}
                report = [f'# {task}：{title} 功能与性能报告', '', f'> 更新时间：{stamp}', '',
                    '## 1. Pattern 意图与代码', '', intent, '', '```python', f'# 文件：{source}', code, '```', '',
                    '## 2. GPU / NPU 行为对照', '', f'GPU：{gpu_behavior}', '', f'NPU：{npu_behavior}', '',
                    'NPU 后端为 `triton_experimental`。原生入口先记录跳过/不生成测试类，再做设备 harness 适配；没有放宽社区数值断言或覆盖冻结源码。', '',
                    f'原例步骤、最小适配和失败记录见 [适配报告](../../../issues/{CASES[name]}/适配报告.md)。', '',
                    '## 3. 性能测例来源和测法', '',
                    '冻结社区中未找到精确切换本目标的原生 benchmark。本次复用社区功能图和原 shape/dtype 派生子图性能测例；不是社区原生性能测试，也不是完整模型端到端。', '',
                    '每臂新进程，OFF1/ON1/ON2/OFF2/OFF3/ON3，预热 10 次、每臂 100 次。每臂求 p50/p99，再取三臂中位数；编译首调用单列，不纳入稳态。记录同步 host 与 NPU Event 两套时钟、峰值 allocated/reserved。', '',
                    '训练单元计时包含前向+反向，不含 SGD；其他单元计时涵盖社区子图的一次调用。性能前已核验原合同、精确目标 FX、全图编译和设备 placement。cat/matmul_backward 的图内 NPU extern 与 CPU/图外 fallback 分开计数。', '',
                    '## 4. 本次结果', '', f"结论：`{indexed[au]['verdict']}`；保留现有默认配置，不凭一次局部图收益全局开启。", '',
                    '| 时钟 | p50 改善 | p99 改善 |', '| --- | ---: | ---: |']
                for clock, values in indexed[au]['improvement_percent'].items():
                    report.append(f"| {clock} | {values['p50']:.2f}% | {values['p99']:.2f}% |")
                report += ['', '负数表示回退。若臂间 p50 波动比超过 1.5，按高方差 PERF_MIXED，不声称稳定收益。', '',
                    f"原始样本、显存、编译时间及逐臂 FX/IR/output_code 索引见 [性能汇总](performance_summary.json)；[功能与门禁原件](functional/{name}.json)。",
                    '', '## 5. 修复/适配边界', '',
                    '这五个单元使用现有 NPU 产品实现即可运行；本轮没有把“测试 harness 适配”称为“原产品不支持、修复后支持”。',
                    '若某个 GPU/NPU kernel 不同，按后端 lowering 差异明确保留，不仅凭数值通过宣称全栈相同。', '']
                (directory/f'{name}_讲解.md').write_text('\n'.join(report))
        manifest['generated_at'] = now.isoformat()
        manifest['status'] = 'completed' if closed==len(manifest['acceptance_units']) else 'npu-partially-completed'
        manifest['reference_contract']['suite_status'] = 'valid-reference-suite'
        manifest['counting_policy'].update(current_frozen_denominator_units=len(manifest['acceptance_units']), current_formally_closed_units=closed)
        plan['generated_at'] = now.isoformat()
        plan['implementation']['status'] = 'implemented-runtime-validated'
        for unit in plan['acceptance_units']:
            if unit['acceptance_unit_id'] in indexed:
                item = indexed[unit['acceptance_unit_id']]
                unit.update(performance_status=item['performance_status'], verdict=item['verdict'])
        plan['status'] = 'performance-disposition-complete' if len(indexed)==len(plan['acceptance_units']) else 'partially-measured'
        summary['generated_at'] = now.isoformat()
        summary['status'] = plan['status']
        dump(manifest_path, manifest)
        dump(plan_path, plan)
        dump(perf_path, summary)
        dump(directory/'npu_functional_summary.json', {'schema_version':'1.0', 'task_id':task,
            'generated_at':now.isoformat(), 'backend':'triton_experimental',
            'status':'functional-passed-performance-gates-signed', 'units':completed_units,
            'scope':'仅units列出的已通过单元；不包含安装态失败的设备解析候选'})
        print(f'finalized={task} functional_closed={closed}/{len(manifest["acceptance_units"])}')


if __name__ == '__main__':
    main()
