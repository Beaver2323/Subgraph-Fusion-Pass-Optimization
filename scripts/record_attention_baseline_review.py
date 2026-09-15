#!/usr/bin/env python3
"""按明确的安装态 run_result 归档 attention 阶段事实；不自动确诊或签性能门禁。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from inspect_npu_codegen import inspect
from npu_stage_review import verify

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/home/z50063656/tmp')
APPROVED = set(range(2,16)) | set(range(18,25)) | {28,30}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def preserve_previous_stage(root, task, unit, previous, current, supersede):
    if previous is None or previous == current:
        return
    require(supersede, '已有不同阶段评审；先明确 --supersede 并保留历史')
    require(previous.get('candidate_verified') is False
            and previous.get('performance_gate_issued') is False,
            '不能自动替换候选验证或已签性能状态')
    encoded = (json.dumps(previous, ensure_ascii=False, indent=2)+'\n').encode()
    digest = hashlib.sha256(encoded).hexdigest()
    destination = root/'results/history'/task/f'{unit}-stage-{digest[:16]}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(destination.read_bytes() == encoded, '历史快照冲突')
    else:
        with destination.open('xb') as stream:
            stream.write(encoded)


def collect(run_path, root=ROOT):
    run_path = run_path.resolve(strict=True)
    require(run_path.is_relative_to(root/'issues'), '只接收本仓明确的运行记录')
    parent = read(run_path)
    match = re.fullmatch(r'REF-sfdp-pattern-(\d+)-native', parent['case_id'])
    require(match is not None and int(match[1]) in APPROVED, '编号未在本轮基线范围')
    number = int(match[1])
    require(parent['mode'] == 'adapter' and parent['attention_registration_candidate'] is False
            and parent.get('attention_select_slice_candidate', False) is False
            and parent['return_code'] in (0,1), '不得混入候选或中断运行')
    require(parent['installed_product_before'] == parent['installed_product_after'], '测试期间产品文件变化')
    raw = Path(parent['raw_artifact_dir']).resolve()/'adapter'
    require(raw.is_relative_to(WORK.resolve()), '原件不在工作临时目录')
    record = read(raw/'result.json')
    require(record['case_id'] == parent['case_id'] and record['task_id'] == parent['task_id']
            and record['isolated_registration_candidate'] is False
            and record.get('isolated_codegen_candidate', False) is False
            and record['backend'] == 'triton_experimental'
            and record['pytorch_commit'] == '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
            and record['product_gate_bypassed'] is False
            and record['numerical_assertions_modified'] is False, '原件后端/身份/断言范围不符')
    source_name = record['source_test'].split('::',1)[0]
    frozen_source = subprocess.check_output(['git','-C','/home/z50063656/Pass/src/pytorch',
                                            'show',record['pytorch_commit']+':'+source_name])
    require(hashlib.sha256(frozen_source).hexdigest() == record['source_sha256'] == parent['source_sha256'],
            '运行方法源码不等于冻结community文件')
    require(sha(raw/'harness_source.py') == record['harness_snapshot_sha256'], '执行器原始快照不符')
    require(all(t.get('passed') is True and t.get('actual_device','').startswith('npu:')
                and t.get('expected_device','').startswith('npu:') for t in record['tensor_assertions']),
            '数值比较记录不是NPU对NPU已通过比较')
    gpu_path = root/'results/current'/parent['task_id']/'gpu_reference_review.json'
    gpu = read(gpu_path)
    cases = [c for c in gpu['cases'] if c['case_id'] == parent['case_id']]
    require(len(cases)==1 and cases[0]['target_review']['status']=='gpu-contract-reviewed-awaiting-npu', 'GPU本编号未确认')
    code = inspect(raw)
    require(code['output_code_count']>0, '无生成代码，需人工处理前置编译失败')
    passed = (record['status']=='community-contract-passed' and record['tests_ran']==1
              and record['tests_skipped']==0 and record['native_assertions_passed'] is True)
    return parent, raw, record, gpu_path, code, passed, number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-result', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--supersede', action='store_true',
                        help='同一合同新复验替换阶段索引前，按内容哈希保留旧索引；原运行证据不改')
    args = parser.parse_args()
    require(Path.cwd().resolve()==WORK, '必须从 /home/z50063656/tmp 运行')
    parent, raw, record, gpu_path, code, passed, number = collect(args.run_result)
    case, task, unit = record['case_id'],record['task_id'],record['acceptance_unit_id']
    destination = ROOT/'issues'/case/'evidence'/raw.parent.name/'adapter'
    tensors = record['tensor_assertions']
    names = sorted({o['target'] for o in record['observations'] if o.get('graph_changed')})
    failures = record['test_failures']+record['test_errors']
    scope = (f"原方法 tests={record['tests_ran']}/skip={record['tests_skipped']}；"
             f"已通过Tensor比较={len(tensors)}；本编号精确改写={record['exact_target_observations']}；"
             + ('原社区合同通过，仍待逐项图/性能审核' if passed else '原合同未通过；后续分支不计覆盖'))
    if not tensors:
        scope += '；没有数值比较证据，不计精度PASS'
    stage = dict(generated_at=record['generated_at'], task_id=task, acceptance_unit_id=unit,
                 case_id=case, backend='triton_experimental',
                 baseline={'path':str((destination/'result.json').relative_to(ROOT)), 'sha256':sha(raw/'result.json')},
                 candidate_verified=False, correctness_scope=scope,
                 repair_status='awaiting-code-and-performance-review' if passed else 'baseline-failed-diagnosis-pending',
                 community_alignment_status='PARTIAL_ALIGNED' if passed else 'NOT_ALIGNED_REPAIR_REQUIRED',
                 reason='数值、精确目标及训练/推理范围以本例原断言和实际观测为准；不由总counter推导完整对齐。',
                 next_action='人工检查各分支FX/IR/codegen及原断言范围；失败先定位首处分歧，不自动套用pattern 1候选；通过后另建合法性能门禁。',
                 performance_gate_issued=False, report=f'issues/{case}/安装态基线复核.md',
                 gpu_review={'path':str(gpu_path.relative_to(ROOT)), 'sha256':sha(gpu_path)},
                 parent_run={'path':str(args.run_result.resolve().relative_to(ROOT)), 'sha256':sha(args.run_result)})
    if not passed and record['exact_target_observations'] > 0 and any(
        'AssertionError: Tensor-likes are not close!' in failure for failure in failures
    ):
        stage.update(failure_layer='numerical-contract',
                     repair_status='precision-failure-diagnosis-pending',
                     community_alignment_status='NOT_ALIGNED_REPAIR_REQUIRED',
                     reason='目标已改图，但原社区数值断言失败；保留实际超差与首个失败分支，未执行的后续分支不计通过。')
    if not passed and record['exact_target_observations'] > 0 and any(
        'AssertionError: set() is not true : CUDA SDPA符号须对应实际NPU fusion-attention调用' in failure
        for failure in failures
    ):
        stage.update(failure_layer='test-adapter',
                     repair_status='codegen-assertion-adaptation-review-pending',
                     community_alignment_status='CODEGEN_DIFFERENCE_NUMERICS_NOT_YET_REACHED',
                     reason='目标已改图；失败在适配器对融合kernel符号的过窄要求。数值断言尚未执行，不能据此判产品精度缺陷。')
    if args.write:
        subprocess.run([sys.executable,str(ROOT/'scripts/archive_prepared_npu_case.py'),
                        '--run-dir',str(raw),'--evidence-only','--keep-reports'],check=True,cwd=WORK)
        verify(ROOT,task,unit,stage)
        stage_path = ROOT/'results/current'/task/'npu_stage_reviews.json'
        collection = read(stage_path) if stage_path.exists() else {'task_id':task,'units':{}}
        existing = collection['units'].get(unit)
        preserve_previous_stage(ROOT,task,unit,existing,stage,args.supersede)
        collection['units'][unit]=stage
        stage_path.write_text(json.dumps(collection,ensure_ascii=False,indent=2)+'\n')
        report = ROOT/'issues'/case/'安装态基线复核.md'
        lines = [f'# {task} pattern {number} 安装态基线复核','',
                 f"> 证据时间：{record['generated_at']}；复核时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。",'',
                 '## 1. 本轮结论与适配范围','',scope+'。','',
                 'GPU已确认本编号；NPU固定triton_experimental。原生类定义阻断已有原件，随后执行本case适配。',
                 '保留原方法、dtype/shape/训练推理参数和数值容差；CUDA代码符号断言显式适配为实际NPU调用。',
                 '未改安装态产品、未加载注册候选、未签性能门禁。详见[适配报告](适配报告.md)。','',
                 '## 2. 原代码合同与必要调用链','',
                 '```text',record['source_test'],
                 ' -> _check_common -> torch.compile -> AOT joint_graph -> fuse_attention pattern',
                 ' -> backend lowering/codegen -> NPU -> 原counter/codegen/数值断言',
                 '```','',
                 '```python','# 冻结 test/inductor/test_fused_attention.py:121、137、140（节选）',
                 'self.assertGreaterEqual(counters["inductor"]["fuse_attention"], 1)',
                 'if not has_dropout or override_check_equal:',
                 '    self.assertEqual(result1, result2, atol=atol, rtol=rtol)',
                 '# training分支执行backward；相同条件成立时才比较浮点输入梯度。','```','',
                 '数值比较为零时不能记精度通过；未执行的后续参数分支不借GPU结果补成NPU通过。','',
                 '## 3. 实际目标与生成代码线索','',
                 '实际发生图变化的注册：'+(', '.join(f'`{n}`' for n in names) or '无')+'。','',
                 f"实际debug生成图数量：{code['output_code_count']}。下表是AST读取的调用，不是性能测量或完整fallback判决。",'',
                 '| 生成调用 | 图内静态出现次数 |','|---|---:|']
        lines += [f'| `{name}` | {count} |' for name,count in code['calls'].items()]
        lines += ['',f"可疑CPU行数量：{len(code['suspicious_cpu_lines'])}；无可疑字符串不自动等于无fallback。",'',
                  '## 4. 原失败栈与后续工作','']
        if failures:
            lines += ['```text','\n'.join(failures),'```','',
                      '以上为真实失败栈。根因未在本生成报告中自动下结论；不能仅因相同counter症状就套用其他编号修复。']
        else:
            lines += ['原方法无失败/异常；仍需按原断言范围审查，不把结构通过外推成全精度/性能通过。']
        lines += ['',f"[本轮原件](evidence/{raw.parent.name}/adapter/)包含 result、执行器快照、目标FX、前后IR和output_code。",'',
                  '[性能测例来源与数值边界](../../report/attention_performance_contract_review_20260914.md)。','']
        require(not report.exists() or existing == stage or (args.supersede and existing is not None),
                '拒绝覆盖未登记报告')
        if report.exists() and existing is not None and existing != stage:
            previous_report = report.with_name('安装态基线复核-'+sha(report)[:16]+'.md')
            if previous_report.exists():
                require(previous_report.read_bytes() == report.read_bytes(), '旧报告快照冲突')
            else:
                with previous_report.open('xb') as stream:
                    stream.write(report.read_bytes())
        report.write_text('\n'.join(lines))
    print(json.dumps({'task':task,'case':case,'baseline_passed':passed,'tensor_comparisons':len(tensors),
                      'exact_target_changes':record['exact_target_observations'],'stage_only':True},ensure_ascii=False))


if __name__=='__main__':
    main()
