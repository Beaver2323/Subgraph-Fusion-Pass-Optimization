#!/usr/bin/env python3
"""21号经用户确认的归因受限结项；只读复核原件，不签计时门禁、不执行设备。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AU = 'AU-fuse-attention-sfdp-pattern-21'
STATUS = 'accepted-not-independently-attributable'
VERDICT = 'PERF_NOT_INDEPENDENTLY_ATTRIBUTABLE'


def read(path):
    return json.loads(path.read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ref(root, path):
    return dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def bound(root, entry):
    path = (root/entry['path']).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), '处置证据缺失或越界')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], '处置证据哈希错误')
    return path


def verify(root, record):
    require(record.get('acceptance_unit_id') == AU and record.get('task_id') == 'T-106'
            and record.get('backend') == 'triton_experimental', '处置合同或后端不符')
    require(record.get('performance_status') == STATUS and record.get('performance_verdict') == VERDICT,
            '归因受限不能改为已测、收益或免测')
    for key in ('performance_measured', 'benefit_counted', 'default_disabled_exempt', 'performance_gate_issued'):
        require(record.get(key) is False, '归因处置不得冒充计时门禁或收益/免测')
    require(not any(k in record for k in ('timing', 'samples', 'improvement_percent', 'gate')), '归因处置不能包含伪造计时')
    require(record.get('correctness') == 'passed' and record.get('numerical_execution') is True
            and record.get('target_rewrite') == 'confirmed', '缺少功能结论')
    approval = bound(root, record['approval'])
    require('接受该处置并明确统计' in approval.read_text(), '缺用户明确处置确认')
    for entry in record['inventories']:
        inventory = read(bound(root, entry))
        for artifact in inventory.get('files', inventory.get('artifacts', [])):
            bound(root, artifact)
    original_path = bound(root, record['community_evidence'])
    original = read(original_path)
    require(original['acceptance_unit_id'] == AU and original['backend'] == 'triton_experimental'
            and original['status'] == 'community-contract-passed' and original['tests_ran'] == 1
            and original['tests_skipped'] == 0 and original['native_assertions_passed'] is True
            and original['numerical_assertions_modified'] is False
            and original['isolated_codegen_candidate'] is False
            and original['isolated_registration_candidate'] is False
            and original['product_gate_bypassed'] is False
            and original['exact_target_observations'] == 2
            and len(original['tensor_assertions']) == 2
            and all(t['passed'] is True for t in original['tensor_assertions']), '原例功能未通过')
    gpu = read(bound(root, record['gpu_reference']))
    case = [c for c in gpu['cases'] if c['acceptance_unit_id'] == AU]
    require(original['pytorch_commit'] == gpu['expected_pytorch_commit']
            == '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
            and len(case) == 1 and case[0]['tests_ran'] == 1 and case[0]['tests_skipped'] == 0
            and case[0]['target_review']['expected_target'] == '_sfdp_pattern_21'
            and case[0]['target_review']['exact_target_observations'] > 0, 'GPU精确目标证据不符')
    execution_path = bound(root, record['off_execution'])
    execution = read(execution_path)
    require(execution['return_code'] == 1, '必须保留归因失败原退出码')
    off = execution_path.parent
    require('RuntimeError: OFF仍命中fuse_attention' in (off/'stderr.log').read_text()
            and not (off/'result.json').exists(), '归因阻断缺少原失败证据')
    observations = [read(p) for p in (off/'target-observations').glob('*/contract_observation.json')]
    require(any(o['target'] == '_sfdp_pattern_22_half_inference' and o['graph_changed'] is True
                for o in observations), '缺少22接替的具名改图证据')
    from review_attention_functional_gate import inspect
    for folder in (off, original_path.parent):
        code = inspect(folder)
        require(code['output_code_count'] > 0 and not code['suspicious_cpu_lines'], '缺实际NPU代码或含CPU线索')
    return True


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    require(Path.cwd().resolve() == Path('/home/z50063656/tmp'), '请从测试临时目录执行')
    current = ROOT/'results/current/T-106'
    path = current/'functional/pattern-21.json'
    if not args.write:
        verify(ROOT, read(path))
        print('attribution_disposition=OK task=T-106 pattern=21 measured=false benefit=false disabled_exempt=false device_execution=false')
        return
    require(not path.exists(), '拒绝覆盖已确认处置')
    issue = ROOT/'issues/REF-sfdp-pattern-21-native'
    original = issue/'evidence/t106-native-6t1pel2i/adapter'
    off = issue/'evidence/functional-20260915T112553+0800'
    now = datetime.now().astimezone().isoformat()
    record = dict(schema_version='1.0', generated_at=now, task_id='T-106', acceptance_unit_id=AU,
        backend='triton_experimental', correctness='passed', numerical_execution=True, target_rewrite='confirmed',
        graph_breaks=0, fallbacks=0, comparison_verdict='BEHAVIOR_UNCHANGED', repair_status='not-needed',
        performance_status=STATUS, performance_verdict=VERDICT, performance_measured=False,
        benefit_counted=False, default_disabled_exempt=False, performance_gate_issued=False,
        approval=ref(ROOT, issue/'最终处置确认.md'), community_evidence=ref(ROOT, original/'result.json'),
        gpu_reference=ref(ROOT, current/'gpu_reference_review.json'), off_execution=ref(ROOT, off/'off/execution.json'),
        inventories=[ref(ROOT, original/'evidence_inventory.json'), ref(ROOT, off/'inventory.json')],
        community_alignment=dict(status='PARTIAL_ALIGNED', aligned_scope=['完整原社区数值与本编号改写通过'],
            divergent_scope=['NPU数学展开与CUDA融合kernel不同；已选输入OFF被22接替，不能独立归因'],
            open_scope=['未覆盖dtype/形状/梯度不外推；不证明所有输入均无法独立归因'],
            disposition='用户接受归因限制结项，不计收益、不计默认关闭免测、不更改产品配置'))
    verify(ROOT, record)
    functional = read(current/'npu_functional_summary.json')
    performance = read(current/'performance_summary.json')
    require(not any(r['acceptance_unit_id'] == AU for r in functional['units']+performance['acceptance_units']), '拒绝重复结项')
    functional.update(generated_at=now, status='functional-passed-performance-disposition-complete')
    functional['units'].append(dict(unit='pattern-21', acceptance_unit_id=AU,
        status='functional-passed-attribution-disposition-accepted', functional_evidence=str(path.relative_to(ROOT))))
    performance['generated_at'] = now
    performance['acceptance_units'].append(dict(unit='pattern-21', acceptance_unit_id=AU,
        performance_status=STATUS, verdict=VERDICT, performance_measured=False, benefit_counted=False,
        default_disabled_exempt=False, disposition_evidence=str(path.relative_to(ROOT)),
        product_action='不改默认开关；不制造ON/OFF；不计收益或默认关闭免测'))
    performance['disposition_counts'] = dict(completed=4, measured=3, attribution_limited=1,
        improved=1, regressed=2, default_disabled_exempt=0)
    from review_attention_completion import completed_manifest
    manifest_path = ROOT/'upstream/t106_manifest.yaml'
    manifest = completed_manifest(read(manifest_path), AU, now)
    plan_path = ROOT/'upstream/t106_performance_plan.yaml'
    plan = read(plan_path)
    plan['generated_at'] = now
    for unit in plan['acceptance_units']:
        if unit['acceptance_unit_id'] == AU:
            unit.update(performance_status=STATUS, verdict=VERDICT, disposition_evidence=str(path.relative_to(ROOT)))
    for destination, payload in ((path,record),(current/'npu_functional_summary.json',functional),
            (current/'performance_summary.json',performance),(manifest_path,manifest),(plan_path,plan)):
        write_json(destination, payload)
    print('attribution_disposition=written; T-106=4/4; measured=3; attribution_limited=1; disabled_exempt=0')


if __name__ == '__main__':
    main()
