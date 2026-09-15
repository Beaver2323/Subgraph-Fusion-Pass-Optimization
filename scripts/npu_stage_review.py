"""有原件约束的NPU失败/候选阶段评审；不能升级成正式comparison或安装态修复。"""
import hashlib
import importlib.util
import json
from pathlib import Path


def verify(root, task, unit, review):
    if review.get('task_id') != task or review.get('acceptance_unit_id') != unit:
        raise ValueError('NPU阶段评审归属不符')
    if review.get('backend') != 'triton_experimental' or review.get('performance_gate_issued') is not False:
        raise ValueError('阶段评审不允许混后端或签性能gate')
    records = {}
    for mode in ('baseline','candidate'):
        item = review.get(mode)
        if not item:
            continue
        path = (root/item['path']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('阶段原件越界或缺失')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('阶段原件哈希不符')
        raw = json.loads(path.read_text())
        if (raw.get('backend') != 'triton_experimental' or raw.get('task_id') != task
                or raw.get('acceptance_unit_id') != unit or raw.get('case_id') != review.get('case_id')
                or raw.get('pytorch_commit') != '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
                or raw.get('product_gate_bypassed') is not False):
            raise ValueError('实际运行合同或后端不符')
        registration_candidate = raw.get('isolated_registration_candidate', False)
        codegen_candidate = raw.get('isolated_codegen_candidate', False)
        if mode == 'baseline' and (registration_candidate or codegen_candidate):
            raise ValueError('不能用隔离候选冒充当前安装态')
        if mode == 'candidate':
            kind = review.get('candidate_kind', 'registration')
            if (kind not in ('registration', 'codegen')
                    or registration_candidate is not (kind == 'registration')
                    or codegen_candidate is not (kind == 'codegen')):
                raise ValueError('候选来源未明确隔离或候选类型不符')
        records[mode] = raw
    if 'baseline' not in records:
        raise ValueError('缺少当前安装态基线')
    candidate_passed = ('candidate' in records and records['candidate'].get('status') == 'community-contract-passed'
                        and records['candidate'].get('tests_ran') == 1 and records['candidate'].get('tests_skipped') == 0
                        and records['candidate'].get('native_assertions_passed') is True)
    if review.get('candidate_verified') is not candidate_passed:
        raise ValueError('候选通过状态与原件不一致')
    baseline = records['baseline']
    if review.get('failure_layer') == 'numerical-contract':
        failures = baseline.get('test_failures', []) + baseline.get('test_errors', [])
        if not any('AssertionError: Tensor-likes are not close!' in failure for failure in failures):
            raise ValueError('数值失败分类缺少对应原断言失败栈')
    if review.get('failure_layer') == 'test-adapter':
        failures = baseline.get('test_failures', []) + baseline.get('test_errors', [])
        if not (baseline.get('exact_target_observations', 0) > 0 and any(
            'AssertionError: set() is not true : CUDA SDPA符号须对应实际NPU fusion-attention调用' in failure
            for failure in failures
        )):
            raise ValueError('测试适配阻断分类没有对应原失败栈和目标改写')
    baseline_passed = (baseline.get('status') == 'community-contract-passed'
                       and baseline.get('tests_ran') == 1 and baseline.get('tests_skipped') == 0
                       and baseline.get('native_assertions_passed') is True)
    state = dict(baseline_passed=baseline_passed,candidate_passed=candidate_passed)
    if review.get('deployment'):
        if task == 'T-102' and unit in {f'AU-fuse-attention-sfdp-pattern-{n}' for n in range(1,6)}:
            spec = importlib.util.spec_from_file_location(
                'training_deployment_review', Path(__file__).with_name('validate_t102_training_deployment.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            state.update(module.verify(root, review['deployment']))
            return state
        if task != 'T-106' or unit != 'AU-fuse-attention-sfdp-pattern-22':
            raise ValueError('部署评审器尚未支持此单元')
        spec = importlib.util.spec_from_file_location(
            'attention_deployment_review', Path(__file__).with_name('validate_attention_slice_deployment.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state.update(module.verify(root, review['deployment']))
    return state
