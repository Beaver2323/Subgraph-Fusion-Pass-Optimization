"""GPU 观察证据复核：不同编号、缺图、伪造状态和旧包混读均不能误通过。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('gpu_observer_review', ROOT/'scripts/review_uploaded_gpu_tasks.py')
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)


def sample(case='REF-sfdp-pattern-1-native', target='_sfdp_pattern_1_inference', changed=True):
    base = f'cases/{case}/debug/native_contract_observer/1-ReplacementPatternEntry-0001'
    row = dict(capture_scope='pattern-entry-apply-after-extra-check', target=target,
               handler_returned=True, graph_changed=changed, test_body_modified=False,
               device_modified=False, assertions_modified=False, product_gate_bypassed=False,
               numerical_correctness_proven_by_observer=False)
    return {base+'/contract_observation.json':json.dumps(row).encode(),
            base+'/fx_graph_readable.py':b'before', base+'/fx_graph_transformed.py':b'after' if changed else b'before'}


class ObserverReviewTests(unittest.TestCase):
    def test_exact_target_and_both_fx(self):
        result = reviewer.observe_case('REF-sfdp-pattern-1-native', sample())
        self.assertEqual(result['status'], 'gpu-contract-reviewed-awaiting-npu')
        self.assertEqual(result['exact_target_observations'], 1)
        self.assertEqual(len(result['observations'][0]['files']), 3)

    def test_one_does_not_match_ten(self):
        result = reviewer.observe_case('REF-sfdp-pattern-1-native', sample(target='_sfdp_pattern_10_inference'))
        self.assertEqual(result['status'], 'native-passed-different-target-observed')

    def test_29_does_not_inherit_30(self):
        case = 'REF-sfdp-pattern-29-native'
        self.assertEqual(reviewer.observe_case(case, sample(case, '_sfdp_pattern_30_half_bs1_inference'))[
            'exact_target_observations'], 0)

    def test_training_is_not_inference_evidence(self):
        result = reviewer.observe_case('REF-sfdp-pattern-1-native', sample(target='_sfdp_pattern_1_half_training'))
        self.assertEqual(result['observed_targets'], ['_sfdp_pattern_1_half_training'])

    def test_no_observation_stays_pending(self):
        self.assertEqual(reviewer.observe_case('REF-sfdp-pattern-1-native', {})['status'],
                         'native-passed-target-attribution-pending')

    def test_returned_without_graph_change_not_positive(self):
        result = reviewer.observe_case('REF-sfdp-pattern-1-native', sample(changed=False))
        self.assertEqual(result['exact_target_observations'], 0)
        self.assertEqual(result['status'], 'native-passed-target-attribution-pending')

    def test_missing_fx_rejected(self):
        data = sample()
        del data[next(k for k in data if k.endswith('fx_graph_transformed.py'))]
        with self.assertRaisesRegex(ValueError, 'FX'):
            reviewer.observe_case('REF-sfdp-pattern-1-native', data)

    def test_forged_change_rejected(self):
        data = sample()
        data[next(k for k in data if k.endswith('fx_graph_transformed.py'))] = b'before'
        with self.assertRaisesRegex(ValueError, '正文'):
            reviewer.observe_case('REF-sfdp-pattern-1-native', data)

    def test_gate_bypass_rejected(self):
        data = sample()
        key = next(k for k in data if k.endswith('contract_observation.json'))
        row = json.loads(data[key]); row['product_gate_bypassed'] = True
        data[key] = json.dumps(row).encode()
        with self.assertRaisesRegex(ValueError, 'product_gate_bypassed'):
            reviewer.observe_case('REF-sfdp-pattern-1-native', data)

    def test_stack_does_not_claim_axis_rewrite(self):
        case = 'REF-stack-axis-normalization-native'
        result = reviewer.observe_case(case, sample(case, 'normalize_stack_default'))
        self.assertIn('不宣称', result['semantic_boundary'])

    def test_ambiguous_packages_require_explicit_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'manifest.json').touch(); (root/'text-handoff.json').touch()
            with self.assertRaisesRegex(ValueError, '不唯一'):
                reviewer.select_input(root)
            self.assertEqual(reviewer.select_input(root, root/'text-handoff.json'), root/'text-handoff.json')

    def test_input_cannot_escape_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'T-091').mkdir(); (root/'outside.json').touch()
            with self.assertRaisesRegex(ValueError, '本任务'):
                reviewer.select_input(root/'T-091', root/'outside.json')


if __name__ == '__main__':
    unittest.main()
