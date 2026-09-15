"""目标补测仅允许已声明的AST变化，拒绝任意测试/guard修改。"""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


supplement = load('supplement', 'runners/attention_target_supplement.py')
reference = load('supplement_reference', 'runners/reference_runner.py')
FIXTURE = '''
class TestSDPAPatternRewriterTemplate:
    def _test_sdpa_rewriter_16(self):
        def dot_prod_attention(query, key, value, training):
            return torch.nn.functional.dropout(query @ key, p=0.4, training=training) @ value
    def _test_sdpa_rewriter_29(self):
        def dot_prod_attention(query, key, value):
            attn_mask = torch.zeros(1, 1, 8, 8, device=query.device, dtype=query.dtype)
            return torch.ops.aten._safe_softmax(query @ key + attn_mask, -1) @ value
'''


class TargetSupplementTests(unittest.TestCase):
    def test_alias_cannot_launch_independent_performance(self):
        import sys
        sys.path.insert(0, str(ROOT/'scripts'))
        try:
            launcher = load('alias_perf_launcher', 'scripts/run_t102_t107_attention_performance.py')
        finally:
            sys.path.pop(0)
        self.assertEqual(launcher.runnable_patterns('T-105'), [16, 18, 19, 20])
        with self.assertRaisesRegex(ValueError, '去重别名'):
            launcher.runnable_patterns('T-105', [17])
        self.assertEqual(launcher.runnable_patterns('T-104', [15]), [15])

    def test_review_transport_retains_supplement_proof(self):
        exporter = load('supplement_exporter', 'scripts/export_reference_text.py')
        base = 'cases/REF-supplement/debug/target_supplement/'
        names = ('supplement_result.json', 'original_function.py',
                 'adapted_function.py', 'runner_source.py')
        records = {base + name: {} for name in names}
        selected = exporter.review_text_paths(
            {'case_audit': [{'case_id': 'REF-supplement'}]}, records)
        self.assertTrue(set(records).issubset(selected))

    def test_alias_proof_detects_changed_replacement(self):
        reviewer = load('alias_reviewer', 'scripts/review_attention_17_alias.py')
        sources = {name: (reviewer.DEST/name).read_text() for name in reviewer.SOURCES}
        self.assertEqual(reviewer.proof(sources)['earlier_canonical'], 15)
        sources['serialized17.py'] = sources['serialized17.py'].replace(
            'aten.', 'changed_aten.', 1)
        with self.assertRaises(AssertionError):
            reviewer.proof(sources)

    def test_only_probability_is_changed_for_16(self):
        before, after = supplement.extracted_function(FIXTURE, 16)
        self.assertEqual(ast.dump(ast.parse(before.replace('p=0.4', 'p=1e-12'))),
                         ast.dump(ast.parse(after)))
        self.assertIn('training=training', after)

    def test_only_mask_assignment_is_changed_for_29(self):
        before, after = supplement.extracted_function(FIXTURE, 29)
        original = ast.parse(before)
        original.body[0].body[0].value = ast.Name(id='supplement_mask', ctx=ast.Load())
        self.assertEqual(ast.dump(original), ast.dump(ast.parse(after)))

    def test_changed_upstream_anchor_rejected(self):
        with self.assertRaises(ValueError):
            supplement.extracted_function(FIXTURE.replace('p=0.4', 'p=0.5'), 16)
        with self.assertRaises(ValueError):
            supplement.extracted_function(FIXTURE.replace('torch.zeros', 'torch.ones'), 29)
        with self.assertRaises(ValueError):
            supplement.extracted_function(FIXTURE, 17)

    def test_plans_keep_original_first_and_supplements_noncounting(self):
        import json
        for task in (105, 107):
            plan = json.loads((ROOT/f'upstream/t{task}_reference_plan.yaml').read_text())
            self.assertEqual(plan['cases'][0]['tracking_mode'], 'direct')
            case = plan['cases'][-1]
            self.assertEqual(case['variant_ids'], [])
            reference.validate_target_derivation(plan['task_id'], case)

    def test_target_contract_rejects_drift(self):
        import json
        plan = json.loads((ROOT/'upstream/t107_reference_plan.yaml').read_text())
        original = plan['cases'][-1]
        for key, value in [('case_id', 'REF-arbitrary'), ('entrypoint_args', ['--pattern', '17']),
                           ('source_test', 'wrong'), ('entrypoint', 'arbitrary.py'),
                           ('acceptance_unit_id', 'wrong')]:
            case = copy.deepcopy(original)
            case[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reference.validate_target_derivation('T-107', case)


if __name__ == '__main__':
    unittest.main()
