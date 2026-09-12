"""隔离候选不得冒充安装态闭环，原件/产物及telemetry名称必须可审计。"""
import ast
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('candidate_review', ROOT/'scripts/review_pending_npu_candidates.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class PendingCandidatesTests(unittest.TestCase):
    def test_archived_baselines_and_candidates_are_bound(self):
        review.validate_archived()

    def test_candidate_cannot_claim_formal_closure(self):
        original = review.load
        def altered(path):
            row = original(path)
            if path.name == 'npu_blocker_review.json' and row.get('task_id') == 'T-087':
                row['candidate']['functional_closure'] = True
            return row
        with patch.object(review, 'load', side_effect=altered):
            with self.assertRaisesRegex(ValueError, '不可冒充'):
                review.validate_archived()

    def test_e8m0_registration_has_precise_telemetry(self):
        tree = ast.parse((ROOT/'issues/REF-e8m0-log2-pattern-native/e8m0_npu_candidate.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == 'register_replacement']
        self.assertEqual(len(calls), 1)
        kwargs = {kw.arg:kw.value for kw in calls[0].keywords}
        self.assertEqual(ast.literal_eval(kwargs['pattern_name']), 'e8m0_rceil_log2_pattern')


if __name__ == '__main__':
    unittest.main()
