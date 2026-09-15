"""归因受限结项与计时、收益、产品免测严格分离；仅校验仓库文本。"""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import review_t106_attribution_disposition as disposition


class AttributionDispositionTests(unittest.TestCase):
    def setUp(self):
        self.record = disposition.read(ROOT/'results/current/T-106/functional/pattern-21.json')

    def test_accepted_evidence_revalidates_without_device(self):
        self.assertTrue(disposition.verify(ROOT, self.record))

    def test_cannot_count_as_measurement_benefit_or_disable(self):
        for key in ('performance_measured','benefit_counted','default_disabled_exempt','performance_gate_issued'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                disposition.verify(ROOT, dict(self.record, **{key: True}))

    def test_cannot_fabricate_timing(self):
        with self.assertRaises(ValueError):
            disposition.verify(ROOT, dict(self.record, timing={}))

    def test_requires_bound_user_confirmation(self):
        record = copy.deepcopy(self.record)
        record['approval']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, '哈希错误'):
            disposition.verify(ROOT, record)

    def test_cannot_extend_exception_to_another_pattern(self):
        with self.assertRaisesRegex(ValueError, '合同或后端'):
            disposition.verify(ROOT, dict(self.record, acceptance_unit_id='AU-fuse-attention-sfdp-pattern-22'))


if __name__ == '__main__':
    unittest.main()
