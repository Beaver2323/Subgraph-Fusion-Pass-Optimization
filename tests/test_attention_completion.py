"""六臂归档验算的样本与去重检查，纯标准库。"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('attention_completion',ROOT/'scripts/review_attention_completion.py')
review=importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class AttentionCompletionTests(unittest.TestCase):
    def fixture(self):
        values=[0.5+i/1000 for i in range(100)]
        return {'samples':{clock:values for clock in ('host_ms','event_ms')},
                'timing':{clock:{key:review.worker.percentile(values,q) for key,q in (('p50',.5),('p99',.99))}
                          for clock in ('host_ms','event_ms')}}

    def test_recompute_samples(self):
        review.validate_samples(self.fixture())

    def test_reject_missing_samples_clock_and_fake_percentile(self):
        for kind in ('count','clock','percentile','nan'):
            record=deepcopy(self.fixture())
            if kind=='count': record['samples']['host_ms'].pop()
            if kind=='clock': del record['samples']['event_ms']
            if kind=='percentile': record['timing']['host_ms']['p99']+=1
            if kind=='nan': record['samples']['host_ms'][0]=float('nan')
            with self.assertRaises(ValueError): review.validate_samples(record)

    def test_summary_preserves_existing_unit_and_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'summary.json'
            row={'acceptance_unit_id':'AU-one'}
            payload=review.merge_summary(path,'T-104','units',row,'fixture-only')
            path.write_text(review.json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'覆盖'):
                review.merge_summary(path,'T-104','units',row,'fixture-only')
            updated=review.merge_summary(path,'T-104','units',{'acceptance_unit_id':'AU-two'},'fixture-only')
            self.assertEqual(updated['units'],[row,{'acceptance_unit_id':'AU-two'}])

    def test_freeze_only_reviewed_unit(self):
        unit=lambda name: dict(acceptance_unit_id=name,denominator_eligible='yes-provisional',
                               review_status='prepared',coverage_phase='awaiting',variants=[{}])
        original=dict(acceptance_units=[unit('one'),unit('two')],counting_policy={},reference_contract={})
        payload=review.completed_manifest(deepcopy(original),'one','fixture-time')
        self.assertEqual(payload['counting_policy']['current_formally_closed_units'],1)
        self.assertEqual(payload['status'],'partially-completed')
        self.assertEqual(payload['acceptance_units'][1],original['acceptance_units'][1])
