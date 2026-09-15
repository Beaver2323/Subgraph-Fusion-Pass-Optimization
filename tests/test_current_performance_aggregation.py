from pathlib import Path
import copy
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_t091_t100_completion import aggregate, ORDER
import review_t091_t100_completion as review


class AggregationTests(unittest.TestCase):
    def records(self,on=1.0):
        return {arm:{'timing':{clock:{'p50':1.0 if arm.startswith('off') else on,
                                     'p99':1.0 if arm.startswith('off') else on}
                              for clock in ('host_ms','event_ms')}} for arm in ORDER}

    def test_opposite_clocks_never_improved(self):
        records=self.records(0.9)
        for arm in ORDER:
            if arm.startswith('on'):records[arm]['timing']['event_ms']['p99']=1.2
        self.assertEqual(aggregate(records)[3],'PERF_MIXED')

    def test_large_round_spread_is_inconclusive(self):
        records=self.records(0.8)
        records['off1']['timing']['host_ms']['p50']=1.5
        self.assertEqual(aggregate(records)[3],'PERF_MIXED')

    def test_neutral_and_stable_improvement(self):
        self.assertEqual(aggregate(self.records())[3],'PERF_NEUTRAL')
        self.assertEqual(aggregate(self.records(0.9))[3],'PERF_IMPROVED')

    def test_new_launcher_uses_cooperative_lock(self):
        source=(Path(__file__).resolve().parents[1]/'scripts/run_t091_t100_performance.py').read_text()
        self.assertIn('pass-tracker-npu-performance.lock',source)
        self.assertIn('fcntl.LOCK_EX | fcntl.LOCK_NB',source)

    def test_current_archives_recompute_without_original_tmp(self):
        is_file = Path.is_file
        def archive_only(path):
            if str(path).startswith('/home/z50063656/tmp/'):
                raise AssertionError('离线复核不应读取原机器tmp')
            return is_file(path)
        with patch.object(Path,'is_file',archive_only):
            for task in review.TASKS:
                review.check_current(task)

    def test_false_improved_summary_rejected(self):
        original=review.load
        def changed(path):
            result=original(path)
            if path.name=='performance_summary.json':
                result=copy.deepcopy(result)
                result['acceptance_units'][0]['verdict']='PERF_IMPROVED'
            return result
        with patch.object(review,'load',side_effect=changed), self.assertRaisesRegex(ValueError,'汇总'):
            review.check_current('T-091')

    def test_missing_event_samples_rejected(self):
        original=review.load
        def changed(path):
            result=original(path)
            if path.name=='result.json' and result.get('phase')=='benchmark':
                result=copy.deepcopy(result)
                result['samples'].pop('event_ms')
            return result
        with patch.object(review,'load',side_effect=changed), self.assertRaisesRegex(ValueError,'计时时钟'):
            review.check_current('T-100')
