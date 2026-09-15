"""静态manifest没有扩展缺口不能计入已完成设备覆盖。"""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'scripts/validate_comparison_data.py'
spec = importlib.util.spec_from_file_location('progress_counts_validator', path)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ComparisonProgressCountsTests(unittest.TestCase):
    def test_unexecuted_manifest_is_not_fully_covered(self):
        units = {'closed': {}, 'not-run': {}, 'failure': {}}
        counts = validator.coverage_counts(units, {'closed','failure'}, {'closed'})
        self.assertEqual(counts, {
            'fully_covered_units': 1,
            'units_without_declared_extension_gap': 3,
            'tracked_ids_without_current_comparison': 1,
            'recorded_open_or_inconclusive_units': 1,
        })

    def test_closed_main_with_pending_extension_is_not_fully_covered(self):
        units = {'old-main': {'pending_variants': ['new-dtype']}}
        counts = validator.coverage_counts(units, {'old-main'}, {'old-main'})
        self.assertEqual(counts['fully_covered_units'], 0)
        self.assertEqual(counts['recorded_open_or_inconclusive_units'], 0)
