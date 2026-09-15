"""边界观察器不执行延迟代码，不干扰 codegen 生命周期。"""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'slice_boundary', Path(__file__).resolve().parents[1] / 'runners/attention_select_slice_boundary.py')
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


class BoundaryObserverTests(unittest.TestCase):
    def test_deferred_objects_are_not_evaluated(self):
        class Deferred:
            def __str__(self):
                raise AssertionError('观察器不能触发延迟展开')
        lines = ['_es_full0 = load()', Deferred(), 'x = _es_full0.shape[0]']
        self.assertEqual(boundary.emitted_source(lines),
                         '_es_full0 = load()\nx = _es_full0.shape[0]')
        self.assertEqual(len(lines), 3)

    def test_case_matrix_is_preserved(self):
        cases = list(boundary.scenarios())
        self.assertEqual(len(cases), 22)
        self.assertEqual(sum(2 if c['dynamic'] else 1 for c in cases), 23)


if __name__ == '__main__':
    unittest.main()
