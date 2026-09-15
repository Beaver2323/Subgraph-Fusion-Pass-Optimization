"""目标OFF只删除精确注册项，不能关闭整轮或误删相邻编号。"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'runners'))
from target_entry_control import disable_entries


class TargetEntryTests(unittest.TestCase):
    def test_pattern_boundary(self):
        one = SimpleNamespace(pattern_name='_sfdp_pattern_1_inference')
        ten = SimpleNamespace(pattern_name='_sfdp_pattern_10_inference')
        other = SimpleNamespace(pattern_name='other')
        registry = SimpleNamespace(patterns={'a':[one,ten], 'b':[other]})
        result = disable_entries(registry, pattern_prefix='_sfdp_pattern_1')
        self.assertEqual(registry.patterns, {'a':[ten],'b':[other]})
        self.assertEqual(result['removed'], ['_sfdp_pattern_1_inference'])
        self.assertFalse(result['whole_pass_disabled'])

    def test_only_stack_handler(self):
        def normalize_stack_default():
            pass
        def normalize_cat_default():
            pass
        stack, cat = (SimpleNamespace(handler=f) for f in (normalize_stack_default, normalize_cat_default))
        registry = SimpleNamespace(patterns={'a':[stack,cat]})
        disable_entries(registry, handler_names=('normalize_stack_default',))
        self.assertEqual(registry.patterns['a'], [cat])

    def test_missing_target_rejected(self):
        with self.assertRaises(ValueError):
            disable_entries(SimpleNamespace(patterns={}), pattern_prefix='_sfdp_pattern_1')


if __name__ == '__main__':
    unittest.main()
