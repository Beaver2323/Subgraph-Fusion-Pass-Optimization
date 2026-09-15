"""共享mask切片诊断候选严格限定到本轮四个加载，不导入torch。"""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'attention_slice_diagnostic',
    Path(__file__).resolve().parents[1]/'runners/attention_22_codegen_diagnostic.py',
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SliceDiagnosticTests(unittest.TestCase):
    def test_changes_only_four_mask_broadcast_extents(self):
        original = module.SOURCE.read_text()
        updated = module.broadcast_slice_candidate(original)
        before, after = original.splitlines(), updated.splitlines()
        changed = [(a,b) for a,b in zip(before, after) if a != b]
        self.assertEqual(len(before), len(after))
        self.assertEqual(len(changed), 4)
        for a,b in changed:
            self.assertEqual(b, a.replace('[real_block_x2, real_block_x1, 1]', '[1, real_block_x1, 1]'))

    def test_rejects_unknown_generated_graph(self):
        with self.assertRaises(ValueError):
            module.broadcast_slice_candidate('some unrelated generated code')


if __name__ == '__main__':
    unittest.main()
