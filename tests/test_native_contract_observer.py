"""原生精确观察器的标准库回归，不导入 torch 或执行设备。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


observer = load('contract_observer', 'runners/native_contract_observer.py')
reference = load('contract_reference', 'runners/reference_runner.py')
launcher = load('native_npu_launcher', 'scripts/run_prepared_npu_case.py')
split_adapter = load('split_npu_adapter', 'runners/t088_t090_npu_case.py')


class ContractObserverTests(unittest.TestCase):
    def test_negative_must_not_be_forced_to_rewrite(self):
        self.assertTrue(split_adapter.rewrite_contract([{'changed':False}], False))
        self.assertFalse(split_adapter.rewrite_contract([{'changed':True}], False))
        self.assertFalse(split_adapter.rewrite_contract([{'changed':False}], True))
    def test_precise_name_no_generic_pattern_match(self):
        self.assertEqual(observer.identity(SimpleNamespace(pattern_name='_sfdp_pattern_12_training')), '_sfdp_pattern_12_training')
        self.assertIsNone(observer.identity(SimpleNamespace(pattern_name='fuse_attention')))

    def test_records_real_entry_and_preserves_return(self):
        class Graph:
            text = 'before'
            def python_code(self, root):
                return SimpleNamespace(src=self.text)
        class Entry:
            pattern_name = '_sfdp_pattern_3_inference'
            def apply(self, match, graph, node):
                graph.text = 'after'
                return 42
        with tempfile.TemporaryDirectory() as tmp:
            observer.install(Entry, Path(tmp))
            self.assertEqual(Entry().apply(None, Graph(), None), 42)
            paths = list(Path(tmp).glob('*/contract_observation.json'))
            self.assertEqual(len(paths), 1)
            data = json.loads(paths[0].read_text())
            self.assertTrue(data['graph_changed'])
            self.assertFalse(data['product_gate_bypassed'])
            self.assertFalse(data['numerical_correctness_proven_by_observer'])

    def test_original_exception_is_not_swallowed(self):
        class Entry:
            pattern_name = '_sfdp_pattern_1_training'
            def apply(self, *args):
                raise RuntimeError('original failure')
        graph = SimpleNamespace(python_code=lambda root:SimpleNamespace(src='before'))
        with tempfile.TemporaryDirectory() as tmp:
            observer.install(Entry, Path(tmp))
            with self.assertRaisesRegex(RuntimeError,'original failure'):
                Entry().apply(None,graph,None)
            self.assertFalse(list(Path(tmp).glob('*/contract_observation.json')))

    def test_command_keeps_method_and_direct_mode(self):
        plan=json.loads((ROOT/'upstream/t102_reference_plan.yaml').read_text())
        command=reference.case_command(plan['cases'][0],ROOT,Path('/frozen'))
        self.assertIn('native_contract_observer.py',command[1])
        self.assertEqual(command[-1],'SDPAPatternRewriterGpuTests.test_sdpa_rewriter_1_gpu')

    def test_native_classification_does_not_count_skip_as_pass(self):
        self.assertEqual(launcher.classify(0,'Ran 1 test in 0s\nOK (skipped=1)\n')[0],'skipped-or-xfail')
        self.assertEqual(launcher.classify(1,'Ran 1 test in 0s\nFAILED (errors=1)')[0],'failed')
        self.assertEqual(launcher.classify(0,'')[0],'no-tests')
        self.assertEqual(launcher.classify(0,'Ran 1 test in 1s\nOK\n')[0],'native-test-passed-awaiting-contract-review')


if __name__ == '__main__':
    unittest.main()
