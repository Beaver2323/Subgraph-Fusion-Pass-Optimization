"""只读观察器统计原成功断言的容器Tensor叶子；不加载torch。"""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'runners/attention_community_npu.py'
spec = importlib.util.spec_from_file_location('attention_tensor_observer', path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class Tensor:
    pass


class TensorObserverTests(unittest.TestCase):
    def test_tuple_outputs_have_three_leaves(self):
        a = tuple(Tensor() for _ in range(3))
        b = tuple(Tensor() for _ in range(3))
        pairs = list(adapter.passed_tensor_pairs(a, b, Tensor))
        self.assertEqual([p[0] for p in pairs], [(0,), (1,), (2,)])
        self.assertIs(pairs[0][1], a[0])

    def test_nested_container_and_scalar_are_distinguished(self):
        a = {'out': [Tensor(), (Tensor(), 3)], 'metadata': 'ok'}
        b = {'out': [Tensor(), (Tensor(), 3)], 'metadata': 'ok'}
        paths = [p[0] for p in adapter.passed_tensor_pairs(a, b, Tensor)]
        self.assertEqual(paths, [('out', 0), ('out', 1, 0)])
        self.assertEqual(list(adapter.passed_tensor_pairs(3, 3, Tensor)), [])

    def test_plain_tensor_is_not_duplicated(self):
        self.assertEqual(len(list(adapter.passed_tensor_pairs(Tensor(), Tensor(), Tensor))), 1)

    def test_math_codegen_requires_exact_target_and_actual_calls(self):
        code='torch.npu.set_device(0)\nextern_kernels.bmm(a,b)\ntriton_safe_softmax.run(x)\nextern_kernels.bmm(c,d)\n'
        observation=[{'target':'_sfdp_pattern_14_inference','graph_changed':True}]
        self.assertTrue(adapter.reviewed_math_codegen(code, observation, 14))
        self.assertFalse(adapter.reviewed_math_codegen(code, observation, 5))
        self.assertFalse(adapter.reviewed_math_codegen(code, [], 14))
        self.assertFalse(adapter.reviewed_math_codegen(repr(code), observation, 14))
        self.assertFalse(adapter.reviewed_math_codegen(code+'x.cpu()\n', observation, 14))
        for number in (21,22,24):
            own=[{'target':f'_sfdp_pattern_{number}_inference','graph_changed':True}]
            self.assertTrue(adapter.reviewed_math_codegen(code, own, number))
            self.assertFalse(adapter.reviewed_math_codegen(code, observation, number))
