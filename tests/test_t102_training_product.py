"""产品注册窄范围、幂等及原 guard/trace 参数保留；不导入 torch。"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


class TrainingProductTests(unittest.TestCase):
    def exercise(self, name, devices, **extra):
        path = Path(__file__).resolve().parents[1]/'issues/REF-sfdp-pattern-1-native/candidate/sfdp_training.py'
        spec = importlib.util.spec_from_file_location('training_product_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Tensor:
            def __init__(self, device):
                self.device = types.SimpleNamespace(type=device)
        torch = types.ModuleType('torch'); torch.Tensor = Tensor
        pm = types.ModuleType('torch._inductor.pattern_matcher')
        pm.gen_pattern_and_search_gm = Mock(return_value=('npu-pattern', None))
        pm.register_replacement = Mock(return_value=True)
        fa = types.ModuleType('torch._inductor.fx_passes.fuse_attention')
        original = Mock(return_value='unchanged'); original._npu_training_patterns = False
        fa.gen_register_replacement = original
        ind = types.ModuleType('torch._inductor'); ind.pattern_matcher = pm
        fx = types.ModuleType('torch._inductor.fx_passes'); fx.fuse_attention = fa
        kwargs = dict(example_inputs=tuple(Tensor(d) for d in devices), search_fn=object(),
                      replace_fn=object(), trace_fn=object(), extra_check=object(),
                      scalar_workaround={'dropout_p':0.113377}, skip_duplicates=True, **extra)
        with patch.dict(sys.modules, {'torch':torch, 'torch._inductor':ind,
                        'torch._inductor.fx_passes':fx}):
            module.install_sfdp_training_patterns()
            once = fa.gen_register_replacement
            module.install_sfdp_training_patterns()
            self.assertIs(once, fa.gen_register_replacement)
            fa.gen_register_replacement(name, **kwargs)
        return original, pm, kwargs

    def test_reviewed_training_and_original_guards(self):
        for number in range(1,6):
            for suffix in ('', '_half', '_bs1', '_half_bs1'):
                with self.subTest(number=number, suffix=suffix):
                    name = f'_sfdp_pattern_{number}{suffix}_training'
                    original, pm, kwargs = self.exercise(name, ['npu']*3,
                        exclusive_arg_names=('query',), get_decomp_fn=object())
                    original.assert_not_called()
                    actual = pm.register_replacement.call_args.kwargs
                    for key, value in kwargs.items():
                        self.assertIs(actual[key], value)
                    self.assertEqual(actual['pattern_name'], name)
                    self.assertEqual(actual['search_fn_pattern'], 'npu-pattern')
                    self.assertIs(pm.gen_pattern_and_search_gm.call_args.kwargs['get_decomp_fn'], kwargs['get_decomp_fn'])
                    self.assertEqual(pm.gen_pattern_and_search_gm.call_args.kwargs['exclusive_arg_names'], ('query',))

    def test_unreviewed_inference_cpu_cuda_and_mixed_are_untouched(self):
        cases = [('_sfdp_pattern_1_inference',['npu']), ('_sfdp_pattern_6_training',['npu']),
                 ('_sfdp_pattern_10_training',['npu']), ('_sfdp_pattern_1_mask_fp32_training',['npu']),
                 ('_sfdp_pattern_1_training',['cuda']), ('_sfdp_pattern_1_training',['cpu']),
                 ('_sfdp_pattern_1_training',['npu','cpu']), ('_sfdp_pattern_1_training',[])]
        for name, devices in cases:
            with self.subTest(name=name, devices=devices):
                original, pm, kwargs = self.exercise(name, devices)
                original.assert_called_once_with(name, **kwargs)
                pm.register_replacement.assert_not_called()
                pm.gen_pattern_and_search_gm.assert_not_called()
