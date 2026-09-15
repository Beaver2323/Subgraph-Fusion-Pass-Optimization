"""隔离注册候选的边界测试；使用假模块，不导入torch或执行设备。"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


class CandidateScopeTests(unittest.TestCase):
    def run_candidate(self, name, device):
        source=Path(__file__).resolve().parents[1]/'runners/attention_registration_candidate.py'
        spec=importlib.util.spec_from_file_location('candidate_scope_test',source)
        candidate=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        class Tensor:
            def __init__(self):
                self.device=types.SimpleNamespace(type=device)
        torch=types.ModuleType('torch'); torch.Tensor=Tensor
        inductor=types.ModuleType('torch._inductor')
        fx=types.ModuleType('torch._inductor.fx_passes')
        pm=types.ModuleType('torch._inductor.pattern_matcher')
        pm.gen_pattern_and_search_gm=Mock(return_value=('actual-npu-pattern',types.SimpleNamespace(code='graph')))
        pm.register_replacement=Mock(return_value='new-registration')
        fa=types.ModuleType('torch._inductor.fx_passes.fuse_attention')
        original=Mock(return_value='unchanged-registration')
        fa.gen_register_replacement=original
        inductor.pattern_matcher=pm; fx.fuse_attention=fa
        torch._inductor=inductor; inductor.fx_passes=fx
        modules={'torch':torch,'torch._inductor':inductor,'torch._inductor.pattern_matcher':pm,
                 'torch._inductor.fx_passes':fx,'torch._inductor.fx_passes.fuse_attention':fa}
        guard=object(); replacement=object(); trace=object(); tensor=Tensor()
        kwargs=dict(example_inputs=[tensor],search_fn=object(),replace_fn=replacement,
                    trace_fn=trace,extra_check=guard,scalar_workaround={'dropout_p':0.1})
        with patch.dict(sys.modules,modules), tempfile.TemporaryDirectory() as temp:
            records=candidate.install(1,Path(temp)/'registration')
            returned=fa.gen_register_replacement(name,**kwargs)
        return original,pm,records,returned,kwargs

    def test_cuda_inference_and_neighbor_remain_unchanged(self):
        for name,device in (('_sfdp_pattern_1_training','cuda'),
                            ('_sfdp_pattern_1_inference','npu'),
                            ('_sfdp_pattern_10_training','npu')):
            with self.subTest(name=name,device=device):
                original,pm,records,returned,kwargs=self.run_candidate(name,device)
                original.assert_called_once_with(name,**kwargs)
                pm.register_replacement.assert_not_called()
                self.assertEqual(records,[])
                self.assertEqual(returned,'unchanged-registration')

    def test_only_npu_exact_training_uses_actual_decomposition(self):
        original,pm,records,returned,kwargs=self.run_candidate('_sfdp_pattern_1_training','npu')
        original.assert_not_called()
        actual=pm.register_replacement.call_args.kwargs
        self.assertIs(actual['extra_check'],kwargs['extra_check'])
        self.assertIs(actual['replace_fn'],kwargs['replace_fn'])
        self.assertIs(actual['trace_fn'],kwargs['trace_fn'])
        self.assertEqual(actual['scalar_workaround'],kwargs['scalar_workaround'])
        self.assertEqual(actual['search_fn_pattern'],'actual-npu-pattern')
        self.assertEqual(actual['pattern_name'],'_sfdp_pattern_1_training')
        self.assertFalse(records[0]['product_gate_bypassed'])
        self.assertEqual(returned,'new-registration')
