"""codegen隔离候选锚点和核心保护的零设备测试。"""
import importlib.util
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

spec=importlib.util.spec_from_file_location('select_slice_candidate',
    Path(__file__).resolve().parents[1]/'runners/attention_select_slice_candidate.py')
candidate=importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
FIXTURE = Path(__file__).resolve().parents[1]/'issues/REF-sfdp-pattern-22-native/evidence/t106-native-nzmb_etp/adapter/codegen-candidate'


def parse_load(line):
    node = ast.parse(line.strip()).body[0]
    if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
        return None
    return '', node.targets[0].id, node.value, node.value


def emit_candidate(line, extent=4, direct=False, indirect=False):
    # 只执行归档代码的文本发射方法；不导入 torch/Triton，不执行生成 kernel。
    sources = [(FIXTURE/f'before-{name}.py').read_text() for name in
               ('_maybe_record_select_lane_load', '_maybe_rewrite_select_lane_load')]
    _, rewritten = candidate.candidate_sources(*sources)
    ns = dict(ast=ast, ncfg=SimpleNamespace(select_extract_slice=True, select_extract_slice_strided=True),
              triton_codegen_linearize=True, _npu_parse_tl_load_assignment=parse_load)
    exec(compile(rewritten, '<isolated-codegen-text-emission>', 'exec'), ns)
    obj = SimpleNamespace(inside_reduction=False, _linearize_applied=True,
        _npu_select_lane_loads={'tmp0':dict(lane=2,index='index',pointer='in_ptr0',storage_size=extent)},
        range_trees=[SimpleNamespace(is_reduction=False,var_tensor_dims={'x2':0,'x1':1,'x0':2},
                                   node_block_constexpr={'x2':'B2','x1':'B1','x0':'B0'})],
        triton_tensor_ndim=lambda:3,indexing_size_str=lambda slot:'[None, None, :]',
        _select_index_node_relation=lambda index,name:(direct,indirect),
        _emit_strided_slice_extract=lambda *args:['unchanged-case-B'],
        body=SimpleNamespace(_lines=[line]),
        overrides=SimpleNamespace(extract_slice=lambda *args:'extract_slice('+', '.join(args)+')'))
    ns['_maybe_rewrite_select_lane_load'](obj)
    return obj.body._lines


class SelectSliceCandidateTests(unittest.TestCase):
    def test_unknown_source_fails_closed(self):
        with self.assertRaises(ValueError):
            candidate.candidate_sources('def wrong(): pass', 'def wrong(): pass')

    def test_repeated_anchor_fails_closed(self):
        with self.assertRaises(ValueError):
            candidate.replace_once('x x','x','new')

    def test_candidate_keeps_declared_guards(self):
        source=Path(candidate.__file__).read_text()
        for required in ('layout.offset == 0', 'storage_size', 'extent is None',
                         '.shape[{s}]', 'load_ast.args[1]', 'mask_kw',
                         'strided-slice branch unchanged'):
            self.assertIn(required,source)

    def test_emit_preserves_each_load_outer_shape(self):
        lines=emit_candidate('tmp0 = tl.load(in_ptr0 + 2*x1, original_mask)')
        self.assertIn('[_es_full0.shape[0], _es_full0.shape[1], 1]', lines[-1])
        self.assertNotIn('[B2, B1, 1]', lines[-1])

    def test_tail_bounds_for_all_mask_forms(self):
        for suffix in (', original_mask', ', mask=original_mask', ''):
            with self.subTest(mask=suffix):
                lines=emit_candidate('tmp0 = tl.load(in_ptr0 + (2*x1 + 1)'+suffix+')')
                load=ast.parse(lines[1]).body[0].value
                mask=load.args[1] if len(load.args)>1 else next(k.value for k in load.keywords if k.arg=='mask')
                expression=compile(ast.Expression(mask),'<bound-test>','eval')
                for lane,wanted in ((0,True),(1,False)):
                    self.assertEqual(eval(expression, {'__builtins__':{}},
                        dict(x1=1,_es_lane0=lane,original_mask=True)), wanted)
                if suffix:
                    self.assertFalse(eval(expression, {'__builtins__':{}},
                        dict(x1=0,_es_lane0=0,original_mask=False)))
                self.assertFalse(eval(expression, {'__builtins__':{}},
                    dict(x1=-1,_es_lane0=0,original_mask=True)))

    def test_unknown_extent_and_pointer_preserve_original_load(self):
        line='tmp0 = tl.load(in_ptr0 + 2*x1, original_mask)'
        self.assertEqual(emit_candidate(line,extent=None),[line])
        other='tmp0 = tl.load(other_ptr + 2*x1, original_mask)'
        self.assertEqual(emit_candidate(other),[other])
        nested='tmp0 = tl.load(in_ptr0 + 2*x1 + 1, original_mask)'
        self.assertEqual(emit_candidate(nested),[nested])

    def test_indirect_and_strided_paths_not_rewritten_as_select(self):
        line='tmp0 = tl.load(in_ptr0 + 2*x1, original_mask)'
        self.assertEqual(emit_candidate(line,indirect=True),[line])
        self.assertEqual(emit_candidate(line,direct=True),['unchanged-case-B'])


if __name__=='__main__':
    unittest.main()
