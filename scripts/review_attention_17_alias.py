#!/usr/bin/env python3
"""零设备复核冻结版本17/15的推理注册等价性；不伪造17独立命中。"""
from __future__ import annotations
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
DEST = ROOT/'results/current/T-105/pattern17_alias_review'
SOURCES = {
    'fuse_attention.py': 'torch/_inductor/fx_passes/fuse_attention.py',
    'pattern_matcher.py': 'torch/_inductor/pattern_matcher.py',
    'serialized15.py': 'torch/_inductor/fx_passes/serialized_patterns/_sfdp_pattern_15.py',
    'serialized17.py': 'torch/_inductor/fx_passes/serialized_patterns/_sfdp_pattern_17.py',
}


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def proof(sources):
    tree = ast.parse(sources['fuse_attention.py'])
    policy = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == '_INFERENCE_ONLY_SFDP_PATTERNS' for t in n.targets))
    assert '_sfdp_pattern_17' in [n.value for n in ast.walk(policy) if isinstance(n, ast.Constant)]
    a = re.sub(r'_sfdp_pattern_15(?=_)', '_canonical', sources['serialized15.py'])
    b = re.sub(r'_sfdp_pattern_17(?=_)', '_canonical', sources['serialized17.py'])
    assert ast.dump(ast.parse(a)) == ast.dump(ast.parse(b)), '序列化匹配结构不再等价'
    assert '_training' not in sources['serialized17.py']
    replacements = []
    for number in (15, 17):
        node = copy.deepcopy(function(tree, f'_sfdp_replacement_{number}'))
        node.name = '_canonical'
        if number == 17:
            node.args.args = [a for a in node.args.args if a.arg != 'dropout_p']
            class ZeroDropout(ast.NodeTransformer):
                def visit_Name(self, item):
                    return ast.Constant(0.0) if item.id == 'dropout_p' else item
            node = ZeroDropout().visit(node)
        replacements.append(ast.dump(node))
    assert replacements[0] == replacements[1], 'dropout=0后替换实现不等价'
    generator = function(tree, '_get_sfdp_patterns')
    candidates = [n for n in ast.walk(generator) if isinstance(n, ast.Tuple) and n.elts
                  and isinstance(n.elts[0], ast.Name) and n.elts[0].id in ('_sfdp_pattern_15', '_sfdp_pattern_17')]
    assert len(candidates) == 2 and candidates[0].elts[0].id == '_sfdp_pattern_15'
    assert ast.dump(candidates[0].elts[2]) == ast.dump(candidates[1].elts[2]), 'trace样例不同'
    assert ast.dump(candidates[0].elts[-1]) == ast.dump(candidates[1].elts[-1]), 'extra_check不同'
    generator_text = ast.unparse(generator)
    assert 'pattern_name not in _INFERENCE_ONLY_SFDP_PATTERNS' in generator_text
    assert 'partialize_and_update_signature(pattern, dropout_p=0.0)' in generator_text
    assert "'skip_duplicates': True" in generator_text
    matcher = ast.parse(sources['pattern_matcher.py'])
    register = ast.unparse(function(matcher, 'register_replacement'))
    duplicate = ast.unparse(function(matcher, 'check_and_add_duplicate_pattern'))
    generated = ast.unparse(function(matcher, 'gen_register_replacement'))
    assert 'search_fn_pattern=pat' in generated and 'gm = None' in register
    assert 'check_and_add_duplicate_pattern' in register
    assert 'if graph is None:\n        if skip_duplicates:\n            return True' in duplicate
    return dict(inference_only=True, serialized_patterns_equal=True,
                replacement_equal_at_zero_dropout=True, same_examples_and_extra_check=True,
                earlier_canonical=15, duplicate_policy='precompiled-pattern-graph-none-skip-duplicates')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pytorch-root', type=Path)
    parser.add_argument('--check-current', action='store_true')
    args = parser.parse_args()
    if args.check_current:
        record = json.loads((DEST/'review.json').read_text())
        sources = {}
        for name, info in record['sources'].items():
            data = (DEST/name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == info['sha256']
            sources[name] = data.decode()
        assert proof(sources) == record['proof']
        assert record['pytorch_commit'] == COMMIT
        print('pattern17_alias_review=OK device_execution=false canonical_pattern=15')
        return
    if args.pytorch_root is None:
        parser.error('生成复核需要冻结源码目录')
    sources = {name: subprocess.check_output(['git','-C',str(args.pytorch_root),'show',f'{COMMIT}:{path}']).decode()
               for name, path in SOURCES.items()}
    findings = proof(sources)
    DEST.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, source in sources.items():
        data = source.encode()
        (DEST/name).write_bytes(data)
        hashes[name] = {'upstream_path': SOURCES[name], 'sha256': hashlib.sha256(data).hexdigest()}
    record = dict(generated_at=datetime.now().astimezone().isoformat(), pytorch_commit=COMMIT,
                  status='duplicate-inference-contract-not-independent-trigger', sources=hashes, proof=findings,
                  canonical_acceptance_unit_id='AU-fuse-attention-sfdp-pattern-15',
                  device_execution=False, independent_unit_contribution=0,
                  boundary='只适用于冻结版本默认推理注册；不声明非零dropout训练支持，不回填原GPU/NPU verdict')
    (DEST/'review.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('pattern17_alias_review=written device_execution=false')


if __name__ == '__main__':
    main()
