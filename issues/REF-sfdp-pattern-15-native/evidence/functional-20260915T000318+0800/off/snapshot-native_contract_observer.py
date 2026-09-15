#!/usr/bin/env python3
"""原生 GPU 合同只读观察：记录已通过 guard 的精确 replacement/handler 及阶段前后图。"""
from __future__ import annotations
import argparse
import functools
import json
import os
from pathlib import Path
import runpy
import sys


def identity(entry):
    name = getattr(entry, 'pattern_name', '')
    if isinstance(name, str) and name.startswith('_sfdp_pattern_'):
        return name
    handler = getattr(entry, 'handler', None)
    name = getattr(handler, '__name__', '')
    return name if name == 'normalize_stack_default' else None


def install(entry_type, root: Path):
    original = entry_type.apply
    serial = 0
    @functools.wraps(original)
    def observed(self, match, graph, node):
        nonlocal serial
        name = identity(self)
        if name is None:
            return original(self, match, graph, node)
        serial += 1
        dest = root/f'{os.getpid()}-{entry_type.__name__}-{serial:04d}'
        dest.mkdir(parents=True, exist_ok=False)
        before = graph.python_code('self').src
        (dest/'fx_graph_readable.py').write_text(before)
        result = original(self, match, graph, node)
        after = graph.python_code('self').src
        (dest/'fx_graph_transformed.py').write_text(after)
        (dest/'contract_observation.json').write_text(json.dumps({
            'capture_scope':'pattern-entry-apply-after-extra-check', 'target':name,
            'entry_type':entry_type.__name__, 'handler_returned':True,
            'graph_changed':before != after, 'test_body_modified':False,
            'device_modified':False, 'assertions_modified':False,
            'product_gate_bypassed':False, 'numerical_correctness_proven_by_observer':False,
        },ensure_ascii=False,indent=2)+'\n')
        return result
    entry_type.apply = observed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('methods', nargs=argparse.REMAINDER)
    args = p.parse_args()
    methods = args.methods[1:] if args.methods[:1] == ['--'] else args.methods
    if not methods or args.source.name not in {'test_fused_attention.py','test_split_cat_fx_passes.py'}:
        p.error('只允许已审核的 attention/stack 社区入口和显式方法')
    from torch._inductor import pattern_matcher
    root = Path(os.environ['TORCH_COMPILE_DEBUG_DIR'])/'native_contract_observer'
    install(pattern_matcher.GraphPatternEntry, root)
    install(pattern_matcher.ReplacementPatternEntry, root)
    sys.path.insert(0,str(args.source.resolve().parent))
    sys.argv = [str(args.source),'-v',*methods]
    runpy.run_path(str(args.source),run_name='__main__')


if __name__ == '__main__':
    main()
