#!/usr/bin/env python3
"""T-102 注册修复的真实设备边界；不改变原社区测例或允许失败后调容差。"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidate', type=Path)
    args = p.parse_args()
    if Path.cwd() != Path('/home/z50063656/tmp') or os.environ.get('TORCHINDUCTOR_NPU_BACKEND') != 'triton_experimental':
        p.error('从 tmp 启动并在导入前指定 triton_experimental')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TORCH_COMPILE_DEBUG='1', TORCH_COMPILE_DEBUG_DIR=str(out/'debug'),
                      TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'), TRITON_CACHE_DIR=str(out/'triton-cache'),
                      TORCHINDUCTOR_COMPILE_THREADS='1', TORCHINDUCTOR_FORCE_DISABLE_CACHES='1')
    import torch
    import torch_npu
    from torch_npu.utils._dynamo import register_inductor_npu, _InductorNpuRegistry
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    assert torch.version.git_version == '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
    torch.npu.set_device(0)
    if args.candidate:
        spec = importlib.util.spec_from_file_location('training_boundary_candidate', args.candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        from torch_npu._inductor.triton_experimental import sfdp_training as module
    module.install_sfdp_training_patterns()
    from torch._inductor import pattern_matcher
    from native_contract_observer import install
    install(pattern_matcher.ReplacementPatternEntry, out/'target-observations')

    def div(q, k, v, extra):
        return (q @ k.transpose(-2,-1)).div(4.0).softmax(-1) @ v

    def mul(q, k, v, extra):
        return (q @ k.transpose(-2,-1)).mul(0.25).softmax(-1) @ v

    def reuse(q, k, v, extra):
        weights = (q @ k.transpose(-2,-1)).div(4.0).softmax(-1)
        return weights @ v, weights

    def tensor_scale(q, k, v, extra):
        return ((q @ k.transpose(-2,-1)) * extra).softmax(-1) @ v

    def mask(q, k, v, extra):
        return ((q @ k.transpose(-2,-1)).div(4.0) + extra).softmax(-1) @ v

    scenarios = [('div-fp32-training', div, torch.float32, False, 'scalar', 1),
                 ('mul-fp16-strided-training', mul, torch.float16, True, 'scalar', 2),
                 ('reused-intermediate-negative', reuse, torch.float32, False, 'scalar', None),
                 ('tensor-scale-negative', tensor_scale, torch.float32, False, 'tensor', None),
                 ('add-mask-fp32-training', mask, torch.float32, False, 'mask', 5),
                 ('scalar-mask-negative', mask, torch.float32, False, 'tensor', None)]
    rows = []
    for index, (name, fn, dtype, strided, extra_kind, target) in enumerate(scenarios):
        torch._dynamo.reset()
        torch.manual_seed(20260915 + index)
        inputs = []
        for _ in range(3):
            value = torch.randn((2,2,16,8) if strided else (2,2,8,16), device='npu', dtype=dtype) * 0.25
            inputs.append((value.transpose(-1,-2) if strided else value).detach().requires_grad_(True))
        other = [v.detach().clone(memory_format=torch.preserve_format).requires_grad_(True) for v in inputs]
        extra = (torch.zeros((8,8), device='npu', dtype=dtype) if extra_kind == 'mask' else
                 torch.tensor(0.25, device='npu', dtype=dtype) if extra_kind == 'tensor' else 0.0)
        prior = set((out/'target-observations').glob('*/contract_observation.json'))
        eager = fn(*inputs, extra)
        compiled = torch.compile(fn, fullgraph=True)(*other, extra)
        left = eager if isinstance(eager, tuple) else (eager,)
        right = compiled if isinstance(compiled, tuple) else (compiled,)
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, atol=0.001, rtol=0.2)
        sum(x.sum() for x in left).backward()
        sum(x.sum() for x in right).backward()
        for a, b in zip(inputs, other):
            torch.testing.assert_close(a.grad, b.grad, atol=0.001, rtol=0.2)
        torch.npu.synchronize()
        paths = set((out/'target-observations').glob('*/contract_observation.json')) - prior
        observations = [json.loads(path.read_text()) for path in sorted(paths)]
        changed = [r['target'] for r in observations if r['graph_changed'] and r['target'].startswith('_sfdp_pattern_')]
        assert (any(n.startswith(f'_sfdp_pattern_{target}_') for n in changed) if target else not changed), (name, changed)
        rows.append(dict(name=name, passed=True, target=target, actual_targets=changed,
                         tensor_comparisons=len(left)+3, dtype=str(dtype), strides=[list(v.stride()) for v in inputs]))
        print(name, 'PASS', flush=True)
    record = dict(generated_at=datetime.now().astimezone().isoformat(), status='passed',
                  backend='triton_experimental', pid=os.getpid(), physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                  candidate=bool(args.candidate), product_module=str(Path(module.__file__).resolve()),
                  product_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  pytorch_commit=torch.version.git_version, scenarios=rows, numerical_tolerance={'atol':0.001,'rtol':0.2})
    (out/'harness_source.py').write_bytes(Path(__file__).read_bytes())
    (out/'candidate_source.py').write_bytes(Path(module.__file__).read_bytes())
    (out/'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
