#!/usr/bin/env python3
"""T-098 裁剪定位实验：复用社区 ConvOp，不替代原 test_basic 或签性能 gate。"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf32', choices=('default', 'off'), required=True)
    parser.add_argument('--mode', choices=('off', 'on'), default='on')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != Path('/home/z50063656/tmp'):
        parser.error('必须从 /home/z50063656/tmp 启动')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(TORCHINDUCTOR_NPU_BACKEND='triton_experimental',
                      TORCHINDUCTOR_FORCE_DISABLE_CACHES='1', TORCHINDUCTOR_COMPILE_THREADS='1',
                      TORCH_COMPILE_DEBUG='1', TORCH_COMPILE_DEBUG_DIR=str(output/'debug'),
                      TORCHINDUCTOR_CACHE_DIR=str(output/'inductor-cache'),
                      TRITON_CACHE_DIR=str(output/'triton-cache'))
    import torch
    import torch_npu
    from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    assert torch.version.git_version == '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
    torch.npu.set_device(0)
    initial_hf32 = torch.npu.conv.allow_hf32
    if args.hf32 == 'off':
        torch.npu.conv.allow_hf32 = False
    pytorch = Path(torch.__file__).resolve().parents[1]
    sys.path.insert(0, str(pytorch/'test'))
    from inductor.test_efficient_conv_bn_eval import ConvOp
    from torch._inductor.fx_passes.efficient_conv_bn_eval import efficient_conv_bn_eval
    from torch._inductor import config
    from torch._dynamo.utils import counters
    def compare(left, right):
        left, right = left.detach().cpu(), right.detach().cpu()
        delta = (left-right).abs()
        failed = ~torch.isclose(left, right, rtol=1.3e-6, atol=1e-5, equal_nan=False)
        return dict(passed=not bool(failed.any()), mismatch=int(failed.sum()), numel=left.numel(),
                    max_abs=float(delta.max()), atol=1e-5, rtol=1.3e-6)
    samples = []
    for seed in (20260914, 20260915, 20260916):
        torch.manual_seed(seed)
        model = ConvOp(torch.nn.Conv1d, torch.nn.BatchNorm1d, True, 3, 32,
                       'npu:0', kernel_size=3, stride=2).eval()
        x = torch.rand(4, 3, 96).to('npu:0')
        eager = model(x)
        folded = efficient_conv_bn_eval(model.bn, model.conv, x)
        sample = dict(seed=seed, eager_folded_vs_eager=compare(folded, eager))
        if args.compile:
            torch._dynamo.reset()
            counters.clear()
            with config.patch(efficient_conv_bn_eval_fx_passes=args.mode=='on',
                              freezing=False, fx_graph_cache=False):
                compiled = torch.compile(copy.deepcopy(model), fullgraph=True, backend='inductor',
                                         options={'npu_backend':'triton_experimental'})
                actual = compiled(x)
                torch.npu.synchronize()
                sample.update(compiled_vs_eager=compare(actual,eager),
                              compiled_vs_eager_folded=compare(actual,folded),
                              target_counter=counters['inductor']['efficient_conv_bn_eval'])
        samples.append(sample)
        print(json.dumps(sample, ensure_ascii=False), flush=True)
    files = [Path(__file__), pytorch/'test/inductor/test_efficient_conv_bn_eval.py',
             pytorch/'torch/_inductor/fx_passes/efficient_conv_bn_eval.py',
             Path(torch_npu.__file__).parent/'npu/npu_config.py']
    result = dict(generated_at=datetime.now().astimezone().isoformat(),
                  experiment='conv1d-forward-precision-diagnostic-not-community-full-contract',
                  backend=_InductorNpuRegistry._loaded_backend, pytorch_commit=torch.version.git_version,
                  device_execution=True, pid=os.getpid(), physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                  initial_conv_hf32=initial_hf32, effective_conv_hf32=torch.npu.conv.allow_hf32,
                  matmul_hf32=torch.npu.matmul.allow_hf32, compile_enabled=args.compile, mode=args.mode,
                  input_shape=[4,3,96], dtype='float32', samples=samples, product_gate_bypassed=False,
                  full_community_contract_passed=False, performance_gate_issued=False,
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
