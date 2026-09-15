#!/usr/bin/env python3
"""重放本机 pattern22 生成代码，逐kernel比较；只作定位，不能代替原合同验收。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'issues/REF-sfdp-pattern-22-native/evidence/t106-native-xtdt3xs0/adapter/debug/torch_compile_debug/run_2026_09_14_22_59_03_658879-pid_5397/torchinductor/model__0_inference_0.0/output_code.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if Path.cwd() != Path('/home/z50063656/tmp'):
        p.error('从 /home/z50063656/tmp 执行')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'triton_experimental'
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(output/'inductor-cache')
    os.environ['TRITON_CACHE_DIR'] = str(output/'triton-cache')
    (output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    (output/'generated_source.py').write_bytes(SOURCE.read_bytes())
    import torch
    import torch_npu
    from torch_npu.utils._dynamo import _InductorNpuRegistry, register_inductor_npu
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    assert torch.version.git_version == '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
    torch.npu.set_device(0)
    spec = importlib.util.spec_from_file_location('local_attention22_generated', output/'generated_source.py')
    generated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated)
    events = []
    current_seed = None

    def compare(actual, expected):
        torch.npu.synchronize()
        if actual.dtype == torch.bool:
            close = actual == expected
            error = float((actual != expected).any().item())
        else:
            close = torch.isclose(actual, expected, atol=1e-5, rtol=1.3e-6)
            error = float((actual-expected).abs().max().item())
        return dict(shape=list(actual.shape), dtype=str(actual.dtype),
                    finite=bool(torch.isfinite(actual).all().item()),
                    mismatch=int((~close).sum().item()), numel=actual.numel(), max_abs_error=error)

    class Capture:
        def __init__(self, original, index, name):
            self.original, self.index, self.name = original, index, name

        def run(self, *values, **kwargs):
            i = self.index
            if i == 0:
                expected, actual = values[0].permute(0,2,1,3).contiguous(), values[1]
            elif i == 1:
                expected, actual = values[0].permute(0,2,3,1).contiguous(), values[1]
            elif i in (2,3,4):
                actual = values[2] if i in (2,3) else values[3]
                scores = values[0].reshape(4,16,2,2) + values[1]
                if i == 2:
                    expected = scores != -float('inf')
                elif i == 3:
                    expected = scores.amax(-1,keepdim=True).expand_as(actual)
                else:
                    expected = (scores-values[2]).exp().sum(-1,keepdim=True).expand_as(actual)
            elif i == 5:
                actual = values[1]
                expected = (~values[0]).expand_as(actual)
            elif i == 6:
                actual = values[0]
                before = actual.clone()
                expected = torch.where(values[1], 0., (before+values[2]-values[3]).exp()/values[4])
            elif i == 7:
                expected, actual = values[0].permute(0,2,1,3).contiguous(), values[1]
            else:
                raise ValueError('未审查的kernel')
            torch.npu.synchronize()
            result = self.original.run(*values, **kwargs)
            event = dict(seed=current_seed, kernel=self.name, **compare(actual, expected))
            events.append(event)
            print(json.dumps(event,ensure_ascii=False),flush=True)
            return result

    kernels = sorted((name,value) for name,value in vars(generated).items()
                     if name.startswith('triton_unk_') and hasattr(value,'run'))
    assert len(kernels) == 8
    for name, value in kernels:
        setattr(generated, name, Capture(value, int(name.rsplit('_',1)[1]), name))
    summaries = []
    for seed in (20260914,20260915,20260916):
        current_seed = seed
        torch.manual_seed(seed)
        mask = torch.randn((1,1,2,2),device='npu')
        q,k,v = [torch.randn((4,2,16,32),device='npu') for _ in range(3)]
        qp,kp,vp = [x.transpose(1,2) for x in (q,k,v)]
        eager = (qp @ kp.transpose(-2,-1) + mask).float().softmax(-1) @ vp
        result = generated.call([mask,k,q,v])
        summaries.append(dict(seed=seed, output=compare(result[0],eager),
                              key=compare(result[1],kp), value=compare(result[2],vp)))
    record = dict(generated_at=datetime.now().astimezone().isoformat(),
                  status='instrumented-codegen-replay-not-community-verdict',
                  backend='triton_experimental', pid=os.getpid(),
                  physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                  python_executable=sys.executable, pytorch_commit=torch.version.git_version,
                  torch_npu_version=torch_npu.__version__, source_path=str(SOURCE),
                  generated_source_sha256=digest(SOURCE), diagnostic_source_sha256=digest(Path(__file__)),
                  events=events, summaries=summaries,
                  caveat='逐kernel额外同步和eager对照会改变时序；不改原kernel，不是原例复验或性能测量')
    (output/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'summaries':summaries},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
