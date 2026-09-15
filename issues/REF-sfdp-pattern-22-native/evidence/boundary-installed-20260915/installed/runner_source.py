#!/usr/bin/env python3
"""T-106 select-load 独立设备边界回归；installed 臂不加载候选，非性能测试。"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def emitted_source(lines):
    # DeferredLine 仍由 Inductor 在原生命周期展开；观察器只读已发射字符串。
    return '\n'.join(line for line in lines if isinstance(line, str))


def scenarios():
    for dtype in ('float32', 'float16', 'bfloat16'):
        for width in (2, 4, 8):
            for lane in (0, width-1):
                yield dict(dtype=dtype, width=width, lane=lane, layout='shared', dynamic=False)
    for layout in ('offset', 'strided', 'full'):
        yield dict(dtype='float32', width=4, lane=3, layout=layout, dynamic=False)
    yield dict(dtype='float32', width=4, lane=3, layout='shared', dynamic=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=('baseline', 'candidate', 'installed'), required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != Path('/home/z50063656/tmp'):
        parser.error('必须从 /home/z50063656/tmp 执行')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(TORCHINDUCTOR_NPU_BACKEND='triton_experimental',
                      TORCH_COMPILE_DEBUG='1', TORCHINDUCTOR_COMPILE_THREADS='1',
                      TORCHINDUCTOR_FORCE_DISABLE_CACHES='1',
                      TORCH_COMPILE_DEBUG_DIR=str(output/'debug'),
                      TORCHINDUCTOR_CACHE_DIR=str(output/'inductor-cache'),
                      TRITON_CACHE_DIR=str(output/'triton-cache'))
    (output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    import torch
    import torch_npu
    from torch_npu.utils._dynamo import register_inductor_npu, _InductorNpuRegistry
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    assert torch.version.git_version == '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
    torch.npu.set_device(0)
    from torch_npu._inductor.triton_experimental.codegen import triton as codegen
    from torch_npu._inductor.triton_experimental import config
    assert config.select_extract_slice and config.select_extract_slice_strided
    product = Path(codegen.__file__)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    before = digest(product)
    (output/'installed_triton_source.py').write_bytes(product.read_bytes())
    candidate = None
    if args.arm == 'candidate':
        from attention_select_slice_candidate import install
        candidate = install(output/'candidate')
    records = []
    emitted = []
    current = None
    original = codegen.NPUTritonKernel._maybe_rewrite_select_lane_load

    def observe(kernel):
        result = original(kernel)
        lines = emitted_source(kernel.body._lines)
        if '_es_full' in lines:
            path = output/f'emitted-{len(emitted):03d}.txt'
            path.write_text(lines)
            emitted.append(dict(case=current, file=path.name, sha256=digest(path),
                                per_load_shape='.shape[' in lines,
                                bounded_load='>= 0' in lines))
        return result
    codegen.NPUTritonKernel._maybe_rewrite_select_lane_load = observe
    for index, case in enumerate(scenarios()):
        current = index
        row = dict(index=index, **case, status='running', executions=[])
        records.append(row)
        try:
            torch._dynamo.reset()
            width, lane = case['width'], case['lane']

            def fn(source, addend):
                return source.select(-1, lane).unsqueeze(-1) + addend

            compiled = torch.compile(fn, fullgraph=True, dynamic=case['dynamic'])
            for rows in ((7, 11) if case['dynamic'] else (7,)):
                outer = 3 if case['layout'] == 'full' else 1
                dtype = getattr(torch, case['dtype'])
                if case['layout'] == 'offset':
                    base = torch.arange(outer*rows*width+1, device='npu', dtype=torch.float32)
                    source = base.to(dtype)[1:].reshape(outer, rows, width)
                elif case['layout'] == 'strided':
                    source = torch.arange(outer*rows*width*2, device='npu', dtype=torch.float32).to(dtype).reshape(outer,rows,width*2)[...,::2]
                else:
                    source = torch.arange(outer*rows*width, device='npu', dtype=torch.float32).to(dtype).reshape(outer,rows,width)
                addend = torch.ones((3, rows, 5), device='npu', dtype=dtype)
                expected = fn(source, addend)
                actual = compiled(source, addend)
                torch.npu.synchronize()
                same = bool(torch.equal(actual, expected))
                row['executions'].append(dict(rows=rows, storage_offset=source.storage_offset(),
                    stride=list(source.stride()), output_shape=list(actual.shape), bitwise_equal=same,
                    mismatch=int((actual != expected).sum().item())))
                assert same, 'select+broadcast单次加法与同设备eager不一致'
            row['status'] = 'passed'
        except Exception:
            row['status'] = 'failed'
            row['traceback'] = traceback.format_exc()
        print(json.dumps(row, ensure_ascii=False), flush=True)
        (output/'progress.json').write_text(json.dumps(dict(status='running-not-a-verdict',cases=records),ensure_ascii=False,indent=2)+'\n')
    after = digest(product)
    target_exercised = bool(emitted) and (args.arm == 'baseline' or any(x['per_load_shape'] and x['bounded_load'] for x in emitted))
    passed = all(x['status']=='passed' for x in records) and target_exercised and before == after
    result = dict(generated_at=datetime.now().astimezone().isoformat(),
                  status='passed' if passed else 'failed', arm=args.arm, device_execution=True,
                  backend=_InductorNpuRegistry._loaded_backend, pid=os.getpid(),
                  physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                  python=sys.executable, pytorch_commit=torch.version.git_version,
                  torch_npu_version=torch_npu.__version__, candidate=candidate,
                  installed_before=before, installed_after=after, source_sha256=digest(output/'runner_source.py'),
                  target_exercised=target_exercised, cases=records, emitted=emitted,
                  limitation='设备数值/编译边界回归，不是内存sanitizer；不证明所有动态布局，不替代完整社区原例/邻接或性能验收')
    (output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
