#!/usr/bin/env python3
"""E8M0 部署前数值域检查：保存实际 eager/compile 与独立数学编码，不签性能门禁。"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import struct


def exact_positive_encoding(value):
    if not math.isfinite(value) or value <= 0:
        return None
    mantissa, exponent = math.frexp(value)
    ceiling = exponent - 1 if mantissa == 0.5 else exponent
    return min(254, max(0, ceiling + 127))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-dir', type=Path, required=True)
    p.add_argument('--expect-repaired', action='store_true')
    args = p.parse_args()
    if Path.cwd() != Path('/home/z50063656/tmp'):
        p.error('必须从 tmp 启动')
    out = args.artifact_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TORCHINDUCTOR_NPU_BACKEND='triton_experimental', TORCH_COMPILE_DEBUG='1',
        TORCH_COMPILE_DEBUG_DIR=str(out/'debug'), TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'),
        TRITON_CACHE_DIR=str(out/'triton-cache'), TORCHINDUCTOR_FORCE_DISABLE_CACHES='1',
        TORCHINDUCTOR_COMPILE_THREADS='1')
    import torch
    import torch_npu
    from torch_npu.utils._dynamo import register_inductor_npu, _InductorNpuRegistry
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    torch.npu.set_device(0)
    bits = [0, 0x80000000, 1, 0x3fffff, 0x400000, 0x400001, 0x7fffff, 0x800000,
            0x800001, 0x3f000000, 0x3f800000, 0x3f800001, 0x41000001, 0x7f7fffff,
            0x7f800000, 0xff800000, 0xbf800000, 0xff7fffff, 0x80000001, 0x7fc00000,
            0xffc00000, 0x7f800001]
    values = [struct.unpack('<f', struct.pack('<I', b))[0] for b in bits]
    inp = torch.tensor(values, dtype=torch.float32, device='npu')

    def fn(x):
        return (torch.clamp(torch.ceil(torch.log2(x)), -127, 127) + 127).to(torch.uint8)

    eager = fn(inp)
    actual = torch.compile(fn, fullgraph=True)(inp)
    torch.npu.synchronize()
    rows = [dict(bits=f'{b:08x}', value=str(v), eager=e, compiled=c, exact_positive=exact_positive_encoding(v))
            for b, v, e, c in zip(bits, values, eager.tolist(), actual.tolist())]
    passed = all(r['compiled'] == r['exact_positive'] for r in rows if r['exact_positive'] is not None)
    record = dict(generated_at=datetime.now().astimezone().isoformat(), artifact_dir=str(out), backend=_InductorNpuRegistry._loaded_backend,
        pytorch_commit=torch.version.git_version, torch_npu_file=torch_npu.__file__, pid=os.getpid(),
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), rows=rows,
        positive_mathematical_contract_passed=passed, expected_repaired=args.expect_repaired,
        scope='派生输入范围诊断，不增加社区分母、不签性能门禁')
    (out/'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    if args.expect_repaired and not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
