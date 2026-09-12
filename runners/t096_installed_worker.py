#!/usr/bin/env python3
"""T-096 安装态精确开关、dtype/device/layout 边界及原七元素子图性能臂。"""
import argparse
from datetime import datetime
import functools
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from unittest import mock

from t087_t090_performance_worker import percentile, sha256, source_hashes
from t096_domain_probe import exact_positive_encoding

ROOT = Path(__file__).resolve().parents[1]
AU = 'AU-misc-patterns-e8m0-rceil-log2'
COMMIT = '8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b'
CASES = ('ordinary', 'fp16-guard', 'bf16-guard', 'cpu-guard', 'strided', 'exponent-sweep')
ORDINARY = [1.0, 2.0, 4.0, 3.0, 1.5, 0.5, 0.25]


def read_gate(path):
    gate = json.loads(path.read_text())
    for key, value in dict(acceptance_unit_id=AU, backend='triton_experimental', pytorch_commit=COMMIT,
                          worker_sha256=sha256(Path(__file__)), benchmark_allowed=True,
                          workload='community-ordinary-seven-fp32', correctness='passed').items():
        if gate.get(key) != value:
            raise ValueError(f'性能门禁 {key} 不符')
    for item in gate['evidence']:
        source = (ROOT/item['path']).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file() or sha256(source) != item['sha256']:
            raise ValueError('性能门禁证据不存在、逃逸或hash错误')
    for name, digest in gate['installed_files'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('性能门禁对应的产品文件已变化')
    return gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=CASES, default='ordinary')
    p.add_argument('--mode', choices=['off', 'on'], required=True)
    p.add_argument('--phase', choices=['functional', 'benchmark'], default='functional')
    p.add_argument('--artifact-dir', type=Path, required=True)
    p.add_argument('--gate', type=Path)
    args = p.parse_args()
    if Path.cwd() != Path('/home/z50063656/tmp'):
        p.error('必须从 tmp 启动')
    gate = None
    if args.phase == 'benchmark':
        if args.case != 'ordinary' or not args.gate:
            p.error('只允许对原七元素正确输入在签发门禁后计时')
        gate = read_gate(args.gate)
    out = args.artifact_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TORCHINDUCTOR_NPU_BACKEND='triton_experimental', TORCH_COMPILE_DEBUG='1',
        TORCH_COMPILE_DEBUG_DIR=str(out/'debug'), TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'),
        TRITON_CACHE_DIR=str(out/'triton-cache'), TORCHINDUCTOR_FORCE_DISABLE_CACHES='1',
        TORCHINDUCTOR_COMPILE_THREADS='1')
    import torch
    import torch_npu
    from torch_npu._inductor.triton_experimental import config as ncfg
    from torch_npu.utils._dynamo import register_inductor_npu, _InductorNpuRegistry
    from torch._inductor.pattern_matcher import ReplacementPatternEntry
    from torch._dynamo.utils import counters
    ncfg.enable_e8m0_rceil_log2 = args.mode == 'on'
    register_inductor_npu()
    assert _InductorNpuRegistry._loaded_backend == 'triton_experimental'
    assert torch.version.git_version == COMMIT
    torch.npu.set_device(0)
    dtype = {'fp16-guard': torch.float16, 'bf16-guard': torch.bfloat16}.get(args.case, torch.float32)
    device = 'cpu' if args.case == 'cpu-guard' else 'npu'
    values = ORDINARY
    if args.case == 'exponent-sweep':
        values = [struct.unpack('<f', struct.pack('<I', (e << 23) | m))[0]
                  for e in range(1, 255) for m in (0, 1, 0x3FFFFF, 0x7FFFFF)]
    if args.case == 'strided':
        inp = torch.tensor([v for value in values for v in (value, 42.)], dtype=dtype, device=device)[::2]
        assert inp.stride() == (2,)
    else:
        inp = torch.tensor(values, dtype=dtype, device=device)

    def fn(x):
        return (torch.clamp(torch.ceil(torch.log2(x)), -127, 127) + 127).to(torch.uint8)

    expected = torch.tensor([exact_positive_encoding(v) for v in values], dtype=torch.uint8, device=device)
    eager = fn(inp)
    original = ReplacementPatternEntry.apply
    state = {'calls': 0, 'changes': 0}

    @functools.wraps(original)
    def observed(entry, match, graph, node):
        if getattr(entry, 'pattern_name', None) != 'e8m0_rceil_log2_pattern':
            return original(entry, match, graph, node)
        index = state['calls']
        before = graph.python_code('self').src
        (out/f'target-{index}-before.txt').write_text(before)
        result = original(entry, match, graph, node)
        after = graph.python_code('self').src
        (out/f'target-{index}-after.txt').write_text(after)
        state['calls'] += 1
        state['changes'] += int(before != after)
        return result

    counters.clear()
    with mock.patch.object(ReplacementPatternEntry, 'apply', observed):
        start = time.perf_counter()
        compiled = torch.compile(fn, backend='inductor', fullgraph=True)
        actual = compiled(inp)
        torch.npu.synchronize()
        compile_ms = (time.perf_counter() - start) * 1000
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_hit = args.mode == 'on' and dtype == torch.float32 and device == 'npu'
    assert (state['changes'] > 0) if expected_hit else state['calls'] == 0, state
    assert sum(counters['graph_break'].values()) == 0
    samples = None
    memory = None
    if args.phase == 'benchmark':
        assert torch.equal(eager, expected), '错误 eager/OFF 不能用于计时'
        for _ in range(10):
            compiled(inp)
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        samples = {'host_ms': [], 'event_ms': []}
        for _ in range(100):
            torch.npu.synchronize()
            start_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)
            before = time.perf_counter()
            start_event.record()
            compiled(inp)
            end_event.record()
            torch.npu.synchronize()
            samples['host_ms'].append((time.perf_counter() - before)*1000)
            samples['event_ms'].append(start_event.elapsed_time(end_event))
        memory = dict(allocated=torch.npu.max_memory_allocated(), reserved=torch.npu.max_memory_reserved())
    sources = source_hashes()
    sources[str(Path(__file__).with_name('t096_domain_probe.py'))] = sha256(Path(__file__).with_name('t096_domain_probe.py'))
    record = dict(generated_at=datetime.now().astimezone().isoformat(), task_id='T-096', acceptance_unit_id=AU,
        phase=args.phase, case=args.case, mode=args.mode, status='passed', correctness='passed',
        backend=_InductorNpuRegistry._loaded_backend, backend_selected_before_import=True, pytorch_commit=COMMIT,
        pytorch_worktree_status=subprocess.check_output(['git', '-C', '/home/z50063656/Pass/src/pytorch', 'status', '--porcelain'], text=True),
        pid=os.getpid(), physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), device_name=torch.npu.get_device_name(0),
        input_spec=dict(shape=list(inp.shape), stride=list(inp.stride()), dtype=str(dtype), device=str(inp.device)),
        numerical_execution=device == 'npu', eager_equal=bool(torch.equal(actual, eager)), mathematical_equal=True,
        state=state, graph_breaks=0, target_rewrite='confirmed' if expected_hit else 'guard-preserved',
        compile_ms=compile_ms, samples=samples, memory=memory,
        timing={k: dict(p50=percentile(v,.5), p99=percentile(v,.99)) for k,v in samples.items()} if samples else None,
        worker_sha256=sha256(Path(__file__)), loaded_source_sha256=sources,
        gate_sha256=sha256(args.gate) if gate else None, product_candidate=None,
        product_gate_bypassed=False, fallback_review='pending-generated-code-review')
    (out/'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(f't096_installed_worker=passed case={args.case} mode={args.mode} phase={args.phase}', flush=True)


if __name__ == '__main__':
    main()
