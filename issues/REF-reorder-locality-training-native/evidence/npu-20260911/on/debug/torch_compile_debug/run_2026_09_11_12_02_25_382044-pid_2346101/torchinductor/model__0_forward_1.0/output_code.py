# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental import get_current_raw_stream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
import torch_npu
empty_strided_npu = torch_npu._C._empty_strided_npu


# kernel path: /home/z50063656/tmp/t087-npu-functional-20260911/on/inductor-cache/tmp61wivn1h/pl/cpldt63pyzruik2hh3haanjw62lwper6hcnt7jwgcirdwjyfjwqr.py
# Topologically Sorted Source Nodes: [a, b, mul, sin, cos, mul_1, add, sum_1], Original ATen: [aten.tanh, aten.sigmoid, aten.mul, aten.sin, aten.cos, aten.add, aten.sum]
# Source node to ATen node mapping:
#   a => tanh
#   add => add
#   b => sigmoid
#   cos => cos
#   mul => mul
#   mul_1 => mul_1
#   sin => sin
#   sum_1 => sum_1
# Graph fragment:
#   %mm : Tensor "f32[8, 16][16, 1]npu:0" = PlaceHolder[target=mm]
#   %mm_1 : Tensor "f32[8, 16][16, 1]npu:0" = PlaceHolder[target=mm_1]
#   %tanh : Tensor "f32[8, 16][16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%mm,), kwargs = {})
#   %sigmoid : Tensor "f32[8, 16][16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mm_1,), kwargs = {})
#   %mul : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %sigmoid), kwargs = {})
#   %sin : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh,), kwargs = {})
#   %cos : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%sigmoid,), kwargs = {})
#   %mul_1 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, %cos), kwargs = {})
#   %add : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %sum_1 : Tensor "f32[8][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add, [1]), kwargs = {})
#   return %sum_1
triton_unk_fused_add_cos_mul_sigmoid_sin_sum_tanh_0 = async_compile.triton('triton_unk_fused_add_cos_mul_sigmoid_sin_sum_tanh_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.reduction(
    size_hints={'x': 8, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'r0_numel': 16}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [8, {'divisors': [1]}], 'R0_BLOCK_HINT': [16, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 8, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 16, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_cos_mul_sigmoid_sin_sum_tanh_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused_add_cos_mul_sigmoid_sin_sum_tanh_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 8
    r0_numel = 16
    x_g_tile0 : tl.constexpr = (8) if (8) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 8
    real_block_x0 : tl.constexpr = x_g_tile0
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        rbase = tl.arange(0, R0_BLOCK)[None, :]
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x0index = x0offset + tl.arange(0, real_block_x0)[:, None]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp10 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp2 = tl.load(in_ptr1 + (r0_1 + 16*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = libdevice.tanh(tmp0)
            tmp3 = tl.sigmoid(tmp2)
            tmp4 = tmp1 * tmp3
            tmp5 = tl_math.sin(tmp1)
            tmp6 = tl_math.cos(tmp3)
            tmp7 = tmp5 * tmp6
            tmp8 = tmp4 + tmp7
            tmp9 = tl.broadcast_to(tmp8, [real_block_x0, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tl.store(out_ptr0 + (x0), tmp10, x0mask)
''', device_str='npu')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            primals_2 = copy_if_misaligned(primals_2)
            buf0 = empty_strided_npu((8, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.matmul]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(primals_2, primals_1, out=buf0)
            buf1 = empty_strided_npu((8, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.matmul]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(primals_2, primals_3, out=buf1)
            buf2 = empty_strided_npu((8, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [a, b, mul, sin, cos, mul_1, add, sum_1], Original ATen: [aten.tanh, aten.sigmoid, aten.mul, aten.sin, aten.cos, aten.add, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_cos_mul_sigmoid_sin_sum_tanh_0.run(buf0, buf1, buf2, 8, 16, stream=raw_stream0)
        return (buf2, primals_1, primals_2, primals_3, buf0, buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((16, 16), (16, 1), device='npu:0', dtype=torch.float32)
    primals_2 = rand_strided((8, 16), (16, 1), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((16, 16), (16, 1), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
