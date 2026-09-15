# AOT ID: ['0_inference']
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


# kernel path: /home/z50063656/tmp/attention-performance-20260914/benchmark-20260914T230901+0800/pattern-13/off3/inductor-cache/tmpm9r0s301/w3/cw3ogxy6jvqcbzkfo7vxtbtgsgybbg6zx436dspt7mwghsse77gb.py
# Topologically Sorted Source Nodes: [attn_weight], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_weight => amax, convert_element_type_2, convert_element_type_3, div, exp, sub, sum_1
# Graph fragment:
#   %bmm : Tensor "f16[1024, 128, 128][16384, 128, 1]npu:0" = PlaceHolder[target=bmm]
#   %amax : Tensor "f32[1024, 128, 1][128, 1, 131072]npu:0" = PlaceHolder[target=amax]
#   %sum_1 : Tensor "f32[1024, 128, 1][128, 1, 131072]npu:0" = PlaceHolder[target=sum_1]
#   %convert_element_type_2 : Tensor "f32[1024, 128, 128][16384, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm, torch.float32), kwargs = {})
#   %amax : Tensor "f32[1024, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_2, [-1], True), kwargs = {})
#   %sub : Tensor "f32[1024, 128, 128][16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %amax), kwargs = {})
#   %exp : Tensor "f32[1024, 128, 128][16384, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : Tensor "f32[1024, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : Tensor "f32[1024, 128, 128][16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %convert_element_type_3 : Tensor "f16[1024, 128, 128][16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
#   return %amax,%sum_1,%convert_element_type_3
triton_unk_fused__softmax_0 = async_compile.triton('triton_unk_fused__softmax_0', '''
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
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'r0_numel': 128}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [131072, {'divisors': [1]}], 'R0_BLOCK_HINT': [128, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 131072, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 128, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused__softmax_0(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 131072
    r0_numel = 128
    x_g_tile0 : tl.constexpr = (131072) if (131072) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 131072
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
        _tmp3 = tl.full([real_block_x0, R0_BLOCK], float("-inf"), tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), r0_mask & x0mask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tl.broadcast_to(tmp1, [real_block_x0, R0_BLOCK])
            tmp4 = tl.maximum(_tmp3, tmp2)
            _tmp3 = tmp4
        tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
        _tmp10 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp5 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), r0_mask & x0mask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
            tmp6 = tmp5.to(tl.float32)
            tmp7 = tmp6 - tmp3
            tmp8 = libdevice.exp(tmp7)
            tmp9 = tl.broadcast_to(tmp8, [real_block_x0, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tmp11
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp12 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp13 = tmp12.to(tl.float32)
            tmp14 = tmp13 - tmp3
            tmp15 = libdevice.exp(tmp14)
            tmp16 = (tmp15 / tmp10)
            tmp17 = tmp16.to(tl.float32)
            tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp17, r0_mask & x0mask)
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
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg1_1 = copy_if_misaligned(arg1_1)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_npu((1024, 128, 128), (16384, 128, 1), torch.float16)
            # Topologically Sorted Source Nodes: [transpose, bmm], Original ATen: [aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:1
            extern_kernels.bmm(arg1_1, reinterpret_tensor(arg0_1, (1024, 128, 128), (16384, 1, 128), 0), out=buf0)
            del arg0_1
            del arg1_1
            buf3 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [attn_weight], Original ATen: [aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_0.run(buf3, 131072, 128, stream=raw_stream0)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf4 = empty_strided_npu((1024, 128, 128), (16384, 128, 1), torch.float16)
            # Topologically Sorted Source Nodes: [attn_weight, bmm_1], Original ATen: [aten._softmax, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:2
            extern_kernels.bmm(buf3, arg2_1, out=buf4)
            del arg2_1
            del buf3
        return (buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1024, 128, 128), (16384, 128, 1), device='npu:0', dtype=torch.float16)
    arg1_1 = rand_strided((1024, 128, 128), (16384, 128, 1), device='npu:0', dtype=torch.float16)
    arg2_1 = rand_strided((1024, 128, 128), (16384, 128, 1), device='npu:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
