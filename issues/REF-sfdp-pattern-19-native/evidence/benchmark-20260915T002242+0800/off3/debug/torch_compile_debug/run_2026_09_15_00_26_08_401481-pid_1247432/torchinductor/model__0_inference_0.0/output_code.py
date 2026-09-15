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


# kernel path: /home/z50063656/tmp/attention-performance-20260915/benchmark-20260915T002242+0800/pattern-19/off3/inductor-cache/tmppjd5ovin/e7/ce7ohqwfxiziti5a2t7wem7dbkikwcet2nlagozpe2b6igvbrwgy.py
# Topologically Sorted Source Nodes: [attn_weights, inv_scale, attn_weights_1, causal_mask_value, attn_weights_2, attn_weights_3, softmax], Original ATen: [aten.matmul, aten.full, aten.div, aten.where, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_weights => view_2
#   attn_weights_1 => div
#   attn_weights_2 => where
#   attn_weights_3 => add
#   causal_mask_value => full_default_1
#   inv_scale => full_default
#   softmax => amax, div_1, exp, sub, sum_1
# Graph fragment:
#   %arg2_1 : Tensor "b8[1, 1, 8, 8][64, 64, 8, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %bmm : Tensor "f32[8, 8, 8][64, 8, 1]npu:0" = PlaceHolder[target=bmm]
#   %arg3_1 : Tensor "f32[1, 1, 8, 8][64, 64, 8, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %amax : Tensor "f32[2, 4, 8, 1][32, 8, 1, 64]npu:0" = PlaceHolder[target=amax]
#   %sum_1 : Tensor "f32[2, 4, 8, 1][32, 8, 1, 64]npu:0" = PlaceHolder[target=sum_1]
#   %view_2 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2, 4, 8, 8]), kwargs = {})
#   %full_default : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.66666), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %div : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_2, %full_default), kwargs = {})
#   %full_default_1 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -3.4028234663852886e+38), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%arg2_1, %div, %full_default_1), kwargs = {})
#   %add : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %arg3_1), kwargs = {})
#   %amax : Tensor "f32[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add, [-1], True), kwargs = {})
#   %sub : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %amax), kwargs = {})
#   %exp : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : Tensor "f32[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   return %amax,%sum_1,%expand_2
triton_unk_fused__softmax_add_div_full_matmul_where_0 = async_compile.triton('triton_unk_fused__softmax_add_div_full_matmul_where_0', '''
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
    size_hints={'x': 64, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'r0_numel': 8}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [8, 8, {'divisors': [1, 8]}], 'R0_BLOCK_HINT': [8, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 8, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 8, 'divisor': 8, 'seed': 2}, {'name': 'r0_2', 'length': 8, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_add_div_full_matmul_where_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused__softmax_add_div_full_matmul_where_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 64
    r0_numel = 8
    x_g_tile0 : tl.constexpr = (8) if (8) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (8) if (8) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 8
    x1numel : tl.constexpr = 8
    real_block_x0 : tl.constexpr = x_g_tile0
    real_block_x1 : tl.constexpr = x_g_tile1
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x_cumblk_1 = x0_blocks
    total_blocks = x0_blocks * x1_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        rbase = tl.arange(0, R0_BLOCK)[None, None, :]
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x1offset = (group_base + i) // x_cumblk_1 % x1_blocks * real_block_x1
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :, None]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[:, None, None]
        x1 = x1index
        x1mask = x1index < x1numel
        xmask = x0mask & x1mask
        r0_base = tl.arange(0, R0_BLOCK)[None, None, :]
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp9 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], float("-inf"), tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2 + 8 * x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0) != 0
            tmp1 = tl.load(in_out_ptr0 + (r0_2 + 8*x0 + 64*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=0.0)
            tmp6 = tl.load(in_ptr1 + (r0_2 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=float("-inf"))
            tmp2 = tl.full([1, 1], 1.5000150001500014, tl.float32)
            tmp3 = tmp1 * tmp2
            tmp4 = tl.full([1, 1], -3.4028234663852886e+38, tl.float32)
            tmp5 = tl.where(tmp0, tmp3, tmp4)
            tmp7 = tmp5 + tmp6
            tmp8 = tl.broadcast_to(tmp7, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp10 = tl.maximum(_tmp9, tmp8)
            _tmp9 = tmp10
        tmp9 = triton_helpers.max2(_tmp9, 2)[:, :, None]
        _tmp22 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp11 = tl.load(in_ptr0 + (r0_2 + 8 * x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0) != 0
            tmp12 = tl.load(in_out_ptr0 + (r0_2 + 8*x0 + 64*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=0.0)
            tmp17 = tl.load(in_ptr1 + (r0_2 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=float("-inf"))
            tmp13 = tl.full([1, 1], 1.5000150001500014, tl.float32)
            tmp14 = tmp12 * tmp13
            tmp15 = tl.full([1, 1], -3.4028234663852886e+38, tl.float32)
            tmp16 = tl.where(tmp11, tmp14, tmp15)
            tmp18 = tmp16 + tmp17
            tmp19 = tmp18 - tmp9
            tmp20 = libdevice.exp(tmp19)
            tmp21 = tl.broadcast_to(tmp20, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp23 = _tmp22 + tmp21
            _tmp22 = tmp23
        tmp22 = tl.sum(_tmp22, 2)[:, :, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp24 = tl.load(in_ptr0 + (r0_2 + 8 * x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0) != 0
            tmp25 = tl.load(in_out_ptr0 + (r0_2 + 8*x0 + 64*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_first', other=0.0)
            tmp30 = tl.load(in_ptr1 + (r0_2 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp26 = tl.full([1, 1], 1.5000150001500014, tl.float32)
            tmp27 = tmp25 * tmp26
            tmp28 = tl.full([1, 1], -3.4028234663852886e+38, tl.float32)
            tmp29 = tl.where(tmp24, tmp27, tmp28)
            tmp31 = tmp29 + tmp30
            tmp32 = tmp31 - tmp9
            tmp33 = libdevice.exp(tmp32)
            tmp34 = (tmp33 / tmp22)
            tl.store(in_out_ptr0 + (r0_2 + 8*x0 + 64*x1), tmp34, r0_mask & x0mask & x1mask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg1_1 = copy_if_misaligned(arg1_1)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_npu((8, 8, 8), (64, 8, 1), torch.float32)
            # Topologically Sorted Source Nodes: [attn_weights, permute], Original ATen: [aten.matmul, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:1
            extern_kernels.bmm(reinterpret_tensor(arg1_1, (8, 8, 16), (128, 16, 1), 0), reinterpret_tensor(arg0_1, (8, 16, 8), (128, 1, 16), 0), out=buf0)
            del arg0_1
            del arg1_1
            arg2_1 = copy_if_misaligned(arg2_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            buf3 = reinterpret_tensor(buf0, (2, 4, 8, 8), (256, 64, 8, 1), 0); del buf0  # reuse
            # Topologically Sorted Source Nodes: [attn_weights, inv_scale, attn_weights_1, causal_mask_value, attn_weights_2, attn_weights_3, softmax], Original ATen: [aten.matmul, aten.full, aten.div, aten.where, aten.add, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_add_div_full_matmul_where_0.run(buf3, arg2_1, arg3_1, 64, 8, stream=raw_stream0)
            del arg2_1
            del arg3_1
            arg4_1 = copy_if_misaligned(arg4_1)
            buf4 = empty_strided_npu((8, 8, 16), (128, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [attn_weights, inv_scale, attn_weights_1, causal_mask_value, attn_weights_2, attn_weights_3, softmax, matmul_1], Original ATen: [aten.matmul, aten.full, aten.div, aten.where, aten.add, aten._softmax]
            # [Provenance debug handles] extern_kernels.bmm:2
            extern_kernels.bmm(reinterpret_tensor(buf3, (8, 8, 8), (64, 8, 1), 0), reinterpret_tensor(arg4_1, (8, 8, 16), (128, 16, 1), 0), out=buf4)
            del arg4_1
            del buf3
        return (reinterpret_tensor(buf4, (2, 4, 8, 16), (512, 128, 16, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 1, 8, 8), (64, 64, 8, 1), device='npu:0', dtype=torch.bool)
    arg3_1 = rand_strided((1, 1, 8, 8), (64, 64, 8, 1), device='npu:0', dtype=torch.float32)
    arg4_1 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
