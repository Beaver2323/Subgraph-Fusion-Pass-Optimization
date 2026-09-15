# AOT ID: ['10_forward']
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


# kernel path: /home/z50063656/tmp/t102-training-performance-20260915/functional-20260915T194132+0800/pattern-2/on/inductor-cache/tmp0oe_bqhh/7u/c7unobvuvkywfhnf6g43rdxjjgrltxjaryj2wbs3543ibpu7e5zx.py
# Topologically Sorted Source Nodes: [matmul, ], Original ATen: [aten.matmul, aten.mul, aten._to_copy, aten.isfinite, aten.all]
# Source node to ATen node mapping:
#    => abs_default, convert_element_type_default, eq_tensor, logical_not_default, mul_tensor, mul_tensor_4, ne_scalar
#   matmul => view_2
# Graph fragment:
#   %bmm : Tensor "f16[8, 8, 8][64, 8, 1]npu:0" = PlaceHolder[target=bmm]
#   %primals_3 : Tensor "f16[][]npu:0" = PlaceHolder[target=primals_3]
#   %view_2 : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2, 4, 8, 8]), kwargs = {})
#   %mul_tensor : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %primals_3), kwargs = {})
#   %convert_element_type_default : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.float32), kwargs = {})
#   %eq_tensor : Tensor "b8[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%convert_element_type_default, %convert_element_type_default), kwargs = {})
#   %abs_default : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convert_element_type_default,), kwargs = {})
#   %ne_scalar : Tensor "b8[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_default, inf), kwargs = {})
#   %mul_tensor_4 : Tensor "b8[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_tensor, %ne_scalar), kwargs = {})
#   %logical_not_default : Tensor "b8[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%mul_tensor_4,), kwargs = {})
#   return %logical_not_default
triton_unk_fused__to_copy_all_isfinite_matmul_mul_0 = async_compile.triton('triton_unk_fused__to_copy_all_isfinite_matmul_mul_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math

from triton.language.extra.cann.extension import extract_slice

@npu_triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [512, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 512, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__to_copy_all_isfinite_matmul_mul_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__to_copy_all_isfinite_matmul_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 512
    x_g_tile0 : tl.constexpr = (512) if (512) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 512
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x0index = x0offset + tl.arange(0, real_block_x0)[:]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        tmp0 = tl.load(in_ptr0 + (x0), x0mask).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (0)).to(tl.float32)
        tmp2 = tmp1
        tmp3 = tmp0 * tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4 == tmp4
        tmp6 = tl_math.abs(tmp4)
        tmp7 = tl.full([1], float("inf"), tl.float32)
        tmp8 = tmp6 != tmp7
        tmp9 = tmp5 & tmp8
        tmp10 = tmp9 == 0
        tl.store(out_ptr0 + (x0), tmp10, x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t102-training-performance-20260915/functional-20260915T194132+0800/pattern-2/on/inductor-cache/tmp0oe_bqhh/mr/cmrgy5strl5xzavj2kqqu2scvkjyeraytaailpw4hyd3zbht3eld.py
# Topologically Sorted Source Nodes: [matmul, , softmax], Original ATen: [aten.matmul, aten.mul, aten._to_copy, aten.scalar_tensor, aten.ge, aten.neg, aten.where, aten.amax, aten.sub, aten.all, aten._softmax]
# Source node to ATen node mapping:
#    => amax_default, amax_default_1, convert_element_type_default, convert_element_type_default_1, ge_scalar, logical_not_default_1, mul_tensor, mul_tensor_1, mul_tensor_2, mul_tensor_3, neg_default, scalar_tensor_default, sub_tensor, sub_tensor_1, where_self, where_self_1
#   matmul => view_2
#   softmax => convert_element_type_3, div, exp, sum_1
# Graph fragment:
#   %bmm : Tensor "f16[8, 8, 8][64, 8, 1]npu:0" = PlaceHolder[target=bmm]
#   %primals_3 : Tensor "f16[][]npu:0" = PlaceHolder[target=primals_3]
#   %any_dims : Tensor "b8[2, 4, 8, 1][32, 8, 1, 1]npu:0" = PlaceHolder[target=any_dims]
#   %amax_default : Tensor "f32[2, 4, 8, 1][32, 8, 1, 64]npu:0" = PlaceHolder[target=amax_default]
#   %amax_default_1 : Tensor "f32[2, 4, 8, 1][32, 8, 1, 64]npu:0" = PlaceHolder[target=amax_default_1]
#   %exp : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0" = PlaceHolder[target=exp]
#   %sum_1 : Tensor "f32[2, 4, 8, 1][32, 8, 1, 64]npu:0" = PlaceHolder[target=sum_1]
#   %view_2 : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2, 4, 8, 8]), kwargs = {})
#   %mul_tensor : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %primals_3), kwargs = {})
#   %convert_element_type_default : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.float32), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]npu:0"[num_users=2] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (1,), kwargs = {dtype: torch.float32, device: npu:0, pin_memory: False})
#   %ge_scalar : Tensor "b8[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%primals_3, 0), kwargs = {})
#   %neg_default : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%scalar_tensor_default,), kwargs = {})
#   %where_self : Tensor "f32[][]npu:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%ge_scalar, %scalar_tensor_default, %neg_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, %where_self), kwargs = {})
#   %amax_default : Tensor "f32[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_1, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %amax_default), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self, %primals_3), kwargs = {})
#   %mul_tensor_3 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor, %mul_tensor_2), kwargs = {})
#   %amax_default_1 : Tensor "f32[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_default, [-1], True), kwargs = {})
#   %sub_tensor_1 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_default, %amax_default_1), kwargs = {})
#   %logical_not_default_1 : Tensor "b8[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dims,), kwargs = {})
#   %where_self_1 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_1, %mul_tensor_3, %sub_tensor_1), kwargs = {})
#   %exp : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%where_self_1,), kwargs = {})
#   %sum_1 : Tensor "f32[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %convert_element_type_3 : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
#   return %amax_default,%amax_default_1,%exp,%sum_1,%expand_2
triton_unk_fused__softmax__to_copy_all_amax_ge_matmul_mul_neg_scalar_tensor_sub_where_1 = async_compile.triton('triton_unk_fused__softmax__to_copy_all_amax_ge_matmul_mul_neg_scalar_tensor_sub_where_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*i1', 'out_ptr2': '*fp32', 'out_ptr4': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'r0_numel': 8}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [64, {'divisors': [1]}], 'R0_BLOCK_HINT': [8, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 8, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax__to_copy_all_amax_ge_matmul_mul_neg_scalar_tensor_sub_where_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 3, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused__softmax__to_copy_all_amax_ge_matmul_mul_neg_scalar_tensor_sub_where_1(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 64
    r0_numel = 8
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 64
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
        tmp2 = tl.load(in_ptr1 + (0)).to(tl.float32)
        tmp3 = tmp2
        _tmp11 = tl.full([real_block_x0, R0_BLOCK], float("-inf"), tl.float32)
        _tmp16 = tl.full([real_block_x0, R0_BLOCK], float("-inf"), tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp4 = tl.full([1, 1], 0.0, tl.float32)
            tmp5 = tmp3 >= tmp4
            tmp6 = tl.full([1, 1], 1.0, tl.float32)
            tmp7 = tl.full([1, 1], -1.0, tl.float32)
            tmp8 = tl.where(tmp5, tmp6, tmp7)
            tmp9 = tmp1 * tmp8
            tmp10 = tl.broadcast_to(tmp9, [real_block_x0, R0_BLOCK])
            tmp12 = tl.maximum(_tmp11, tmp10)
            _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
            tmp13 = tmp0 * tmp3
            tmp14 = tmp13.to(tl.float32)
            tmp15 = tl.broadcast_to(tmp14, [real_block_x0, R0_BLOCK])
            tmp17 = tl.maximum(_tmp16, tmp15)
            _tmp16 = tl.where(r0_mask & xmask, tmp17, _tmp16)
        tmp11 = triton_helpers.max2(_tmp11, 1)[:, None]
        tmp16 = triton_helpers.max2(_tmp16, 1)[:, None]
        tmp18 = tl.load(in_ptr2 + x0, x0mask, eviction_policy='evict_last') != 0
        tmp22 = tl.load(in_ptr1 + (0)).to(tl.float32)
        tmp23 = tl.broadcast_to(tmp22, [1])
        _tmp40 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp20 = tl.load(in_ptr0 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp19 = tmp18 == 0
            tmp21 = tmp20.to(tl.float32)
            tmp24 = tl.full([1, 1], 0.0, tl.float32)
            tmp25 = tmp23 >= tmp24
            tmp26 = tl.full([1, 1], 1.0, tl.float32)
            tmp27 = tl.full([1, 1], -1.0, tl.float32)
            tmp28 = tl.where(tmp25, tmp26, tmp27)
            tmp29 = tmp21 * tmp28
            tmp30 = tmp29 - tmp11
            tmp31 = tmp23.to(tl.float32)
            tmp32 = tmp28 * tmp31
            tmp33 = tmp30 * tmp32
            tmp34 = tmp20 * tmp23
            tmp35 = tmp34.to(tl.float32)
            tmp36 = tmp35 - tmp16
            tmp37 = tl.where(tmp19, tmp33, tmp36)
            tmp38 = libdevice.exp(tmp37)
            tmp39 = tl.broadcast_to(tmp38, [real_block_x0, R0_BLOCK])
            tmp41 = _tmp40 + tmp39
            _tmp40 = tl.where(r0_mask & xmask, tmp41, _tmp40)
            tl.store(out_ptr2 + (r0_1 + 8*x0), tmp38, r0_mask & x0mask)
        tmp40 = tl.sum(_tmp40, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp42 = tl.load(out_ptr2 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp43 = (tmp42 / tmp40)
            tmp44 = tmp43.to(tl.float32)
            tmp45 = tmp44.to(tl.float16)
            tmp46 = tmp45.to(tl.float32)
            tl.store(out_ptr4 + (r0_1 + 8*x0), tmp46, r0_mask & x0mask)
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
        primals_1, primals_2, primals_3, primals_4 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            primals_2 = copy_if_misaligned(primals_2)
            primals_1 = copy_if_misaligned(primals_1)
            buf0 = empty_strided_npu((8, 8, 8), (64, 8, 1), torch.float16)
            # Topologically Sorted Source Nodes: [transpose, matmul], Original ATen: [aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:1
            extern_kernels.bmm(reinterpret_tensor(primals_2, (8, 8, 16), (128, 16, 1), 0), reinterpret_tensor(primals_1, (8, 16, 8), (128, 1, 16), 0), out=buf0)
            primals_3 = copy_if_misaligned(primals_3)
            buf3 = empty_strided_npu((2, 4, 8, 8), (256, 64, 8, 1), torch.bool)
            # Topologically Sorted Source Nodes: [matmul, ], Original ATen: [aten.matmul, aten.mul, aten._to_copy, aten.isfinite, aten.all]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_all_isfinite_matmul_mul_0.run(buf0, primals_3, buf3, 512, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [matmul, ], Original ATen: [aten.matmul, aten.mul, aten._to_copy, aten.isfinite, aten.all]
            # [Provenance debug handles] torch.ops.aten.any.dims:2
            buf4 = torch.ops.aten.any.dims(buf3, [-1], True)
            assert_alignment(buf4, 16, 'torch.ops.aten.any.dims')
            del buf3
            buf5 = empty_strided_npu((2, 4, 8, 8), (256, 64, 8, 1), torch.float32)
            buf7 = empty_strided_npu((2, 4, 8, 8), (256, 64, 8, 1), torch.float16)
            # Topologically Sorted Source Nodes: [matmul, , softmax], Original ATen: [aten.matmul, aten.mul, aten._to_copy, aten.scalar_tensor, aten.ge, aten.neg, aten.where, aten.amax, aten.sub, aten.all, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax__to_copy_all_amax_ge_matmul_mul_neg_scalar_tensor_sub_where_1.run(buf0, primals_3, buf4, buf5, buf7, 64, 8, stream=raw_stream0)
            del buf0
            del buf4
            del buf5
            primals_4 = copy_if_misaligned(primals_4)
            buf8 = empty_strided_npu((8, 8, 16), (128, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:3
            extern_kernels.bmm(reinterpret_tensor(buf7, (8, 8, 8), (64, 8, 1), 0), reinterpret_tensor(primals_4, (8, 8, 16), (128, 16, 1), 0), out=buf8)
        return (reinterpret_tensor(buf8, (2, 4, 8, 16), (512, 128, 16, 1), 0), primals_2, primals_3, primals_4, reinterpret_tensor(primals_1, (2, 4, 16, 8), (512, 128, 1, 16), 0), buf7, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float16)
    primals_2 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float16)
    primals_3 = rand_strided((), (), device='npu:0', dtype=torch.float16)
    primals_4 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float16)
    return [primals_1, primals_2, primals_3, primals_4]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
