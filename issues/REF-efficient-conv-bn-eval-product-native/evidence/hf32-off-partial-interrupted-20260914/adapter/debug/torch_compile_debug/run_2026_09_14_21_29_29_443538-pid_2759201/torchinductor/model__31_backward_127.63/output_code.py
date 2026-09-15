# AOT ID: ['31_backward']
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


# kernel path: /home/z50063656/tmp/t098-native-doq0hsal/adapter/inductor-cache/tmphhii09kp/hk/chkvwwfr3376uxkmmrvyaapqpri4k3tnmhticoy2hyo7nwebz2k6.py
# Topologically Sorted Source Nodes: [sum_1, view_3, efficient_conv_bn_eval, mul_3, view_4, mul_4, sum_2, add_2, mul_6], Original ATen: [aten.sum, aten.view, aten.zeros_like, aten.sub, aten.mul, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#   add_2 => add_2
#   efficient_conv_bn_eval => add, full_default, rsqrt, sub, view
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_6 => mul_6
#   sum_1 => sum_1
#   sum_2 => sum_2
#   view_3 => view_3
#   view_4 => view_4
# Graph fragment:
#   %tangents_1 : Tensor "f32[4, 32][32, 1]npu:0" = PlaceHolder[target=tangents_1]
#   %sum_1 : Tensor "f32[1, 32][32, 1]npu:0" = PlaceHolder[target=sum_1]
#   %primals_2 : Tensor "f32[32][1]npu:0" = PlaceHolder[target=primals_2]
#   %mm : Tensor "f32[32, 3][3, 1]npu:0" = PlaceHolder[target=mm]
#   %primals_4 : Tensor "f32[32, 3][3, 1]npu:0" = PlaceHolder[target=primals_4]
#   %primals_3 : Tensor "f32[32][1]npu:0" = PlaceHolder[target=primals_3]
#   %sum_1 : Tensor "f32[1, 32][32, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0], True), kwargs = {})
#   %view_3 : Tensor "f32[32][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_1, [32]), kwargs = {})
#   %full_default : Tensor "f32[32][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %sub : Tensor "f32[32][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %primals_2), kwargs = {})
#   %mul_3 : Tensor "f32[32][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %sub), kwargs = {})
#   %view_4 : Tensor "f32[32, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_3, [32, 1]), kwargs = {})
#   %mul_4 : Tensor "f32[32, 3][3, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, %primals_4), kwargs = {})
#   %add : Tensor "f32[32][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[32][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %view : Tensor "f32[32, 1][1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%rsqrt, [-1, 1]), kwargs = {})
#   %sum_2 : Tensor "f32[32, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [1], True), kwargs = {})
#   %add_2 : Tensor "f32[32, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %sum_2), kwargs = {})
#   %mul_6 : Tensor "f32[32, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %view), kwargs = {})
#   return %sum_1,%mul_6
triton_unk_fused_add_mul_rsqrt_sub_sum_view_zeros_like_0 = async_compile.triton('triton_unk_fused_add_mul_rsqrt_sub_sum_view_zeros_like_0', '''
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
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [16, 2, {'divisors': [1, 16]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x1', 'length': 16, 'divisor': 1, 'seed': 1}, {'name': 'x2', 'length': 2, 'divisor': 16, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_mul_rsqrt_sub_sum_view_zeros_like_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_mul_rsqrt_sub_sum_view_zeros_like_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 32
    x_g_tile0 : tl.constexpr = (16) if (16) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (2) if (2) < (x_g_rem1) else (x_g_rem1)
    x1numel : tl.constexpr = 16
    x2numel : tl.constexpr = 2
    real_block_x1 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x2 : tl.constexpr = x_g_tile1
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x_cumblk_1 = x1_blocks
    total_blocks = x1_blocks * x2_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x1offset = (group_base + i) % x1_blocks * real_block_x1
        x2offset = (group_base + i) // x_cumblk_1 % x2_blocks * real_block_x2
        x1index = x1offset + tl.arange(0, real_block_x1)[None, :]
        x1 = x1index
        x1mask = x1index < x1numel
        x2index = x2offset + tl.arange(0, real_block_x2)[:, None]
        x2 = x2index
        x2mask = x2index < x2numel
        xmask = x1mask & x2mask
        x0 = x1 + 16*x2
        x0mask = x1mask & x2mask
        tmp0 = tl.load(in_ptr0 + (x0), x0mask)
        tmp1 = tl.load(in_ptr0 + (32 + x0), x0mask)
        tmp3 = tl.load(in_ptr0 + (64 + x0), x0mask)
        tmp5 = tl.load(in_ptr0 + (96 + x0), x0mask)
        tmp7 = tl.load(in_ptr1 + (x0), x0mask)
        tmp11 = tl.load(in_ptr2 + (3*x1 + 48*x2), x1mask & x2mask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr3 + (3*x1 + 48*x2), x1mask & x2mask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr2 + (1 + 3*x1 + 48*x2), x1mask & x2mask, eviction_policy='evict_last')
        tmp15 = tl.load(in_ptr3 + (1 + 3*x1 + 48*x2), x1mask & x2mask, eviction_policy='evict_last')
        tmp18 = tl.load(in_ptr2 + (2 + 3*x1 + 48*x2), x1mask & x2mask, eviction_policy='evict_last')
        tmp19 = tl.load(in_ptr3 + (2 + 3*x1 + 48*x2), x1mask & x2mask, eviction_policy='evict_last')
        tmp23 = tl.load(in_ptr4 + (x1 + 16*x2), x1mask & x2mask)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tl.full([1], 0.0, tl.float32)
        tmp9 = tmp8 - tmp7
        tmp10 = tmp6 * tmp9
        tmp13 = tmp11 * tmp12
        tmp16 = tmp14 * tmp15
        tmp17 = tmp13 + tmp16
        tmp20 = tmp18 * tmp19
        tmp21 = tmp17 + tmp20
        tmp22 = tmp10 + tmp21
        tmp24 = tl.full([1], 1e-05, tl.float32)
        tmp25 = tmp23 + tmp24
        tmp26 = tl.rsqrt(tmp25)
        tmp27 = tmp22 * tmp26
        tl.store(out_ptr0 + (x0), tmp6, x0mask)
        tl.store(out_ptr1 + (x1 + 16*x2), tmp27, x1mask & x2mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t098-native-doq0hsal/adapter/inductor-cache/tmphhii09kp/id/cidax6xznyq23ididi3623dsesyysfsigfumotzemkvreq4nbggl.py
# Topologically Sorted Source Nodes: [efficient_conv_bn_eval, mul_5], Original ATen: [aten.add, aten.rsqrt, aten.view, aten.mul]
# Source node to ATen node mapping:
#   efficient_conv_bn_eval => add, mul, rsqrt, view, view_1
#   mul_5 => mul_5
# Graph fragment:
#   %primals_5 : Tensor "f32[32][1]npu:0" = PlaceHolder[target=primals_5]
#   %primals_3 : Tensor "f32[32][1]npu:0" = PlaceHolder[target=primals_3]
#   %add : Tensor "f32[32][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[32][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %view : Tensor "f32[32, 1][1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%rsqrt, [-1, 1]), kwargs = {})
#   %view_1 : Tensor "f32[32, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_5, [32, 1]), kwargs = {})
#   %mul : Tensor "f32[32, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %view), kwargs = {})
#   %mul_5 : Tensor "f32[32, 3][3, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, %mul), kwargs = {})
#   return %buf2
triton_unk_fused_add_mul_rsqrt_view_1 = async_compile.triton('triton_unk_fused_add_mul_rsqrt_view_1', '''
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
    size_hints={'x': 96}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [3, 32, {'divisors': [1, 3]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 3, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 32, 'divisor': 3, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_mul_rsqrt_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_mul_rsqrt_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 96
    x_g_tile0 : tl.constexpr = (3) if (3) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (32) if (32) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 3
    x1numel : tl.constexpr = 32
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
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x1offset = (group_base + i) // x_cumblk_1 % x1_blocks * real_block_x1
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[:, None]
        x1 = x1index
        x1mask = x1index < x1numel
        xmask = x0mask & x1mask
        tmp0 = tl.load(in_ptr0 + (x1), x1mask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x1), x1mask, eviction_policy='evict_last')
        tmp2 = tl.full([1], 1e-05, tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tl.rsqrt(tmp3)
        tmp5 = tmp0 * tmp4
        tl.store(out_ptr0 + (x0 + 3*x1), tmp5, x0mask & x1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t098-native-doq0hsal/adapter/inductor-cache/tmphhii09kp/v4/cv4p2267wc27ymwahgyxqbntuqvtxanxwqazjrvxy4lhigkljh45.py
# Topologically Sorted Source Nodes: [efficient_conv_bn_eval, mul_5], Original ATen: [aten.add, aten.rsqrt, aten.view, aten.mul]
# Source node to ATen node mapping:
#   efficient_conv_bn_eval => add, mul, rsqrt, view, view_1
#   mul_5 => mul_5
# Graph fragment:
#   %mm : Tensor "f32[32, 3][3, 1]npu:0" = PlaceHolder[target=mm]
#   %buf2 : Tensor "f32[32, 3][3, 1]npu:0" = PlaceHolder[target=buf2]
#   %add : Tensor "f32[32][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[32][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %view : Tensor "f32[32, 1][1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%rsqrt, [-1, 1]), kwargs = {})
#   %view_1 : Tensor "f32[32, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_5, [32, 1]), kwargs = {})
#   %mul : Tensor "f32[32, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %view), kwargs = {})
#   %mul_5 : Tensor "f32[32, 3][3, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, %mul), kwargs = {})
#   return %mul_5
triton_unk_fused_add_mul_rsqrt_view_2 = async_compile.triton('triton_unk_fused_add_mul_rsqrt_view_2', '''
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
    size_hints={'x': 96}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [96, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 96, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_mul_rsqrt_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_mul_rsqrt_view_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 96
    x_g_tile0 : tl.constexpr = (96) if (96) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 96
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
        tmp0 = tl.load(in_out_ptr0 + (x0), x0mask)
        tmp1 = tl.load(in_ptr0 + (x0), x0mask)
        tmp2 = tmp0 * tmp1
        tl.store(in_out_ptr0 + (x0), tmp2, x0mask)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, tangents_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            tangents_1 = copy_if_misaligned(tangents_1)
            buf0 = empty_strided_npu((32, 3), (3, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_1, permute_3], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:32
            extern_kernels.mm(reinterpret_tensor(tangents_1, (32, 4), (1, 32), 0), primals_1, out=buf0)
            del primals_1
            buf1 = empty_strided_npu((1, 32), (32, 1), torch.float32)
            buf4 = empty_strided_npu((32, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sum_1, view_3, efficient_conv_bn_eval, mul_3, view_4, mul_4, sum_2, add_2, mul_6], Original ATen: [aten.sum, aten.view, aten.zeros_like, aten.sub, aten.mul, aten.add, aten.rsqrt]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_rsqrt_sub_sum_view_zeros_like_0.run(tangents_1, primals_2, buf0, primals_4, primals_3, buf1, buf4, 32, stream=raw_stream0)
            del primals_2
            del primals_4
            del tangents_1
            buf2 = empty_strided_npu((32, 3), (3, 1), torch.float32)
            # Topologically Sorted Source Nodes: [efficient_conv_bn_eval, mul_5], Original ATen: [aten.add, aten.rsqrt, aten.view, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_rsqrt_view_1.run(primals_5, primals_3, buf2, 96, stream=raw_stream0)
            del primals_3
            del primals_5
            buf3 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [efficient_conv_bn_eval, mul_5], Original ATen: [aten.add, aten.rsqrt, aten.view, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_mul_rsqrt_view_2.run(buf3, buf2, 96, stream=raw_stream0)
            del buf2
        return (None, None, None, buf3, reinterpret_tensor(buf4, (32, ), (1, ), 0), reinterpret_tensor(buf1, (32, ), (1, ), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((4, 3), (3, 1), device='npu:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 3), (3, 1), device='npu:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='npu:0', dtype=torch.float32)
    tangents_1 = rand_strided((4, 32), (32, 1), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
