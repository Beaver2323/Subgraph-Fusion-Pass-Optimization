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


# kernel path: /home/z50063656/tmp/attention-functional-20260914-revised-domains/functional-20260915T000459+0800/pattern-19/on/inductor-cache/tmp91byb43h/wa/cwahuntji74bqbitl2kbmqjmuqiwfbn5tpl62cfxo3tp5thhxodd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#    => mul_scalar
# Graph fragment:
#   %arg1_1 : Tensor "f32[2, 4, 8, 16][512, 128, 16, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %mul_scalar : Tensor "f32[2, 4, 8, 16][512, 128, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%arg1_1, 1.2247509951618742), kwargs = {})
#   return %expand_default
triton_unk_fused_mul_0 = async_compile.triton('triton_unk_fused_mul_0', '''
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
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 1024, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 1024
    x_g_tile0 : tl.constexpr = (1024) if (1024) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 1024
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
        tmp0 = tl.load(in_ptr0 + (x0), x0mask)
        tmp1 = tl.full([1], 1.2247509951618742, tl.float32)
        tmp2 = tmp0 * tmp1
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/attention-functional-20260914-revised-domains/functional-20260915T000459+0800/pattern-19/on/inductor-cache/tmp91byb43h/cd/ccd2fjpnwd2ipuj7itbqxqyjaltlvgmd7pfajewbbxek6k4j4i6r.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.mul]
# Source node to ATen node mapping:
#    => mul_scalar_1, permute_default
# Graph fragment:
#   %arg0_1 : Tensor "f32[2, 4, 8, 16][512, 128, 16, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %permute_default : Tensor "f32[2, 4, 16, 8][512, 128, 1, 16]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg0_1, [0, 1, 3, 2]), kwargs = {})
#   %mul_scalar_1 : Tensor "f32[2, 4, 16, 8][512, 128, 1, 16]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_default, 1.2247509951618742), kwargs = {})
#   return %expand_default_1
triton_unk_fused_mul_transpose_1 = async_compile.triton('triton_unk_fused_mul_transpose_1', '''
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
    size_hints={'y': 128, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'YBLOCK_HINT': [16, 8, {'divisors': [1, 16]}], 'XBLOCK_HINT': [8, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 16, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 8, 'divisor': 16, 'seed': 2}, {'name': 'x2', 'length': 8, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_mul_transpose_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_mul_transpose_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    ynumel = 128
    xnumel = 8
    y_g_tile0 : tl.constexpr = (16) if (16) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (8) if (8) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 16
    y1numel : tl.constexpr = 8
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y_cumblk_1 = y0_blocks
    x_g_tile0 : tl.constexpr = (8) if (8) < (XBLOCK) else (XBLOCK)
    x2numel : tl.constexpr = 8
    real_block_x2 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x_cumblk_0 = y0_blocks * y1_blocks
    total_blocks = y0_blocks * y1_blocks * x2_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x2offset = (group_base + i) // x_cumblk_0 % x2_blocks * real_block_x2
        x2index = x2offset + tl.arange(0, real_block_x2)[None, None, :]
        x2 = x2index
        x2mask = x2index < x2numel
        xmask = x2mask
        y0offset = (group_base + i) % y0_blocks * real_block_y0
        y1offset = (group_base + i) // y_cumblk_1 % y1_blocks * real_block_y1
        y0index = y0offset + tl.arange(0, real_block_y0)[None, :, None]
        y0 = y0index
        y0mask = y0index < y0numel
        y1index = y1offset + tl.arange(0, real_block_y1)[:, None, None]
        y1 = y1index
        y1mask = y1index < y1numel
        ymask = y0mask & y1mask
        tmp0 = tl.load(in_ptr0 + (y0 + 16*x2 + 128*y1), x2mask & y0mask & y1mask)
        tmp1 = tl.full([1, 1], 1.2247509951618742, tl.float32)
        tmp2 = tmp0 * tmp1
        tl.store(out_ptr0 + (x2 + 8*y0 + 128*y1), tmp2, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/attention-functional-20260914-revised-domains/functional-20260915T000459+0800/pattern-19/on/inductor-cache/tmp91byb43h/o5/co5mb7xqs6ggiavamp3ngesg7hjxfbvikqglhptylbnh5rfn2nf6.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.full, aten.where, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#    => add_tensor, eq_scalar, full_default_2, logical_not_default, view_default_2, where_self
# Graph fragment:
#   %bmm_default : Tensor "f32[8, 8, 8][64, 8, 1]npu:0" = PlaceHolder[target=bmm_default]
#   %arg2_1 : Tensor "b8[1, 1, 8, 8][64, 64, 8, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[1, 1, 8, 8][64, 64, 8, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %view_default_2 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default, [2, 4, 8, 8]), kwargs = {})
#   %full_default_2 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, device: npu:0, pin_memory: False})
#   %where_self : Tensor "f32[1, 1, 8, 8][64, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%arg2_1, %arg3_1, %full_default_2), kwargs = {})
#   %add_tensor : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %where_self), kwargs = {})
#   %eq_scalar : Tensor "b8[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_tensor, -inf), kwargs = {})
#   %logical_not_default : Tensor "b8[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_scalar,), kwargs = {})
#   return %logical_not_default
triton_unk_fused__safe_softmax_add_full_matmul_where_2 = async_compile.triton('triton_unk_fused__safe_softmax_add_full_matmul_where_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [64, 8, {'divisors': [1, 64]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 8, 'divisor': 64, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax_add_full_matmul_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__safe_softmax_add_full_matmul_where_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 512
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (8) if (8) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 64
    x1numel : tl.constexpr = 8
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), x0mask & x1mask)
        tmp1 = tl.load(in_ptr1 + x0, x0mask, eviction_policy='evict_last') != 0
        tmp2 = tl.load(in_ptr2 + (x0), x0mask, eviction_policy='evict_last')
        tmp3 = tl.full([1], float("-inf"), tl.float32)
        tmp4 = tl.where(tmp1, tmp2, tmp3)
        tmp5 = tmp0 + tmp4
        tmp6 = tmp5 == tmp3
        tmp7 = tmp6 == 0
        tl.store(out_ptr0 + (x0 + 64*x1), tmp7, x0mask & x1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/attention-functional-20260914-revised-domains/functional-20260915T000459+0800/pattern-19/on/inductor-cache/tmp91byb43h/3x/c3xc7pvik3xeaocgcfums2nudnpbslyngacpawg6kfyuvqneyq2y.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.full, aten.where, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#    => add_tensor, amax_default, div_tensor, exp_default, full_default_2, full_default_3, logical_not_default_1, sub_tensor, sum_dim_int_list, view_default_2, where_self, where_self_1
# Graph fragment:
#   %bmm_default : Tensor "f32[8, 8, 8][64, 8, 1]npu:0" = PlaceHolder[target=bmm_default]
#   %arg2_1 : Tensor "b8[1, 1, 8, 8][64, 64, 8, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[1, 1, 8, 8][64, 64, 8, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %amax_default : Tensor "f32[2, 4, 8, 1][32, 8, 1, 64]npu:0" = PlaceHolder[target=amax_default]
#   %any_dim : Tensor "b8[2, 4, 8, 1][32, 8, 1, 1]npu:0" = PlaceHolder[target=any_dim]
#   %sum_dim_int_list : Tensor "f32[2, 4, 8, 1][32, 8, 1, 64]npu:0" = PlaceHolder[target=sum_dim_int_list]
#   %view_default_2 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default, [2, 4, 8, 8]), kwargs = {})
#   %full_default_2 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, device: npu:0, pin_memory: False})
#   %where_self : Tensor "f32[1, 1, 8, 8][64, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%arg2_1, %arg3_1, %full_default_2), kwargs = {})
#   %add_tensor : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %where_self), kwargs = {})
#   %logical_not_default_1 : Tensor "b8[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dim,), kwargs = {})
#   %full_default_3 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 4, 8, 8], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %amax_default : Tensor "f32[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_tensor, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_tensor, %amax_default), kwargs = {})
#   %exp_default : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %sum_dim_int_list : Tensor "f32[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_default, [-1], True), kwargs = {})
#   %div_tensor : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %sum_dim_int_list), kwargs = {})
#   %where_self_1 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_1, %full_default_3, %div_tensor), kwargs = {})
#   return %amax_default,%sum_dim_int_list,%expand_default_2
triton_unk_fused__safe_softmax_add_full_matmul_where_3 = async_compile.triton('triton_unk_fused__safe_softmax_add_full_matmul_where_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'r0_numel': 8}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [8, 8, {'divisors': [1, 8]}], 'R0_BLOCK_HINT': [8, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 8, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 8, 'divisor': 8, 'seed': 2}, {'name': 'r0_2', 'length': 8, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax_add_full_matmul_where_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused__safe_softmax_add_full_matmul_where_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        _tmp7 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], float("-inf"), tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (r0_2 + 8*x0 + 64*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr0 + (r0_2 + 8 * x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0) != 0
            tmp2 = tl.load(in_ptr1 + (r0_2 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=float("-inf"))
            tmp3 = tl.full([1, 1], float("-inf"), tl.float32)
            tmp4 = tl.where(tmp1, tmp2, tmp3)
            tmp5 = tmp0 + tmp4
            tmp6 = tl.broadcast_to(tmp5, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp8 = tl.maximum(_tmp7, tmp6)
            _tmp7 = tmp8
        tmp7 = triton_helpers.max2(_tmp7, 2)[:, :, None]
        _tmp18 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp9 = tl.load(in_out_ptr0 + (r0_2 + 8*x0 + 64*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=0.0)
            tmp10 = tl.load(in_ptr0 + (r0_2 + 8 * x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0) != 0
            tmp11 = tl.load(in_ptr1 + (r0_2 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=float("-inf"))
            tmp12 = tl.full([1, 1], float("-inf"), tl.float32)
            tmp13 = tl.where(tmp10, tmp11, tmp12)
            tmp14 = tmp9 + tmp13
            tmp15 = tmp14 - tmp7
            tmp16 = libdevice.exp(tmp15)
            tmp17 = tl.broadcast_to(tmp16, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp19 = _tmp18 + tmp17
            _tmp18 = tmp19
        tmp18 = tl.sum(_tmp18, 2)[:, :, None]
        tmp20 = tl.load(in_ptr2 + (x0 + 8 * x1), x0mask & x1mask, eviction_policy='evict_last') != 0
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp22 = tl.load(in_out_ptr0 + (r0_2 + 8*x0 + 64*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_first', other=0.0)
            tmp23 = tl.load(in_ptr0 + (r0_2 + 8 * x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0) != 0
            tmp24 = tl.load(in_ptr1 + (r0_2 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp21 = tmp20 == 0
            tmp25 = tl.full([1, 1], float("-inf"), tl.float32)
            tmp26 = tl.where(tmp23, tmp24, tmp25)
            tmp27 = tmp22 + tmp26
            tmp28 = tmp27 - tmp7
            tmp29 = libdevice.exp(tmp28)
            tmp30 = (tmp29 / tmp18)
            tmp31 = tl.full([1, 1], 0.0, tl.float32)
            tmp32 = tl.where(tmp21, tmp31, tmp30)
            tl.store(in_out_ptr0 + (r0_2 + 8*x0 + 64*x1), tmp32, r0_mask & x0mask & x1mask)
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
            buf0 = empty_strided_npu((2, 4, 8, 16), (512, 128, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_mul_0.run(arg1_1, buf0, 1024, stream=raw_stream0)
            del arg1_1
            arg0_1 = copy_if_misaligned(arg0_1)
            buf1 = empty_strided_npu((2, 4, 16, 8), (512, 128, 8, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_mul_transpose_1.run(arg0_1, buf1, 128, 8, stream=raw_stream0)
            del arg0_1
            buf2 = empty_strided_npu((8, 8, 8), (64, 8, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:1
            extern_kernels.bmm(reinterpret_tensor(buf0, (8, 8, 16), (128, 16, 1), 0), reinterpret_tensor(buf1, (8, 16, 8), (128, 8, 1), 0), out=buf2)
            del buf0
            del buf1
            arg2_1 = copy_if_misaligned(arg2_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            buf3 = empty_strided_npu((2, 4, 8, 8), (256, 64, 8, 1), torch.bool)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.full, aten.where, aten.add, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax_add_full_matmul_where_2.run(buf2, arg2_1, arg3_1, buf3, 512, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.full, aten.where, aten.add, aten._safe_softmax]
            # [Provenance debug handles] torch.ops.aten.any.dim:2
            buf4 = torch.ops.aten.any.dim(buf3, -1, True)
            assert_alignment(buf4, 16, 'torch.ops.aten.any.dim')
            del buf3
            buf7 = reinterpret_tensor(buf2, (2, 4, 8, 8), (256, 64, 8, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.full, aten.where, aten.add, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax_add_full_matmul_where_3.run(buf7, arg2_1, arg3_1, buf4, 64, 8, stream=raw_stream0)
            del arg2_1
            del arg3_1
            del buf4
            arg4_1 = copy_if_misaligned(arg4_1)
            buf8 = empty_strided_npu((8, 8, 16), (128, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.full, aten.where, aten.add, aten._safe_softmax]
            # [Provenance debug handles] extern_kernels.bmm:3
            extern_kernels.bmm(reinterpret_tensor(buf7, (8, 8, 8), (64, 8, 1), 0), reinterpret_tensor(arg4_1, (8, 8, 16), (128, 16, 1), 0), out=buf8)
            del arg4_1
            del buf7
        return (reinterpret_tensor(buf8, (2, 4, 8, 16), (512, 128, 16, 1), 0), )

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
