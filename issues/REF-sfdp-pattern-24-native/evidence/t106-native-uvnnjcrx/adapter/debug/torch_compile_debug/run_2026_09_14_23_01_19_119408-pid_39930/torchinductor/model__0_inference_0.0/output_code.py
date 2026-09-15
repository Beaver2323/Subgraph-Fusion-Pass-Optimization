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


# kernel path: /home/z50063656/tmp/t106-native-uvnnjcrx/adapter/inductor-cache/tmpepowv7g4/rk/crkhwngi4fdav7heyyyfkc2ochrlfsx2kdxqmxdlqbx5r6bwmchd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#    => mul_scalar
# Graph fragment:
#   %arg2_1 : Tensor "f32[4, 2, 16, 32][1024, 512, 32, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %mul_scalar : Tensor "f32[4, 2, 16, 32][1024, 512, 32, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%arg2_1, 1.0), kwargs = {})
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
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 4096
    x_g_tile0 : tl.constexpr = (4096) if (4096) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 4096
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
        tmp1 = tl.full([1], 1.0, tl.float32)
        tmp2 = tmp0 * tmp1
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t106-native-uvnnjcrx/adapter/inductor-cache/tmpepowv7g4/u7/cu75zpfjecwp3ffcmzffdwy3tz7lvw3wdfx3vvwyf3zwro6hfwiv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.mul]
# Source node to ATen node mapping:
#    => mul_scalar_1, permute_default
# Graph fragment:
#   %arg1_1 : Tensor "f32[4, 2, 16, 32][1024, 512, 32, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %permute_default : Tensor "f32[4, 2, 32, 16][1024, 512, 1, 32]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg1_1, [0, 1, 3, 2]), kwargs = {})
#   %mul_scalar_1 : Tensor "f32[4, 2, 32, 16][1024, 512, 1, 32]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_default, 1.0), kwargs = {})
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
    size_hints={'y': 256, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'YBLOCK_HINT': [32, 8, {'divisors': [1, 32]}], 'XBLOCK_HINT': [16, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 32, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 8, 'divisor': 32, 'seed': 2}, {'name': 'x2', 'length': 16, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_mul_transpose_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_mul_transpose_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    ynumel = 256
    xnumel = 16
    y_g_tile0 : tl.constexpr = (32) if (32) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (8) if (8) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 32
    y1numel : tl.constexpr = 8
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y_cumblk_1 = y0_blocks
    x_g_tile0 : tl.constexpr = (16) if (16) < (XBLOCK) else (XBLOCK)
    x2numel : tl.constexpr = 16
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
        tmp0 = tl.load(in_ptr0 + (y0 + 32*x2 + 512*y1), x2mask & y0mask & y1mask)
        tmp1 = tl.full([1, 1], 1.0, tl.float32)
        tmp2 = tmp0 * tmp1
        tl.store(out_ptr0 + (x2 + 16*y0 + 512*y1), tmp2, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t106-native-uvnnjcrx/adapter/inductor-cache/tmpepowv7g4/5q/c5qcsflxsurha5ppdvhomr6zlyt33r4fuueskzy4om2roza76myp.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#    => add_tensor, eq_scalar, logical_not_default, view_default_2
# Graph fragment:
#   %bmm_default : Tensor "f32[8, 16, 16][256, 16, 1]npu:0" = PlaceHolder[target=bmm_default]
#   %arg0_1 : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %view_default_2 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default, [4, 2, 16, 16]), kwargs = {})
#   %add_tensor : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %arg0_1), kwargs = {})
#   %eq_scalar : Tensor "b8[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_tensor, -inf), kwargs = {})
#   %logical_not_default : Tensor "b8[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_scalar,), kwargs = {})
#   return %logical_not_default
triton_unk_fused__safe_softmax_add_matmul_2 = async_compile.triton('triton_unk_fused__safe_softmax_add_matmul_2', '''
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
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [256, 8, {'divisors': [1, 256]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 256, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 8, 'divisor': 256, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax_add_matmul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__safe_softmax_add_matmul_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 2048
    x_g_tile0 : tl.constexpr = (256) if (256) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (8) if (8) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 256
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
        tmp0 = tl.load(in_ptr0 + (x0 + 256*x1), x0mask & x1mask)
        tmp1 = tl.load(in_ptr1 + (x0), x0mask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1], float("-inf"), tl.float32)
        tmp4 = tmp2 == tmp3
        tmp5 = tmp4 == 0
        tl.store(out_ptr0 + (x0 + 256*x1), tmp5, x0mask & x1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t106-native-uvnnjcrx/adapter/inductor-cache/tmpepowv7g4/76/c76ovz36qr4itck4xnwqnay5yas2fotnknasr2ksjx7d22frzy5h.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#    => add_tensor, amax_default, div_tensor, exp_default, full_default, logical_not_default_1, sub_tensor, sum_dim_int_list, view_default_2, where_self
# Graph fragment:
#   %bmm_default : Tensor "f32[8, 16, 16][256, 16, 1]npu:0" = PlaceHolder[target=bmm_default]
#   %arg0_1 : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %amax_default : Tensor "f32[4, 2, 16, 1][32, 16, 1, 128]npu:0" = PlaceHolder[target=amax_default]
#   %any_dim : Tensor "b8[4, 2, 16, 1][32, 16, 1, 1]npu:0" = PlaceHolder[target=any_dim]
#   %sum_dim_int_list : Tensor "f32[4, 2, 16, 1][32, 16, 1, 128]npu:0" = PlaceHolder[target=sum_dim_int_list]
#   %view_default_2 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default, [4, 2, 16, 16]), kwargs = {})
#   %add_tensor : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %arg0_1), kwargs = {})
#   %logical_not_default_1 : Tensor "b8[4, 2, 16, 1][32, 16, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dim,), kwargs = {})
#   %full_default : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 2, 16, 16], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %amax_default : Tensor "f32[4, 2, 16, 1][32, 16, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_tensor, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_tensor, %amax_default), kwargs = {})
#   %exp_default : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %sum_dim_int_list : Tensor "f32[4, 2, 16, 1][32, 16, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_default, [-1], True), kwargs = {})
#   %div_tensor : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %sum_dim_int_list), kwargs = {})
#   %where_self : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_1, %full_default, %div_tensor), kwargs = {})
#   return %amax_default,%sum_dim_int_list,%expand_default_2
triton_unk_fused__safe_softmax_add_matmul_3 = async_compile.triton('triton_unk_fused__safe_softmax_add_matmul_3', '''
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
    size_hints={'x': 128, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'r0_numel': 16}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [16, 8, {'divisors': [1, 16]}], 'R0_BLOCK_HINT': [16, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 16, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 8, 'divisor': 16, 'seed': 2}, {'name': 'r0_2', 'length': 16, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax_add_matmul_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused__safe_softmax_add_matmul_3(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 128
    r0_numel = 16
    x_g_tile0 : tl.constexpr = (16) if (16) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (8) if (8) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 16
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
        _tmp4 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], float("-inf"), tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (r0_2 + 16*x0 + 256*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=float("-inf"))
            tmp1 = tl.load(in_ptr0 + (r0_2 + 16*x0), r0_mask & x0mask, eviction_policy='evict_last', other=float("-inf"))
            tmp2 = tmp0 + tmp1
            tmp3 = tl.broadcast_to(tmp2, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp5 = tl.maximum(_tmp4, tmp3)
            _tmp4 = tmp5
        tmp4 = triton_helpers.max2(_tmp4, 2)[:, :, None]
        _tmp12 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp6 = tl.load(in_out_ptr0 + (r0_2 + 16*x0 + 256*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=float("-inf"))
            tmp7 = tl.load(in_ptr0 + (r0_2 + 16*x0), r0_mask & x0mask, eviction_policy='evict_last', other=float("-inf"))
            tmp8 = tmp6 + tmp7
            tmp9 = tmp8 - tmp4
            tmp10 = libdevice.exp(tmp9)
            tmp11 = tl.broadcast_to(tmp10, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp13 = _tmp12 + tmp11
            _tmp12 = tmp13
        tmp12 = tl.sum(_tmp12, 2)[:, :, None]
        tmp14 = tl.load(in_ptr1 + (x0 + 16 * x1), x0mask & x1mask, eviction_policy='evict_last') != 0
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp16 = tl.load(in_out_ptr0 + (r0_2 + 16*x0 + 256*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_first', other=0.0)
            tmp17 = tl.load(in_ptr0 + (r0_2 + 16*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp15 = tmp14 == 0
            tmp18 = tmp16 + tmp17
            tmp19 = tmp18 - tmp4
            tmp20 = libdevice.exp(tmp19)
            tmp21 = (tmp20 / tmp12)
            tmp22 = tl.full([1, 1], 0.0, tl.float32)
            tmp23 = tl.where(tmp15, tmp22, tmp21)
            tl.store(in_out_ptr0 + (r0_2 + 16*x0 + 256*x1), tmp23, r0_mask & x0mask & x1mask)
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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf0 = empty_strided_npu((4, 2, 16, 32), (1024, 512, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_mul_0.run(arg2_1, buf0, 4096, stream=raw_stream0)
            del arg2_1
            arg1_1 = copy_if_misaligned(arg1_1)
            buf1 = empty_strided_npu((4, 2, 32, 16), (1024, 512, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_mul_transpose_1.run(arg1_1, buf1, 256, 16, stream=raw_stream0)
            del arg1_1
            buf2 = empty_strided_npu((8, 16, 16), (256, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:1
            extern_kernels.bmm(reinterpret_tensor(buf0, (8, 16, 32), (512, 32, 1), 0), reinterpret_tensor(buf1, (8, 32, 16), (512, 16, 1), 0), out=buf2)
            del buf0
            del buf1
            arg0_1 = copy_if_misaligned(arg0_1)
            buf3 = empty_strided_npu((4, 2, 16, 16), (512, 256, 16, 1), torch.bool)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.add, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax_add_matmul_2.run(buf2, arg0_1, buf3, 2048, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.add, aten._safe_softmax]
            # [Provenance debug handles] torch.ops.aten.any.dim:2
            buf4 = torch.ops.aten.any.dim(buf3, -1, True)
            assert_alignment(buf4, 16, 'torch.ops.aten.any.dim')
            del buf3
            buf7 = reinterpret_tensor(buf2, (4, 2, 16, 16), (512, 256, 16, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.add, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax_add_matmul_3.run(buf7, arg0_1, buf4, 128, 16, stream=raw_stream0)
            del arg0_1
            del buf4
            arg3_1 = copy_if_misaligned(arg3_1)
            buf8 = empty_strided_npu((8, 16, 32), (512, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.add, aten._safe_softmax]
            # [Provenance debug handles] extern_kernels.bmm:3
            extern_kernels.bmm(reinterpret_tensor(buf7, (8, 16, 16), (256, 16, 1), 0), reinterpret_tensor(arg3_1, (8, 16, 32), (512, 32, 1), 0), out=buf8)
            del arg3_1
            del buf7
        return (reinterpret_tensor(buf8, (4, 2, 16, 32), (1024, 512, 32, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 1, 16, 16), (256, 256, 16, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 2, 16, 32), (1024, 512, 32, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 2, 16, 32), (1024, 512, 32, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((4, 2, 16, 32), (1024, 512, 32, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
