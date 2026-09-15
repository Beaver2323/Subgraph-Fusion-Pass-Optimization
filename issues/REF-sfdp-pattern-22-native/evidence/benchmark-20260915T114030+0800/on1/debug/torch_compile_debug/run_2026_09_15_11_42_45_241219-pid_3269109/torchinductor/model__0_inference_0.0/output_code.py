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


# kernel path: /home/z50063656/tmp/t106-npu-results/benchmark-20260915T114030+0800/pattern-22/on1/inductor-cache/tmpwna1l8we/kk/ckkamhw5kdix7dvbiqqn3ci52p77zrsvbmmrscy6g4bajefdwnnh.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.permute, aten._to_copy, aten.mul, aten.matmul]
# Source node to ATen node mapping:
#    => clone_default, convert_element_type_default_1, mul_scalar, permute_default
# Graph fragment:
#   %arg0_1 : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %permute_default : Tensor "f16[2, 8, 4, 16][512, 16, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg0_1, [0, 2, 1, 3]), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[2, 8, 4, 16][512, 16, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_default, torch.float32), kwargs = {})
#   %mul_scalar : Tensor "f32[2, 8, 4, 16][512, 16, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type_default_1, 1.0), kwargs = {})
#   %clone_default : Tensor "f32[2, 8, 4, 16][512, 64, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_default,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_default
triton_unk_fused__to_copy_matmul_mul_permute_0 = async_compile.triton('triton_unk_fused__to_copy_matmul_mul_permute_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [16, 4, 8, 2, {'divisors': [1, 16, 64, 512]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 16, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 16, 'seed': 2}, {'name': 'x2', 'length': 8, 'divisor': 64, 'seed': 2}, {'name': 'x3', 'length': 2, 'divisor': 512, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__to_copy_matmul_mul_permute_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__to_copy_matmul_mul_permute_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 1024
    x_g_tile0 : tl.constexpr = (16) if (16) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (4) if (4) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (8) if (8) < (x_g_rem2) else (x_g_rem2)
    x_g_rem3 : tl.constexpr = ((x_g_rem2) // x_g_tile2) if ((x_g_rem2) // x_g_tile2) > 1 else 1
    x_g_tile3 : tl.constexpr = (2) if (2) < (x_g_rem3) else (x_g_rem3)
    x0numel : tl.constexpr = 16
    x1numel : tl.constexpr = 4
    x2numel : tl.constexpr = 8
    x3numel : tl.constexpr = 2
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x1 : tl.constexpr = x_g_tile1
    real_block_x2 : tl.constexpr = x_g_tile2
    real_block_x3 : tl.constexpr = x_g_tile3
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_1 = x0_blocks
    x_cumblk_2 = x0_blocks * x1_blocks
    x_cumblk_3 = x0_blocks * x1_blocks * x2_blocks
    total_blocks = x0_blocks * x1_blocks * x2_blocks * x3_blocks
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
        x2offset = (group_base + i) // x_cumblk_2 % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_3 % x3_blocks * real_block_x3
        x0index = x0offset + tl.arange(0, real_block_x0)[None, None, None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[None, None, :, None]
        x1 = x1index
        x1mask = x1index < x1numel
        x2index = x2offset + tl.arange(0, real_block_x2)[None, :, None, None]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[:, None, None, None]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x0mask & x1mask & x2mask & x3mask
        tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 128*x1 + 512*x3), x0mask & x1mask & x2mask & x3mask).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.full([1], 1.0, tl.float32)
        tmp3 = tmp1 * tmp2
        tl.store(out_ptr0 + (x0 + 16*x1 + 64*x2 + 512*x3), tmp3, x0mask & x1mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t106-npu-results/benchmark-20260915T114030+0800/pattern-22/on1/inductor-cache/tmpwna1l8we/hq/chqqohgrigaws3zogcw2zolisf5ygycxyzb4vyjicegaglxsks75.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.permute, aten._to_copy, aten.transpose, aten.mul, aten.matmul]
# Source node to ATen node mapping:
#    => clone_default_1, convert_element_type_default_2, mul_scalar_1, permute_default_1, permute_default_3
# Graph fragment:
#   %arg1_1 : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %permute_default_1 : Tensor "f16[2, 8, 4, 16][512, 16, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%arg1_1, [0, 2, 1, 3]), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[2, 8, 4, 16][512, 16, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_default_1, torch.float32), kwargs = {})
#   %permute_default_3 : Tensor "f32[2, 8, 16, 4][512, 16, 1, 128]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_default_2, [0, 1, 3, 2]), kwargs = {})
#   %mul_scalar_1 : Tensor "f32[2, 8, 16, 4][512, 16, 1, 128]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_default_3, 1.0), kwargs = {})
#   %clone_default_1 : Tensor "f32[2, 8, 16, 4][512, 64, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_default_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_default_1
triton_unk_fused__to_copy_matmul_mul_permute_transpose_1 = async_compile.triton('triton_unk_fused__to_copy_matmul_mul_permute_transpose_1', '''
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
    size_hints={'y': 256, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'YBLOCK_HINT': [128, 2, {'divisors': [1, 128]}], 'XBLOCK_HINT': [4, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 2, 'divisor': 128, 'seed': 2}, {'name': 'x2', 'length': 4, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__to_copy_matmul_mul_permute_transpose_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__to_copy_matmul_mul_permute_transpose_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    ynumel = 256
    xnumel = 4
    y_g_tile0 : tl.constexpr = (128) if (128) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (2) if (2) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 128
    y1numel : tl.constexpr = 2
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y_cumblk_1 = y0_blocks
    x_g_tile0 : tl.constexpr = (4) if (4) < (XBLOCK) else (XBLOCK)
    x2numel : tl.constexpr = 4
    real_block_x2 : tl.constexpr = x_g_tile0
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
        tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 512*y1), x2mask & y0mask & y1mask).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.full([1, 1], 1.0, tl.float32)
        tmp3 = tmp1 * tmp2
        tl.store(out_ptr0 + (x2 + 4*y0 + 512*y1), tmp3, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t106-npu-results/benchmark-20260915T114030+0800/pattern-22/on1/inductor-cache/tmpwna1l8we/fa/cfasa73nfzzzv7ryxnjy67zojq5scnl45j7jg3lrbqslhze6edyk.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten._to_copy, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#    => add_tensor, convert_element_type_default, eq_scalar, logical_not_default, view_default_2
# Graph fragment:
#   %bmm_default : Tensor "f32[16, 4, 4][16, 4, 1]npu:0" = PlaceHolder[target=bmm_default]
#   %arg3_1 : Tensor "f32[2, 1, 1, 4][4, 4, 4, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %view_default_2 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default, [2, 8, 4, 4]), kwargs = {})
#   %convert_element_type_default : Tensor "f16[2, 1, 1, 4][4, 4, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float16), kwargs = {})
#   %add_tensor : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %convert_element_type_default), kwargs = {})
#   %eq_scalar : Tensor "b8[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_tensor, -inf), kwargs = {})
#   %logical_not_default : Tensor "b8[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_scalar,), kwargs = {})
#   return %logical_not_default
triton_unk_fused__safe_softmax__to_copy_add_matmul_2 = async_compile.triton('triton_unk_fused__safe_softmax__to_copy_add_matmul_2', '''
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
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [4, 32, 2, {'divisors': [1, 4, 128]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 32, 'divisor': 4, 'seed': 2}, {'name': 'x2', 'length': 2, 'divisor': 128, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax__to_copy_add_matmul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__safe_softmax__to_copy_add_matmul_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 256
    x_g_tile0 : tl.constexpr = (4) if (4) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (32) if (32) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (2) if (2) < (x_g_rem2) else (x_g_rem2)
    x0numel : tl.constexpr = 4
    x1numel : tl.constexpr = 32
    x2numel : tl.constexpr = 2
    real_block_x0 : tl.constexpr = x_g_tile0
    real_block_x1 : tl.constexpr = x_g_tile1
    real_block_x2 : tl.constexpr = x_g_tile2
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x_cumblk_1 = x0_blocks
    x_cumblk_2 = x0_blocks * x1_blocks
    total_blocks = x0_blocks * x1_blocks * x2_blocks
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
        x2offset = (group_base + i) // x_cumblk_2 % x2_blocks * real_block_x2
        x0index = x0offset + tl.arange(0, real_block_x0)[None, None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[None, :, None]
        x1 = x1index
        x1mask = x1index < x1numel
        x2index = x2offset + tl.arange(0, real_block_x2)[:, None, None]
        x2 = x2index
        x2mask = x2index < x2numel
        xmask = x0mask & x1mask & x2mask
        tmp0 = tl.load(in_ptr0 + (x0 + 4*x1 + 128*x2), x0mask & x1mask & x2mask)
        tmp1 = tl.load(in_ptr1 + (x0 + 4*x2), x0mask & x2mask, eviction_policy='evict_last')
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp0 + tmp3
        tmp5 = tl.full([1], float("-inf"), tl.float32)
        tmp6 = tmp4 == tmp5
        tmp7 = tmp6 == 0
        tl.store(out_ptr0 + (x0 + 4*x1 + 128*x2), tmp7, x0mask & x1mask & x2mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t106-npu-results/benchmark-20260915T114030+0800/pattern-22/on1/inductor-cache/tmpwna1l8we/tk/ctkj34kjtnr74idkfikues3kssrbob6vzahevamz4f3cezo4lchf.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten._to_copy, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#    => add_tensor, amax_default, convert_element_type_default, sub_tensor, view_default_2
# Graph fragment:
#   %bmm_default : Tensor "f32[16, 4, 4][16, 4, 1]npu:0" = PlaceHolder[target=bmm_default]
#   %arg3_1 : Tensor "f32[2, 1, 1, 4][4, 4, 4, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %view_default_2 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default, [2, 8, 4, 4]), kwargs = {})
#   %convert_element_type_default : Tensor "f16[2, 1, 1, 4][4, 4, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float16), kwargs = {})
#   %add_tensor : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %convert_element_type_default), kwargs = {})
#   %amax_default : Tensor "f32[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_tensor, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_tensor, %amax_default), kwargs = {})
#   return %buf5
triton_unk_fused__safe_softmax__to_copy_add_matmul_3 = async_compile.triton('triton_unk_fused__safe_softmax__to_copy_add_matmul_3', '''
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
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [4, 32, 2, {'divisors': [1, 4, 128]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 32, 'divisor': 4, 'seed': 2}, {'name': 'x2', 'length': 2, 'divisor': 128, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax__to_copy_add_matmul_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__safe_softmax__to_copy_add_matmul_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 256
    x_g_tile0 : tl.constexpr = (4) if (4) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (32) if (32) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (2) if (2) < (x_g_rem2) else (x_g_rem2)
    x0numel : tl.constexpr = 4
    x1numel : tl.constexpr = 32
    x2numel : tl.constexpr = 2
    real_block_x0 : tl.constexpr = x_g_tile0
    real_block_x1 : tl.constexpr = x_g_tile1
    real_block_x2 : tl.constexpr = x_g_tile2
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x_cumblk_1 = x0_blocks
    x_cumblk_2 = x0_blocks * x1_blocks
    total_blocks = x0_blocks * x1_blocks * x2_blocks
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
        x2offset = (group_base + i) // x_cumblk_2 % x2_blocks * real_block_x2
        x0index = x0offset + tl.arange(0, real_block_x0)[None, None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[None, :, None]
        x1 = x1index
        x1mask = x1index < x1numel
        x2index = x2offset + tl.arange(0, real_block_x2)[:, None, None]
        x2 = x2index
        x2mask = x2index < x2numel
        xmask = x0mask & x1mask & x2mask
        _es_lane0 = tl.arange(0, 4)[None, None, :]
        _es_full0 = tl.load(in_ptr0 + (4 * x1 + 128 * x2) + _es_lane0, x1mask & x2mask & ((4 * x1 + 128 * x2 + _es_lane0 >= 0) & (4 * x1 + 128 * x2 + _es_lane0 < 256)), eviction_policy='evict_last')
        tmp0 = extract_slice(_es_full0, [0, 0, 0], [_es_full0.shape[0], _es_full0.shape[1], 1], [1, 1, 1])
        _es_lane1 = tl.arange(0, 4)[None, None, :]
        _es_full1 = tl.load(in_ptr1 + 4 * x2 + _es_lane1, x2mask & ((4 * x2 + _es_lane1 >= 0) & (4 * x2 + _es_lane1 < 8)), eviction_policy='evict_last')
        tmp1 = extract_slice(_es_full1, [0, 0, 0], [_es_full1.shape[0], _es_full1.shape[1], 1], [1, 1, 1])
        _es_lane2 = tl.arange(0, 4)[None, None, :]
        _es_full2 = tl.load(in_ptr0 + (1 + 4 * x1 + 128 * x2) + _es_lane2, x1mask & x2mask & ((1 + 4 * x1 + 128 * x2 + _es_lane2 >= 0) & (1 + 4 * x1 + 128 * x2 + _es_lane2 < 256)), eviction_policy='evict_last')
        tmp5 = extract_slice(_es_full2, [0, 0, 0], [_es_full2.shape[0], _es_full2.shape[1], 1], [1, 1, 1])
        _es_lane3 = tl.arange(0, 4)[None, None, :]
        _es_full3 = tl.load(in_ptr1 + (1 + 4 * x2) + _es_lane3, x2mask & ((1 + 4 * x2 + _es_lane3 >= 0) & (1 + 4 * x2 + _es_lane3 < 8)), eviction_policy='evict_last')
        tmp6 = extract_slice(_es_full3, [0, 0, 0], [_es_full3.shape[0], _es_full3.shape[1], 1], [1, 1, 1])
        _es_lane4 = tl.arange(0, 4)[None, None, :]
        _es_full4 = tl.load(in_ptr0 + (2 + 4 * x1 + 128 * x2) + _es_lane4, x1mask & x2mask & ((2 + 4 * x1 + 128 * x2 + _es_lane4 >= 0) & (2 + 4 * x1 + 128 * x2 + _es_lane4 < 256)), eviction_policy='evict_last')
        tmp11 = extract_slice(_es_full4, [0, 0, 0], [_es_full4.shape[0], _es_full4.shape[1], 1], [1, 1, 1])
        _es_lane5 = tl.arange(0, 4)[None, None, :]
        _es_full5 = tl.load(in_ptr1 + (2 + 4 * x2) + _es_lane5, x2mask & ((2 + 4 * x2 + _es_lane5 >= 0) & (2 + 4 * x2 + _es_lane5 < 8)), eviction_policy='evict_last')
        tmp12 = extract_slice(_es_full5, [0, 0, 0], [_es_full5.shape[0], _es_full5.shape[1], 1], [1, 1, 1])
        _es_lane6 = tl.arange(0, 4)[None, None, :]
        _es_full6 = tl.load(in_ptr0 + (3 + 4 * x1 + 128 * x2) + _es_lane6, x1mask & x2mask & ((3 + 4 * x1 + 128 * x2 + _es_lane6 >= 0) & (3 + 4 * x1 + 128 * x2 + _es_lane6 < 256)), eviction_policy='evict_last')
        tmp17 = extract_slice(_es_full6, [0, 0, 0], [_es_full6.shape[0], _es_full6.shape[1], 1], [1, 1, 1])
        _es_lane7 = tl.arange(0, 4)[None, None, :]
        _es_full7 = tl.load(in_ptr1 + (3 + 4 * x2) + _es_lane7, x2mask & ((3 + 4 * x2 + _es_lane7 >= 0) & (3 + 4 * x2 + _es_lane7 < 8)), eviction_policy='evict_last')
        tmp18 = extract_slice(_es_full7, [0, 0, 0], [_es_full7.shape[0], _es_full7.shape[1], 1], [1, 1, 1])
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp0 + tmp3
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp5 + tmp8
        tmp10 = tl.maximum(tmp4, tmp9)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp11 + tmp14
        tmp16 = tl.maximum(tmp10, tmp15)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp17 + tmp20
        tmp22 = tl.maximum(tmp16, tmp21)
        tl.store(out_ptr0 + (x0 + 4*x1 + 128*x2), tmp22, x0mask & x1mask & x2mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t106-npu-results/benchmark-20260915T114030+0800/pattern-22/on1/inductor-cache/tmpwna1l8we/yd/cydwdu6cyvwgysce5vrjpsfhdbjbreaxwvew3e7knrp6cwwjdt3q.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten._to_copy, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#    => add_tensor, amax_default, convert_element_type_default, exp_default, sub_tensor, sum_dim_int_list, view_default_2
# Graph fragment:
#   %bmm_default : Tensor "f32[16, 4, 4][16, 4, 1]npu:0" = PlaceHolder[target=bmm_default]
#   %arg3_1 : Tensor "f32[2, 1, 1, 4][4, 4, 4, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %buf5 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0" = PlaceHolder[target=buf5]
#   %view_default_2 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default, [2, 8, 4, 4]), kwargs = {})
#   %convert_element_type_default : Tensor "f16[2, 1, 1, 4][4, 4, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float16), kwargs = {})
#   %add_tensor : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %convert_element_type_default), kwargs = {})
#   %amax_default : Tensor "f32[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_tensor, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_tensor, %amax_default), kwargs = {})
#   %exp_default : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %sum_dim_int_list : Tensor "f32[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_default, [-1], True), kwargs = {})
#   return %sum_dim_int_list
triton_unk_fused__safe_softmax__to_copy_add_matmul_4 = async_compile.triton('triton_unk_fused__safe_softmax__to_copy_add_matmul_4', '''
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
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [32, 2, {'divisors': [1, 32]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 32, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 2, 'divisor': 32, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax__to_copy_add_matmul_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__safe_softmax__to_copy_add_matmul_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 64
    x_g_tile0 : tl.constexpr = (32) if (32) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (2) if (2) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 32
    x1numel : tl.constexpr = 2
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
        _es_slane0 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask0 = (4*x0offset + _es_slane0) < 4*x0numel
        _es_sfull0 = tl.load(in_ptr0 + (128 * x1 + 4 * x0offset + _es_slane0), _es_smask0 & x1mask, eviction_policy='evict_last')
        tmp0 = extract_slice(_es_sfull0, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_lane1 = tl.arange(0, 4)[None, :]
        _es_full1 = tl.load(in_ptr1 + 4 * x1 + _es_lane1, x1mask & ((4 * x1 + _es_lane1 >= 0) & (4 * x1 + _es_lane1 < 8)), eviction_policy='evict_last')
        tmp1 = extract_slice(_es_full1, [0, 0], [_es_full1.shape[0], 1], [1, 1])
        _es_slane2 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask2 = (4*x0offset + _es_slane2) < 4*x0numel
        _es_sfull2 = tl.load(in_ptr2 + (128 * x1 + 4 * x0offset + _es_slane2), _es_smask2 & x1mask, eviction_policy='evict_last')
        tmp5 = extract_slice(_es_sfull2, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_slane3 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask3 = (4*x0offset + _es_slane3) < 4*x0numel
        _es_sfull3 = tl.load(in_ptr0 + (1 + 128 * x1 + 4 * x0offset + _es_slane3), _es_smask3 & x1mask, eviction_policy='evict_last')
        tmp8 = extract_slice(_es_sfull3, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_lane4 = tl.arange(0, 4)[None, :]
        _es_full4 = tl.load(in_ptr1 + (1 + 4 * x1) + _es_lane4, x1mask & ((1 + 4 * x1 + _es_lane4 >= 0) & (1 + 4 * x1 + _es_lane4 < 8)), eviction_policy='evict_last')
        tmp9 = extract_slice(_es_full4, [0, 0], [_es_full4.shape[0], 1], [1, 1])
        _es_slane5 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask5 = (4*x0offset + _es_slane5) < 4*x0numel
        _es_sfull5 = tl.load(in_ptr2 + (1 + 128 * x1 + 4 * x0offset + _es_slane5), _es_smask5 & x1mask, eviction_policy='evict_last')
        tmp13 = extract_slice(_es_sfull5, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_slane6 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask6 = (4*x0offset + _es_slane6) < 4*x0numel
        _es_sfull6 = tl.load(in_ptr0 + (2 + 128 * x1 + 4 * x0offset + _es_slane6), _es_smask6 & x1mask, eviction_policy='evict_last')
        tmp17 = extract_slice(_es_sfull6, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_lane7 = tl.arange(0, 4)[None, :]
        _es_full7 = tl.load(in_ptr1 + (2 + 4 * x1) + _es_lane7, x1mask & ((2 + 4 * x1 + _es_lane7 >= 0) & (2 + 4 * x1 + _es_lane7 < 8)), eviction_policy='evict_last')
        tmp18 = extract_slice(_es_full7, [0, 0], [_es_full7.shape[0], 1], [1, 1])
        _es_slane8 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask8 = (4*x0offset + _es_slane8) < 4*x0numel
        _es_sfull8 = tl.load(in_ptr2 + (2 + 128 * x1 + 4 * x0offset + _es_slane8), _es_smask8 & x1mask, eviction_policy='evict_last')
        tmp22 = extract_slice(_es_sfull8, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_slane9 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask9 = (4*x0offset + _es_slane9) < 4*x0numel
        _es_sfull9 = tl.load(in_ptr0 + (3 + 128 * x1 + 4 * x0offset + _es_slane9), _es_smask9 & x1mask, eviction_policy='evict_last')
        tmp26 = extract_slice(_es_sfull9, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_lane10 = tl.arange(0, 4)[None, :]
        _es_full10 = tl.load(in_ptr1 + (3 + 4 * x1) + _es_lane10, x1mask & ((3 + 4 * x1 + _es_lane10 >= 0) & (3 + 4 * x1 + _es_lane10 < 8)), eviction_policy='evict_last')
        tmp27 = extract_slice(_es_full10, [0, 0], [_es_full10.shape[0], 1], [1, 1])
        _es_slane11 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask11 = (4*x0offset + _es_slane11) < 4*x0numel
        _es_sfull11 = tl.load(in_ptr2 + (3 + 128 * x1 + 4 * x0offset + _es_slane11), _es_smask11 & x1mask, eviction_policy='evict_last')
        tmp31 = extract_slice(_es_sfull11, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp0 + tmp3
        tmp6 = tmp4 - tmp5
        tmp7 = libdevice.exp(tmp6)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp8 + tmp11
        tmp14 = tmp12 - tmp13
        tmp15 = libdevice.exp(tmp14)
        tmp16 = tmp7 + tmp15
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp17 + tmp20
        tmp23 = tmp21 - tmp22
        tmp24 = libdevice.exp(tmp23)
        tmp25 = tmp16 + tmp24
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp26 + tmp29
        tmp32 = tmp30 - tmp31
        tmp33 = libdevice.exp(tmp32)
        tmp34 = tmp25 + tmp33
        tl.store(out_ptr0 + (x0 + 32*x1), tmp34, x0mask & x1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t106-npu-results/benchmark-20260915T114030+0800/pattern-22/on1/inductor-cache/tmpwna1l8we/th/cthhhsf2xu75mkk5diqewg6f3gm3og7iqz3xpsi5dikfumepiiyz.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten._to_copy, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#    => add_tensor, amax_default, convert_element_type_default, div_tensor, exp_default, sub_tensor, view_default_2
# Graph fragment:
#   %sum_dim_int_list : Tensor "f32[2, 8, 4, 1][32, 4, 1, 64]npu:0" = PlaceHolder[target=sum_dim_int_list]
#   %view_default_2 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default, [2, 8, 4, 4]), kwargs = {})
#   %convert_element_type_default : Tensor "f16[2, 1, 1, 4][4, 4, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float16), kwargs = {})
#   %add_tensor : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %convert_element_type_default), kwargs = {})
#   %amax_default : Tensor "f32[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_tensor, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_tensor, %amax_default), kwargs = {})
#   %exp_default : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_tensor : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %sum_dim_int_list), kwargs = {})
#   return %buf7
triton_unk_fused__safe_softmax__to_copy_add_matmul_5 = async_compile.triton('triton_unk_fused__safe_softmax__to_copy_add_matmul_5', '''
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
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [4, 64, {'divisors': [1, 4]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 64, 'divisor': 4, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax__to_copy_add_matmul_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__safe_softmax__to_copy_add_matmul_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 256
    x_g_tile0 : tl.constexpr = (4) if (4) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (64) if (64) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 4
    x1numel : tl.constexpr = 64
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
        tl.store(out_ptr0 + (x0 + 4*x1), tmp0, x0mask & x1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t106-npu-results/benchmark-20260915T114030+0800/pattern-22/on1/inductor-cache/tmpwna1l8we/5i/c5ibgktsksgksrg3egbilllc6qvmrmgom4nxl5efaqcpzfgphzle.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten._to_copy, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#    => add_tensor, amax_default, convert_element_type_default, div_tensor, exp_default, full_default, logical_not_default_1, sub_tensor, view_default_2, where_self
# Graph fragment:
#   %any_dim : Tensor "b8[2, 8, 4, 1][32, 4, 1, 1]npu:0" = PlaceHolder[target=any_dim]
#   %view_default_2 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default, [2, 8, 4, 4]), kwargs = {})
#   %convert_element_type_default : Tensor "f16[2, 1, 1, 4][4, 4, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float16), kwargs = {})
#   %add_tensor : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %convert_element_type_default), kwargs = {})
#   %logical_not_default_1 : Tensor "b8[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dim,), kwargs = {})
#   %full_default : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 8, 4, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %amax_default : Tensor "f32[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_tensor, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_tensor, %amax_default), kwargs = {})
#   %exp_default : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_tensor : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %sum_dim_int_list), kwargs = {})
#   %where_self : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_1, %full_default, %div_tensor), kwargs = {})
#   return %buf8
triton_unk_fused__safe_softmax__to_copy_add_matmul_6 = async_compile.triton('triton_unk_fused__safe_softmax__to_copy_add_matmul_6', '''
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
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [4, 64, {'divisors': [1, 4]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 64, 'divisor': 4, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax__to_copy_add_matmul_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__safe_softmax__to_copy_add_matmul_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 256
    x_g_tile0 : tl.constexpr = (4) if (4) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (64) if (64) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 4
    x1numel : tl.constexpr = 64
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
        tmp0 = tl.load(in_ptr0 + x1, x1mask, eviction_policy='evict_last') != 0
        tmp1 = tmp0 == 0
        tl.store(out_ptr0 + (x0 + 4*x1), tmp1, x0mask & x1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t106-npu-results/benchmark-20260915T114030+0800/pattern-22/on1/inductor-cache/tmpwna1l8we/ua/cua5e6wrjtqsvznwyi7v43yftebetkyawgbylpf76aqj2bog5jhu.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten._to_copy, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#    => add_tensor, amax_default, convert_element_type_default, div_tensor, exp_default, full_default, logical_not_default_1, sub_tensor, view_default_2, where_self
# Graph fragment:
#   %buf8 : Tensor "b8[2, 8, 4, 4][128, 16, 4, 1]npu:0" = PlaceHolder[target=buf8]
#   %bmm_default : Tensor "f32[16, 4, 4][16, 4, 1]npu:0" = PlaceHolder[target=bmm_default]
#   %arg3_1 : Tensor "f32[2, 1, 1, 4][4, 4, 4, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %buf5 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0" = PlaceHolder[target=buf5]
#   %buf7 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0" = PlaceHolder[target=buf7]
#   %view_default_2 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default, [2, 8, 4, 4]), kwargs = {})
#   %convert_element_type_default : Tensor "f16[2, 1, 1, 4][4, 4, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float16), kwargs = {})
#   %add_tensor : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %convert_element_type_default), kwargs = {})
#   %logical_not_default_1 : Tensor "b8[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dim,), kwargs = {})
#   %full_default : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 8, 4, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %amax_default : Tensor "f32[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_tensor, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_tensor, %amax_default), kwargs = {})
#   %exp_default : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_tensor : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %sum_dim_int_list), kwargs = {})
#   %where_self : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_1, %full_default, %div_tensor), kwargs = {})
#   return %expand_default_2
triton_unk_fused__safe_softmax__to_copy_add_matmul_7 = async_compile.triton('triton_unk_fused__safe_softmax__to_copy_add_matmul_7', '''
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
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [4, 32, 2, {'divisors': [1, 4, 128]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 32, 'divisor': 4, 'seed': 2}, {'name': 'x2', 'length': 2, 'divisor': 128, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax__to_copy_add_matmul_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__safe_softmax__to_copy_add_matmul_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 256
    x_g_tile0 : tl.constexpr = (4) if (4) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (32) if (32) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (2) if (2) < (x_g_rem2) else (x_g_rem2)
    x0numel : tl.constexpr = 4
    x1numel : tl.constexpr = 32
    x2numel : tl.constexpr = 2
    real_block_x0 : tl.constexpr = x_g_tile0
    real_block_x1 : tl.constexpr = x_g_tile1
    real_block_x2 : tl.constexpr = x_g_tile2
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x_cumblk_1 = x0_blocks
    x_cumblk_2 = x0_blocks * x1_blocks
    total_blocks = x0_blocks * x1_blocks * x2_blocks
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
        x2offset = (group_base + i) // x_cumblk_2 % x2_blocks * real_block_x2
        x0index = x0offset + tl.arange(0, real_block_x0)[None, None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[None, :, None]
        x1 = x1index
        x1mask = x1index < x1numel
        x2index = x2offset + tl.arange(0, real_block_x2)[:, None, None]
        x2 = x2index
        x2mask = x2index < x2numel
        xmask = x0mask & x1mask & x2mask
        tmp0 = tl.load(in_ptr0 + (x0 + 4 * x1 + 128 * x2), x0mask & x1mask & x2mask) != 0
        tmp1 = tl.load(in_out_ptr0 + (x0 + 4*x1 + 128*x2), x0mask & x1mask & x2mask)
        tmp2 = tl.load(in_ptr1 + (x0 + 4*x2), x0mask & x2mask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (x0 + 4*x1 + 128*x2), x0mask & x1mask & x2mask)
        tmp9 = tl.load(in_ptr3 + (x0 + 4*x1 + 128*x2), x0mask & x1mask & x2mask)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp1 + tmp4
        tmp7 = tmp5 - tmp6
        tmp8 = libdevice.exp(tmp7)
        tmp10 = (tmp8 / tmp9)
        tmp11 = tl.full([1], 0.0, tl.float32)
        tmp12 = tl.where(tmp0, tmp11, tmp10)
        tl.store(in_out_ptr0 + (x0 + 4*x1 + 128*x2), tmp12, x0mask & x1mask & x2mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t106-npu-results/benchmark-20260915T114030+0800/pattern-22/on1/inductor-cache/tmpwna1l8we/eq/ceqppph7dlrdb343nyd64yo2opblkp3j2vll6v6www3y4qqujlom.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.permute, aten._to_copy, aten.matmul]
# Source node to ATen node mapping:
#    => clone_default_2, convert_element_type_default_3, permute_default_2
# Graph fragment:
#   %arg2_1 : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %permute_default_2 : Tensor "f16[2, 8, 4, 16][512, 16, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%arg2_1, [0, 2, 1, 3]), kwargs = {})
#   %convert_element_type_default_3 : Tensor "f32[2, 8, 4, 16][512, 16, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_default_2, torch.float32), kwargs = {})
#   %clone_default_2 : Tensor "f32[2, 8, 4, 16][512, 64, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_default_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_default_2
triton_unk_fused__to_copy_matmul_permute_8 = async_compile.triton('triton_unk_fused__to_copy_matmul_permute_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [16, 4, 8, 2, {'divisors': [1, 16, 64, 512]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 16, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 16, 'seed': 2}, {'name': 'x2', 'length': 8, 'divisor': 64, 'seed': 2}, {'name': 'x3', 'length': 2, 'divisor': 512, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__to_copy_matmul_permute_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__to_copy_matmul_permute_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 1024
    x_g_tile0 : tl.constexpr = (16) if (16) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (4) if (4) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (8) if (8) < (x_g_rem2) else (x_g_rem2)
    x_g_rem3 : tl.constexpr = ((x_g_rem2) // x_g_tile2) if ((x_g_rem2) // x_g_tile2) > 1 else 1
    x_g_tile3 : tl.constexpr = (2) if (2) < (x_g_rem3) else (x_g_rem3)
    x0numel : tl.constexpr = 16
    x1numel : tl.constexpr = 4
    x2numel : tl.constexpr = 8
    x3numel : tl.constexpr = 2
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x1 : tl.constexpr = x_g_tile1
    real_block_x2 : tl.constexpr = x_g_tile2
    real_block_x3 : tl.constexpr = x_g_tile3
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_1 = x0_blocks
    x_cumblk_2 = x0_blocks * x1_blocks
    x_cumblk_3 = x0_blocks * x1_blocks * x2_blocks
    total_blocks = x0_blocks * x1_blocks * x2_blocks * x3_blocks
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
        x2offset = (group_base + i) // x_cumblk_2 % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_3 % x3_blocks * real_block_x3
        x0index = x0offset + tl.arange(0, real_block_x0)[None, None, None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x1index = x1offset + tl.arange(0, real_block_x1)[None, None, :, None]
        x1 = x1index
        x1mask = x1index < x1numel
        x2index = x2offset + tl.arange(0, real_block_x2)[None, :, None, None]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[:, None, None, None]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x0mask & x1mask & x2mask & x3mask
        tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 128*x1 + 512*x3), x0mask & x1mask & x2mask & x3mask).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tl.store(out_ptr0 + (x0 + 16*x1 + 64*x2 + 512*x3), tmp1, x0mask & x1mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t106-npu-results/benchmark-20260915T114030+0800/pattern-22/on1/inductor-cache/tmpwna1l8we/ra/crasb3ccsyoeryap2fdlo737bxukx7wtzj4kfpvkq4oxsi7u66u4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul]
# Source node to ATen node mapping:
#    => convert_element_type_default_4, view_default_5
# Graph fragment:
#   %bmm_default_1 : Tensor "f32[16, 4, 16][64, 16, 1]npu:0" = PlaceHolder[target=bmm_default_1]
#   %view_default_5 : Tensor "f32[2, 8, 4, 16][512, 64, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default_1, [2, 8, 4, 16]), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f16[2, 8, 4, 16][512, 64, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_default_5, torch.float16), kwargs = {})
#   return %convert_element_type_default_4
triton_unk_fused_matmul_9 = async_compile.triton('triton_unk_fused_matmul_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [1024, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 1024, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        tmp1 = tmp0.to(tl.float32)
        tl.store(out_ptr0 + (x0), tmp1, x0mask)
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
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_npu((2, 8, 4, 16), (512, 64, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.permute, aten._to_copy, aten.mul, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_matmul_mul_permute_0.run(arg0_1, buf0, 1024, stream=raw_stream0)
            del arg0_1
            arg1_1 = copy_if_misaligned(arg1_1)
            buf1 = empty_strided_npu((2, 8, 16, 4), (512, 64, 4, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.permute, aten._to_copy, aten.transpose, aten.mul, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_matmul_mul_permute_transpose_1.run(arg1_1, buf1, 256, 4, stream=raw_stream0)
            buf2 = empty_strided_npu((16, 4, 4), (16, 4, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.permute, aten._to_copy, aten.mul, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:1
            extern_kernels.bmm(reinterpret_tensor(buf0, (16, 4, 16), (64, 16, 1), 0), reinterpret_tensor(buf1, (16, 16, 4), (64, 4, 1), 0), out=buf2)
            del buf0
            del buf1
            arg3_1 = copy_if_misaligned(arg3_1)
            buf3 = empty_strided_npu((2, 8, 4, 4), (128, 16, 4, 1), torch.bool)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten._to_copy, aten.add, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax__to_copy_add_matmul_2.run(buf2, arg3_1, buf3, 256, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten._to_copy, aten.add, aten._safe_softmax]
            # [Provenance debug handles] torch.ops.aten.any.dim:2
            buf4 = torch.ops.aten.any.dim(buf3, -1, True)
            assert_alignment(buf4, 16, 'torch.ops.aten.any.dim')
            del buf3
            buf5 = empty_strided_npu((2, 8, 4, 4), (128, 16, 4, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten._to_copy, aten.add, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax__to_copy_add_matmul_3.run(buf2, arg3_1, buf5, 256, stream=raw_stream0)
            buf6 = empty_strided_npu((2, 8, 4, 1), (32, 4, 1, 64), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten._to_copy, aten.add, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax__to_copy_add_matmul_4.run(buf2, arg3_1, buf5, buf6, 64, stream=raw_stream0)
            buf7 = empty_strided_npu((2, 8, 4, 4), (128, 16, 4, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten._to_copy, aten.add, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax__to_copy_add_matmul_5.run(buf6, buf7, 256, stream=raw_stream0)
            del buf6
            buf8 = empty_strided_npu((2, 8, 4, 4), (128, 16, 4, 1), torch.bool)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten._to_copy, aten.add, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax__to_copy_add_matmul_6.run(buf4, buf8, 256, stream=raw_stream0)
            del buf4
            buf9 = reinterpret_tensor(buf2, (2, 8, 4, 4), (128, 16, 4, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten._to_copy, aten.add, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax__to_copy_add_matmul_7.run(buf9, buf8, arg3_1, buf5, buf7, 256, stream=raw_stream0)
            del arg3_1
            del buf5
            del buf7
            del buf8
            arg2_1 = copy_if_misaligned(arg2_1)
            buf10 = empty_strided_npu((2, 8, 4, 16), (512, 64, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.permute, aten._to_copy, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_matmul_permute_8.run(arg2_1, buf10, 1024, stream=raw_stream0)
            buf11 = empty_strided_npu((16, 4, 16), (64, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten._to_copy, aten.add, aten._safe_softmax, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:3
            extern_kernels.bmm(reinterpret_tensor(buf9, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf10, (16, 4, 16), (64, 16, 1), 0), out=buf11)
            del buf10
            del buf9
            buf12 = empty_strided_npu((2, 8, 4, 16), (512, 64, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_9.run(buf11, buf12, 1024, stream=raw_stream0)
            del buf11
        return (buf12, reinterpret_tensor(arg1_1, (2, 8, 4, 16), (512, 16, 128, 1), 0), reinterpret_tensor(arg2_1, (2, 8, 4, 16), (512, 16, 128, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float16)
    arg1_1 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float16)
    arg2_1 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float16)
    arg3_1 = rand_strided((2, 1, 1, 4), (4, 4, 4, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
