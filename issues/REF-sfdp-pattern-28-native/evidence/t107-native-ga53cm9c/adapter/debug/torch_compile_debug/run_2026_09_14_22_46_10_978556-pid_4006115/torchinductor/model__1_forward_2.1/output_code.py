# AOT ID: ['1_forward']
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


# kernel path: /home/z50063656/tmp/t107-native-ga53cm9c/adapter/inductor-cache/tmpen76_2w5/ux/cux66nbq56lfnordbc67kg57ndhtlkzcblw5cqyjtdmkltlm6hct.py
# Topologically Sorted Source Nodes: [permute, unbind, matmul], Original ATen: [aten.permute, aten.unbind, aten.matmul]
# Source node to ATen node mapping:
#   matmul => clone
#   permute => permute
#   unbind => unbind
# Graph fragment:
#   %primals_1 : Tensor "f16[2, 3, 4, 16, 8][1536, 512, 128, 8, 1]npu:0" = PlaceHolder[target=primals_1]
#   %permute : Tensor "f16[3, 2, 4, 8, 16][512, 1536, 128, 1, 8]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_1, [1, 0, 2, 4, 3]), kwargs = {})
#   %unbind : [num_users=3] = call_function[target=torch.ops.aten.unbind.int](args = (%permute,), kwargs = {})
#   %clone : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_unk_fused_matmul_permute_unbind_0 = async_compile.triton('triton_unk_fused_matmul_permute_unbind_0', '''
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
    size_hints={'y': 64, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'YBLOCK_HINT': [8, 4, 2, {'divisors': [1, 8, 32]}], 'XBLOCK_HINT': [16, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 8, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 4, 'divisor': 8, 'seed': 2}, {'name': 'y2', 'length': 2, 'divisor': 32, 'seed': 2}, {'name': 'x3', 'length': 16, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_unbind_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_unbind_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    ynumel = 64
    xnumel = 16
    y_g_tile0 : tl.constexpr = (8) if (8) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (4) if (4) < (y_g_rem1) else (y_g_rem1)
    y_g_rem2 : tl.constexpr = ((y_g_rem1) // y_g_tile1) if ((y_g_rem1) // y_g_tile1) > 1 else 1
    y_g_tile2 : tl.constexpr = (2) if (2) < (y_g_rem2) else (y_g_rem2)
    y0numel : tl.constexpr = 8
    y1numel : tl.constexpr = 4
    y2numel : tl.constexpr = 2
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    real_block_y2 : tl.constexpr = y_g_tile2
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y2_blocks : tl.constexpr = (y2numel + real_block_y2 - 1) // real_block_y2
    y_cumblk_1 = y0_blocks
    y_cumblk_2 = y0_blocks * y1_blocks
    x_g_tile0 : tl.constexpr = (16) if (16) < (XBLOCK) else (XBLOCK)
    x3numel : tl.constexpr = 16
    real_block_x3 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_0 = y0_blocks * y1_blocks * y2_blocks
    total_blocks = y0_blocks * y1_blocks * y2_blocks * x3_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x3offset = (group_base + i) // x_cumblk_0 % x3_blocks * real_block_x3
        x3index = x3offset + tl.arange(0, real_block_x3)[None, None, None, :]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x3mask
        y0offset = (group_base + i) % y0_blocks * real_block_y0
        y1offset = (group_base + i) // y_cumblk_1 % y1_blocks * real_block_y1
        y2offset = (group_base + i) // y_cumblk_2 % y2_blocks * real_block_y2
        y0index = y0offset + tl.arange(0, real_block_y0)[None, None, :, None]
        y0 = y0index
        y0mask = y0index < y0numel
        y1index = y1offset + tl.arange(0, real_block_y1)[None, :, None, None]
        y1 = y1index
        y1mask = y1index < y1numel
        y2index = y2offset + tl.arange(0, real_block_y2)[:, None, None, None]
        y2 = y2index
        y2mask = y2index < y2numel
        ymask = y0mask & y1mask & y2mask
        tmp0 = tl.load(in_ptr0 + (y0 + 8*x3 + 128*y1 + 1536*y2), x3mask & y0mask & y1mask & y2mask).to(tl.float32)
        tl.store(out_ptr0 + (x3 + 16*y0 + 128*y1 + 512*y2), tmp0, x3mask & y0mask & y1mask & y2mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t107-native-ga53cm9c/adapter/inductor-cache/tmpen76_2w5/6m/c6mmdadyhacd5imd234lauoepgotop2xawe4kk7voeahvrik2dl5.py
# Topologically Sorted Source Nodes: [permute, unbind, transpose, matmul], Original ATen: [aten.permute, aten.unbind, aten.transpose, aten.matmul]
# Source node to ATen node mapping:
#   matmul => clone_1
#   permute => permute
#   transpose => permute_1
#   unbind => unbind
# Graph fragment:
#   %primals_1 : Tensor "f16[2, 3, 4, 16, 8][1536, 512, 128, 8, 1]npu:0" = PlaceHolder[target=primals_1]
#   %permute : Tensor "f16[3, 2, 4, 8, 16][512, 1536, 128, 1, 8]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_1, [1, 0, 2, 4, 3]), kwargs = {})
#   %unbind : [num_users=3] = call_function[target=torch.ops.aten.unbind.int](args = (%permute,), kwargs = {})
#   %permute_1 : Tensor "f16[2, 4, 16, 8][1536, 128, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_1 : Tensor "f16[2, 4, 16, 8][512, 128, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_unk_fused_matmul_permute_transpose_unbind_1 = async_compile.triton('triton_unk_fused_matmul_permute_transpose_unbind_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [512, 2, {'divisors': [1, 512]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 512, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 2, 'divisor': 512, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_transpose_unbind_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_transpose_unbind_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 1024
    x_g_tile0 : tl.constexpr = (512) if (512) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (2) if (2) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 512
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
        tmp0 = tl.load(in_ptr0 + (512 + x0 + 1536*x1), x0mask & x1mask).to(tl.float32)
        tl.store(out_ptr0 + (x0 + 512*x1), tmp0, x0mask & x1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t107-native-ga53cm9c/adapter/inductor-cache/tmpen76_2w5/r6/cr6sgfni6s3xgq4n4cbd3k7g7lvl2h2omnsxdvnkghkxwvgruw5y.py
# Topologically Sorted Source Nodes: [matmul, ], Original ATen: [aten.matmul, aten.mul, aten._to_copy, aten.isfinite, aten.all]
# Source node to ATen node mapping:
#    => abs_default, convert_element_type_default, eq_tensor, logical_not_default, mul_tensor, mul_tensor_3, ne_scalar
#   matmul => view_2
# Graph fragment:
#   %bmm : Tensor "f16[8, 8, 8][64, 8, 1]npu:0" = PlaceHolder[target=bmm]
#   %view_2 : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2, 4, 8, 8]), kwargs = {})
#   %mul_tensor : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, 0.2), kwargs = {})
#   %convert_element_type_default : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.float32), kwargs = {})
#   %eq_tensor : Tensor "b8[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%convert_element_type_default, %convert_element_type_default), kwargs = {})
#   %abs_default : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convert_element_type_default,), kwargs = {})
#   %ne_scalar : Tensor "b8[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_default, inf), kwargs = {})
#   %mul_tensor_3 : Tensor "b8[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_tensor, %ne_scalar), kwargs = {})
#   %logical_not_default : Tensor "b8[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%mul_tensor_3,), kwargs = {})
#   return %logical_not_default
triton_unk_fused__to_copy_all_isfinite_matmul_mul_2 = async_compile.triton('triton_unk_fused__to_copy_all_isfinite_matmul_mul_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [512, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 512, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__to_copy_all_isfinite_matmul_mul_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__to_copy_all_isfinite_matmul_mul_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        tmp1 = tl.full([1], 0.2, tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 == tmp3
        tmp5 = tl_math.abs(tmp3)
        tmp6 = tl.full([1], float("inf"), tl.float32)
        tmp7 = tmp5 != tmp6
        tmp8 = tmp4 & tmp7
        tmp9 = tmp8 == 0
        tl.store(out_ptr0 + (x0), tmp9, x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t107-native-ga53cm9c/adapter/inductor-cache/tmpen76_2w5/sa/csaq65gfzv2ijfzic4uar6xwxm6xdo7t3sbq3pblob5u7zbyvoy4.py
# Topologically Sorted Source Nodes: [matmul, , softmax], Original ATen: [aten.matmul, aten.mul, aten._to_copy, aten.amax, aten.sub, aten.all, aten._softmax]
# Source node to ATen node mapping:
#    => amax_default, amax_default_1, convert_element_type_default, convert_element_type_default_1, logical_not_default_1, mul_tensor, mul_tensor_1, mul_tensor_2, sub_tensor, sub_tensor_1, where_self
#   matmul => view_2
#   softmax => convert_element_type_3, div, exp, sum_1
# Graph fragment:
#   %bmm : Tensor "f16[8, 8, 8][64, 8, 1]npu:0" = PlaceHolder[target=bmm]
#   %any_dims : Tensor "b8[2, 4, 8, 1][32, 8, 1, 1]npu:0" = PlaceHolder[target=any_dims]
#   %amax_default : Tensor "f32[2, 4, 8, 1][32, 8, 1, 64]npu:0" = PlaceHolder[target=amax_default]
#   %amax_default_1 : Tensor "f32[2, 4, 8, 1][32, 8, 1, 64]npu:0" = PlaceHolder[target=amax_default_1]
#   %sum_1 : Tensor "f32[2, 4, 8, 1][32, 8, 1, 64]npu:0" = PlaceHolder[target=sum_1]
#   %view_2 : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2, 4, 8, 8]), kwargs = {})
#   %mul_tensor : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, 0.2), kwargs = {})
#   %convert_element_type_default : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.float32), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, 1), kwargs = {})
#   %amax_default : Tensor "f32[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_1, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %amax_default), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor, 0.2), kwargs = {})
#   %amax_default_1 : Tensor "f32[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_default, [-1], True), kwargs = {})
#   %sub_tensor_1 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_default, %amax_default_1), kwargs = {})
#   %logical_not_default_1 : Tensor "b8[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dims,), kwargs = {})
#   %where_self : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_1, %mul_tensor_2, %sub_tensor_1), kwargs = {})
#   %exp : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%where_self,), kwargs = {})
#   %sum_1 : Tensor "f32[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %convert_element_type_3 : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
#   return %amax_default,%amax_default_1,%sum_1,%convert_element_type_3
triton_unk_fused__softmax__to_copy_all_amax_matmul_mul_sub_3 = async_compile.triton('triton_unk_fused__softmax__to_copy_all_amax_matmul_mul_sub_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'r0_numel': 8}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [64, {'divisors': [1]}], 'R0_BLOCK_HINT': [8, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 8, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax__to_copy_all_amax_matmul_mul_sub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused__softmax__to_copy_all_amax_matmul_mul_sub_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        _tmp5 = tl.full([real_block_x0, R0_BLOCK], float("-inf"), tl.float32)
        _tmp11 = tl.full([real_block_x0, R0_BLOCK], float("-inf"), tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tl.full([1, 1], 1.0, tl.float32)
            tmp3 = tmp1 * tmp2
            tmp4 = tl.broadcast_to(tmp3, [real_block_x0, R0_BLOCK])
            tmp6 = tl.maximum(_tmp5, tmp4)
            _tmp5 = tmp6
            tmp7 = tl.full([1, 1], 0.2, tl.float32)
            tmp8 = tmp0 * tmp7
            tmp9 = tmp8.to(tl.float32)
            tmp10 = tl.broadcast_to(tmp9, [real_block_x0, R0_BLOCK])
            tmp12 = tl.maximum(_tmp11, tmp10)
            _tmp11 = tmp12
        tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
        tmp11 = triton_helpers.max2(_tmp11, 1)[:, None]
        tmp13 = tl.load(in_ptr0 + x0, x0mask, eviction_policy='evict_last') != 0
        _tmp28 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp15 = tl.load(in_out_ptr0 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp14 = tmp13 == 0
            tmp16 = tmp15.to(tl.float32)
            tmp17 = tl.full([1, 1], 1.0, tl.float32)
            tmp18 = tmp16 * tmp17
            tmp19 = tmp18 - tmp5
            tmp20 = tl.full([1, 1], 0.2, tl.float32)
            tmp21 = tmp19 * tmp20
            tmp22 = tmp15 * tmp20
            tmp23 = tmp22.to(tl.float32)
            tmp24 = tmp23 - tmp11
            tmp25 = tl.where(tmp14, tmp21, tmp24)
            tmp26 = libdevice.exp(tmp25)
            tmp27 = tl.broadcast_to(tmp26, [real_block_x0, R0_BLOCK])
            tmp29 = _tmp28 + tmp27
            _tmp28 = tl.where(r0_mask & xmask, tmp29, _tmp28)
        tmp28 = tl.sum(_tmp28, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp31 = tl.load(in_out_ptr0 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp30 = tmp13 == 0
            tmp32 = tmp31.to(tl.float32)
            tmp33 = tl.full([1, 1], 1.0, tl.float32)
            tmp34 = tmp32 * tmp33
            tmp35 = tmp34 - tmp5
            tmp36 = tl.full([1, 1], 0.2, tl.float32)
            tmp37 = tmp35 * tmp36
            tmp38 = tmp31 * tmp36
            tmp39 = tmp38.to(tl.float32)
            tmp40 = tmp39 - tmp11
            tmp41 = tl.where(tmp30, tmp37, tmp40)
            tmp42 = libdevice.exp(tmp41)
            tmp43 = (tmp42 / tmp28)
            tmp44 = tmp43.to(tl.float32)
            tmp45 = tmp44.to(tl.float16)
            tmp46 = tmp45.to(tl.float32)
            tl.store(in_out_ptr0 + (r0_1 + 8*x0), tmp46, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t107-native-ga53cm9c/adapter/inductor-cache/tmpen76_2w5/p5/cp5v5jwqhsnf7rpqzq5xn6ht4mzmmj3kme2p4gzi34uze3bwox6x.py
# Topologically Sorted Source Nodes: [permute, unbind, matmul_1], Original ATen: [aten.permute, aten.unbind, aten.matmul]
# Source node to ATen node mapping:
#   matmul_1 => clone_2
#   permute => permute
#   unbind => unbind
# Graph fragment:
#   %primals_1 : Tensor "f16[2, 3, 4, 16, 8][1536, 512, 128, 8, 1]npu:0" = PlaceHolder[target=primals_1]
#   %permute : Tensor "f16[3, 2, 4, 8, 16][512, 1536, 128, 1, 8]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_1, [1, 0, 2, 4, 3]), kwargs = {})
#   %unbind : [num_users=3] = call_function[target=torch.ops.aten.unbind.int](args = (%permute,), kwargs = {})
#   %clone_2 : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_unk_fused_matmul_permute_unbind_4 = async_compile.triton('triton_unk_fused_matmul_permute_unbind_4', '''
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
    size_hints={'y': 64, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'YBLOCK_HINT': [8, 4, 2, {'divisors': [1, 8, 32]}], 'XBLOCK_HINT': [16, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 8, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 4, 'divisor': 8, 'seed': 2}, {'name': 'y2', 'length': 2, 'divisor': 32, 'seed': 2}, {'name': 'x3', 'length': 16, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_unbind_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_unbind_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    ynumel = 64
    xnumel = 16
    y_g_tile0 : tl.constexpr = (8) if (8) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (4) if (4) < (y_g_rem1) else (y_g_rem1)
    y_g_rem2 : tl.constexpr = ((y_g_rem1) // y_g_tile1) if ((y_g_rem1) // y_g_tile1) > 1 else 1
    y_g_tile2 : tl.constexpr = (2) if (2) < (y_g_rem2) else (y_g_rem2)
    y0numel : tl.constexpr = 8
    y1numel : tl.constexpr = 4
    y2numel : tl.constexpr = 2
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    real_block_y2 : tl.constexpr = y_g_tile2
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y2_blocks : tl.constexpr = (y2numel + real_block_y2 - 1) // real_block_y2
    y_cumblk_1 = y0_blocks
    y_cumblk_2 = y0_blocks * y1_blocks
    x_g_tile0 : tl.constexpr = (16) if (16) < (XBLOCK) else (XBLOCK)
    x3numel : tl.constexpr = 16
    real_block_x3 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_0 = y0_blocks * y1_blocks * y2_blocks
    total_blocks = y0_blocks * y1_blocks * y2_blocks * x3_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x3offset = (group_base + i) // x_cumblk_0 % x3_blocks * real_block_x3
        x3index = x3offset + tl.arange(0, real_block_x3)[None, None, None, :]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x3mask
        y0offset = (group_base + i) % y0_blocks * real_block_y0
        y1offset = (group_base + i) // y_cumblk_1 % y1_blocks * real_block_y1
        y2offset = (group_base + i) // y_cumblk_2 % y2_blocks * real_block_y2
        y0index = y0offset + tl.arange(0, real_block_y0)[None, None, :, None]
        y0 = y0index
        y0mask = y0index < y0numel
        y1index = y1offset + tl.arange(0, real_block_y1)[None, :, None, None]
        y1 = y1index
        y1mask = y1index < y1numel
        y2index = y2offset + tl.arange(0, real_block_y2)[:, None, None, None]
        y2 = y2index
        y2mask = y2index < y2numel
        ymask = y0mask & y1mask & y2mask
        tmp0 = tl.load(in_ptr0 + (1024 + y0 + 8*x3 + 128*y1 + 1536*y2), x3mask & y0mask & y1mask & y2mask).to(tl.float32)
        tl.store(out_ptr0 + (x3 + 16*y0 + 128*y1 + 512*y2), tmp0, x3mask & y0mask & y1mask & y2mask)
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
        primals_1, = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            primals_1 = copy_if_misaligned(primals_1)
            buf0 = empty_strided_npu((2, 4, 8, 16), (512, 128, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [permute, unbind, matmul], Original ATen: [aten.permute, aten.unbind, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_unbind_0.run(primals_1, buf0, 64, 16, stream=raw_stream0)
            buf1 = empty_strided_npu((2, 4, 16, 8), (512, 128, 8, 1), torch.float16)
            # Topologically Sorted Source Nodes: [permute, unbind, transpose, matmul], Original ATen: [aten.permute, aten.unbind, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_transpose_unbind_1.run(primals_1, buf1, 1024, stream=raw_stream0)
            buf2 = empty_strided_npu((8, 8, 8), (64, 8, 1), torch.float16)
            # Topologically Sorted Source Nodes: [permute, unbind, transpose, matmul], Original ATen: [aten.permute, aten.unbind, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:1
            extern_kernels.bmm(reinterpret_tensor(buf0, (8, 8, 16), (128, 16, 1), 0), reinterpret_tensor(buf1, (8, 16, 8), (128, 8, 1), 0), out=buf2)
            del buf0
            del buf1
            buf5 = empty_strided_npu((2, 4, 8, 8), (256, 64, 8, 1), torch.bool)
            # Topologically Sorted Source Nodes: [matmul, ], Original ATen: [aten.matmul, aten.mul, aten._to_copy, aten.isfinite, aten.all]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_all_isfinite_matmul_mul_2.run(buf2, buf5, 512, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [matmul, ], Original ATen: [aten.matmul, aten.mul, aten._to_copy, aten.isfinite, aten.all]
            # [Provenance debug handles] torch.ops.aten.any.dims:2
            buf6 = torch.ops.aten.any.dims(buf5, [-1], True)
            assert_alignment(buf6, 16, 'torch.ops.aten.any.dims')
            del buf5
            buf8 = reinterpret_tensor(buf2, (2, 4, 8, 8), (256, 64, 8, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [matmul, , softmax], Original ATen: [aten.matmul, aten.mul, aten._to_copy, aten.amax, aten.sub, aten.all, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax__to_copy_all_amax_matmul_mul_sub_3.run(buf8, buf6, 64, 8, stream=raw_stream0)
            del buf6
            # Topologically Sorted Source Nodes: [dropout], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:3
            buf9 = torch.ops.npu._npu_dropout.default(buf8, 0.1)
            buf10 = buf9[0]
            assert_alignment(buf10, 16, 'torch.ops.npu._npu_dropout.default')
            buf11 = buf9[1]
            assert_alignment(buf11, 16, 'torch.ops.npu._npu_dropout.default')
            del buf9
            buf12 = empty_strided_npu((2, 4, 8, 16), (512, 128, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [permute, unbind, matmul_1], Original ATen: [aten.permute, aten.unbind, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_unbind_4.run(primals_1, buf12, 64, 16, stream=raw_stream0)
            buf13 = empty_strided_npu((8, 8, 16), (128, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [permute, unbind, matmul_1], Original ATen: [aten.permute, aten.unbind, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:4
            extern_kernels.bmm(reinterpret_tensor(buf10, (8, 8, 8), (64, 8, 1), 0), reinterpret_tensor(buf12, (8, 8, 16), (128, 16, 1), 0), out=buf13)
            del buf12
        return (reinterpret_tensor(buf13, (2, 4, 8, 16), (512, 128, 16, 1), 0), reinterpret_tensor(primals_1, (2, 4, 8, 16), (1536, 128, 1, 8), 0), reinterpret_tensor(primals_1, (2, 4, 8, 16), (1536, 128, 1, 8), 1024), reinterpret_tensor(primals_1, (2, 4, 16, 8), (1536, 128, 8, 1), 512), buf8, buf10, buf11, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((2, 3, 4, 16, 8), (1536, 512, 128, 8, 1), device='npu:0', dtype=torch.float16)
    return [primals_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
