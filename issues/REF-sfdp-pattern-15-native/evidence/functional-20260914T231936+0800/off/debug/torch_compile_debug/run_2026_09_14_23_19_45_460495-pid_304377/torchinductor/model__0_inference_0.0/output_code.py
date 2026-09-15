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


# kernel path: /home/z50063656/tmp/attention-functional-20260914-unskip/functional-20260914T231936+0800/pattern-15/off/inductor-cache/tmpc20ax0jx/k6/ck674jpirmxolj5h635ld7mjvb67g23yjc4qxg5huoxntqwugl3f.py
# Topologically Sorted Source Nodes: [q, scores], Original ATen: [aten.permute, aten.matmul]
# Source node to ATen node mapping:
#   q => permute
#   scores => clone
# Graph fragment:
#   %arg0_1 : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %permute : Tensor "f16[2, 8, 4, 16][512, 16, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg0_1, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "f16[2, 8, 4, 16][512, 64, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_unk_fused_matmul_permute_0 = async_compile.triton('triton_unk_fused_matmul_permute_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [16, 4, 8, 2, {'divisors': [1, 16, 64, 512]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 16, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 16, 'seed': 2}, {'name': 'x2', 'length': 8, 'divisor': 64, 'seed': 2}, {'name': 'x3', 'length': 2, 'divisor': 512, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        tl.store(out_ptr0 + (x0 + 16*x1 + 64*x2 + 512*x3), tmp0, x0mask & x1mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/attention-functional-20260914-unskip/functional-20260914T231936+0800/pattern-15/off/inductor-cache/tmpc20ax0jx/vy/cvy7lkqhjtituenoxv2hel4ww4vu7wkfg5pj2uinra6f4no2udgo.py
# Topologically Sorted Source Nodes: [k, transpose, scores], Original ATen: [aten.permute, aten.transpose, aten.matmul]
# Source node to ATen node mapping:
#   k => permute_1
#   scores => clone_1
#   transpose => permute_3
# Graph fragment:
#   %arg1_1 : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %permute_1 : Tensor "f16[2, 8, 4, 16][512, 16, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg1_1, [0, 2, 1, 3]), kwargs = {})
#   %permute_3 : Tensor "f16[2, 8, 16, 4][512, 16, 1, 128]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_1 : Tensor "f16[2, 8, 16, 4][512, 64, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_unk_fused_matmul_permute_transpose_1 = async_compile.triton('triton_unk_fused_matmul_permute_transpose_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'YBLOCK_HINT': [128, 2, {'divisors': [1, 128]}], 'XBLOCK_HINT': [4, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 2, 'divisor': 128, 'seed': 2}, {'name': 'x2', 'length': 4, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_transpose_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_transpose_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
        tl.store(out_ptr0 + (x2 + 4*y0 + 512*y1), tmp0, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/attention-functional-20260914-unskip/functional-20260914T231936+0800/pattern-15/off/inductor-cache/tmpc20ax0jx/6k/c6kwztmx6jn7z5qmbzmklxzr2rztcutrmzc3pis6iofomxddd2sp.py
# Topologically Sorted Source Nodes: [eq, view, attn_mask, fill_value, scores, scores_1, masked_fill, softmax], Original ATen: [aten.eq, aten.view, aten.expand, aten.full, aten.matmul, aten.div, aten.masked_fill, aten._softmax]
# Source node to ATen node mapping:
#   attn_mask => expand_2
#   eq => eq
#   fill_value => full_default
#   masked_fill => where
#   scores => view_2
#   scores_1 => div
#   softmax => amax, convert_element_type_2
#   view => view_3
# Graph fragment:
#   %arg4_1 : Tensor "f16[2, 4][4, 1]npu:0" = PlaceHolder[target=arg4_1]
#   %bmm : Tensor "f16[16, 4, 4][16, 4, 1]npu:0" = PlaceHolder[target=bmm]
#   %arg3_1 : Tensor "f16[][]npu:0" = PlaceHolder[target=arg3_1]
#   %eq : Tensor "b8[2, 4][4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg4_1, 0), kwargs = {})
#   %view_3 : Tensor "b8[2, 1, 1, 4][4, 4, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%eq, [2, 1, 1, 4]), kwargs = {})
#   %expand_2 : Tensor "b8[2, 8, 4, 4][4, 0, 0, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_3, [2, 8, 4, 4]), kwargs = {})
#   %full_default : Tensor "f16[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: npu:0, pin_memory: False})
#   %view_2 : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2, 8, 4, 4]), kwargs = {})
#   %div : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_2, %arg3_1), kwargs = {})
#   %where : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand_2, %full_default, %div), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %amax : Tensor "f32[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_2, [-1], True), kwargs = {})
#   return %amax
triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_2 = async_compile.triton('triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [32, 2, {'divisors': [1, 32]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 32, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 2, 'divisor': 32, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        _es_lane0 = tl.arange(0, 4)[None, :]
        _es_full0 = tl.load(in_ptr0 + 4 * x1 + _es_lane0, x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp0 = extract_slice(_es_full0, [0, 0], [real_block_x1, 1], [1, 1])
        _es_slane1 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask1 = (4*x0offset + _es_slane1) < 4*x0numel
        _es_sfull1 = tl.load(in_ptr1 + (128 * x1 + 4 * x0offset + _es_slane1), _es_smask1 & x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = extract_slice(_es_sfull1, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        tmp4 = tl.load(in_ptr2 + (0)).to(tl.float32)
        tmp5 = tmp4
        _es_lane2 = tl.arange(0, 4)[None, :]
        _es_full2 = tl.load(in_ptr0 + (1 + 4 * x1) + _es_lane2, x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp10 = extract_slice(_es_full2, [0, 0], [real_block_x1, 1], [1, 1])
        _es_slane3 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask3 = (4*x0offset + _es_slane3) < 4*x0numel
        _es_sfull3 = tl.load(in_ptr1 + (1 + 128 * x1 + 4 * x0offset + _es_slane3), _es_smask3 & x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = extract_slice(_es_sfull3, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_lane4 = tl.arange(0, 4)[None, :]
        _es_full4 = tl.load(in_ptr0 + (2 + 4 * x1) + _es_lane4, x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp17 = extract_slice(_es_full4, [0, 0], [real_block_x1, 1], [1, 1])
        _es_slane5 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask5 = (4*x0offset + _es_slane5) < 4*x0numel
        _es_sfull5 = tl.load(in_ptr1 + (2 + 128 * x1 + 4 * x0offset + _es_slane5), _es_smask5 & x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp19 = extract_slice(_es_sfull5, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_lane6 = tl.arange(0, 4)[None, :]
        _es_full6 = tl.load(in_ptr0 + (3 + 4 * x1) + _es_lane6, x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp24 = extract_slice(_es_full6, [0, 0], [real_block_x1, 1], [1, 1])
        _es_slane7 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask7 = (4*x0offset + _es_slane7) < 4*x0numel
        _es_sfull7 = tl.load(in_ptr1 + (3 + 128 * x1 + 4 * x0offset + _es_slane7), _es_smask7 & x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp26 = extract_slice(_es_sfull7, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        tmp1 = tl.full([1], 0.0, tl.float32)
        tmp2 = tmp0 == tmp1
        tmp6 = (tmp3 / tmp5)
        tmp7 = tl.full([1], float("-inf"), tl.float32)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp10 == tmp1
        tmp13 = (tmp12 / tmp5)
        tmp14 = tl.where(tmp11, tmp7, tmp13)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.maximum(tmp9, tmp15)
        tmp18 = tmp17 == tmp1
        tmp20 = (tmp19 / tmp5)
        tmp21 = tl.where(tmp18, tmp7, tmp20)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl.maximum(tmp16, tmp22)
        tmp25 = tmp24 == tmp1
        tmp27 = (tmp26 / tmp5)
        tmp28 = tl.where(tmp25, tmp7, tmp27)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tl.maximum(tmp23, tmp29)
        tl.store(out_ptr0 + (x0 + 32*x1), tmp30, x0mask & x1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/attention-functional-20260914-unskip/functional-20260914T231936+0800/pattern-15/off/inductor-cache/tmpc20ax0jx/t5/ct5urmfcjinr5arsvyvdtbqo2xlk7zezkiyfhpil6g6fndx42vuw.py
# Topologically Sorted Source Nodes: [eq, view, attn_mask, fill_value, scores, scores_1, masked_fill, softmax], Original ATen: [aten.eq, aten.view, aten.expand, aten.full, aten.matmul, aten.div, aten.masked_fill, aten._softmax]
# Source node to ATen node mapping:
#   attn_mask => expand_2
#   eq => eq
#   fill_value => full_default
#   masked_fill => where
#   scores => view_2
#   scores_1 => div
#   softmax => convert_element_type_2, sub
#   view => view_3
# Graph fragment:
#   %amax : Tensor "f32[2, 8, 4, 1][32, 4, 1, 64]npu:0" = PlaceHolder[target=amax]
#   %eq : Tensor "b8[2, 4][4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg4_1, 0), kwargs = {})
#   %view_3 : Tensor "b8[2, 1, 1, 4][4, 4, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%eq, [2, 1, 1, 4]), kwargs = {})
#   %expand_2 : Tensor "b8[2, 8, 4, 4][4, 0, 0, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_3, [2, 8, 4, 4]), kwargs = {})
#   %full_default : Tensor "f16[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: npu:0, pin_memory: False})
#   %view_2 : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2, 8, 4, 4]), kwargs = {})
#   %div : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_2, %arg3_1), kwargs = {})
#   %where : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand_2, %full_default, %div), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %sub : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %amax), kwargs = {})
#   return %buf4
triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_3 = async_compile.triton('triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /home/z50063656/tmp/attention-functional-20260914-unskip/functional-20260914T231936+0800/pattern-15/off/inductor-cache/tmpc20ax0jx/jb/cjbiyafcv74xd6iqa3j4ndhbeg3alq24dgxxahbwo7lf7zzxt3ej.py
# Topologically Sorted Source Nodes: [eq, view, attn_mask, fill_value, scores, scores_1, masked_fill, softmax], Original ATen: [aten.eq, aten.view, aten.expand, aten.full, aten.matmul, aten.div, aten.masked_fill, aten._softmax]
# Source node to ATen node mapping:
#   attn_mask => expand_2
#   eq => eq
#   fill_value => full_default
#   masked_fill => where
#   scores => view_2
#   scores_1 => div
#   softmax => convert_element_type_2, exp, sub, sum_1
#   view => view_3
# Graph fragment:
#   %arg4_1 : Tensor "f16[2, 4][4, 1]npu:0" = PlaceHolder[target=arg4_1]
#   %bmm : Tensor "f16[16, 4, 4][16, 4, 1]npu:0" = PlaceHolder[target=bmm]
#   %arg3_1 : Tensor "f16[][]npu:0" = PlaceHolder[target=arg3_1]
#   %buf4 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0" = PlaceHolder[target=buf4]
#   %eq : Tensor "b8[2, 4][4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg4_1, 0), kwargs = {})
#   %view_3 : Tensor "b8[2, 1, 1, 4][4, 4, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%eq, [2, 1, 1, 4]), kwargs = {})
#   %expand_2 : Tensor "b8[2, 8, 4, 4][4, 0, 0, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_3, [2, 8, 4, 4]), kwargs = {})
#   %full_default : Tensor "f16[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: npu:0, pin_memory: False})
#   %view_2 : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2, 8, 4, 4]), kwargs = {})
#   %div : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_2, %arg3_1), kwargs = {})
#   %where : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand_2, %full_default, %div), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %sub : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %amax), kwargs = {})
#   %exp : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : Tensor "f32[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   return %sum_1
triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_4 = async_compile.triton('triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [32, 2, {'divisors': [1, 32]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 32, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 2, 'divisor': 32, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        _es_lane0 = tl.arange(0, 4)[None, :]
        _es_full0 = tl.load(in_ptr0 + 4 * x1 + _es_lane0, x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp0 = extract_slice(_es_full0, [0, 0], [real_block_x1, 1], [1, 1])
        _es_slane1 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask1 = (4*x0offset + _es_slane1) < 4*x0numel
        _es_sfull1 = tl.load(in_ptr1 + (128 * x1 + 4 * x0offset + _es_slane1), _es_smask1 & x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = extract_slice(_es_sfull1, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        tmp4 = tl.load(in_ptr2 + (0)).to(tl.float32)
        tmp5 = tmp4
        _es_slane2 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask2 = (4*x0offset + _es_slane2) < 4*x0numel
        _es_sfull2 = tl.load(in_ptr3 + (128 * x1 + 4 * x0offset + _es_slane2), _es_smask2 & x1mask, eviction_policy='evict_last')
        tmp10 = extract_slice(_es_sfull2, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_lane3 = tl.arange(0, 4)[None, :]
        _es_full3 = tl.load(in_ptr0 + (1 + 4 * x1) + _es_lane3, x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp13 = extract_slice(_es_full3, [0, 0], [real_block_x1, 1], [1, 1])
        _es_slane4 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask4 = (4*x0offset + _es_slane4) < 4*x0numel
        _es_sfull4 = tl.load(in_ptr1 + (1 + 128 * x1 + 4 * x0offset + _es_slane4), _es_smask4 & x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp15 = extract_slice(_es_sfull4, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_slane5 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask5 = (4*x0offset + _es_slane5) < 4*x0numel
        _es_sfull5 = tl.load(in_ptr3 + (1 + 128 * x1 + 4 * x0offset + _es_slane5), _es_smask5 & x1mask, eviction_policy='evict_last')
        tmp19 = extract_slice(_es_sfull5, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_lane6 = tl.arange(0, 4)[None, :]
        _es_full6 = tl.load(in_ptr0 + (2 + 4 * x1) + _es_lane6, x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp23 = extract_slice(_es_full6, [0, 0], [real_block_x1, 1], [1, 1])
        _es_slane7 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask7 = (4*x0offset + _es_slane7) < 4*x0numel
        _es_sfull7 = tl.load(in_ptr1 + (2 + 128 * x1 + 4 * x0offset + _es_slane7), _es_smask7 & x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp25 = extract_slice(_es_sfull7, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_slane8 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask8 = (4*x0offset + _es_slane8) < 4*x0numel
        _es_sfull8 = tl.load(in_ptr3 + (2 + 128 * x1 + 4 * x0offset + _es_slane8), _es_smask8 & x1mask, eviction_policy='evict_last')
        tmp29 = extract_slice(_es_sfull8, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_lane9 = tl.arange(0, 4)[None, :]
        _es_full9 = tl.load(in_ptr0 + (3 + 4 * x1) + _es_lane9, x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = extract_slice(_es_full9, [0, 0], [real_block_x1, 1], [1, 1])
        _es_slane10 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask10 = (4*x0offset + _es_slane10) < 4*x0numel
        _es_sfull10 = tl.load(in_ptr1 + (3 + 128 * x1 + 4 * x0offset + _es_slane10), _es_smask10 & x1mask, eviction_policy='evict_last').to(tl.float32)
        tmp35 = extract_slice(_es_sfull10, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        _es_slane11 = tl.arange(0, 4*real_block_x0)[None, :]
        _es_smask11 = (4*x0offset + _es_slane11) < 4*x0numel
        _es_sfull11 = tl.load(in_ptr3 + (3 + 128 * x1 + 4 * x0offset + _es_slane11), _es_smask11 & x1mask, eviction_policy='evict_last')
        tmp39 = extract_slice(_es_sfull11, [0, 0], [real_block_x1, real_block_x0], [1, 4])
        tmp1 = tl.full([1], 0.0, tl.float32)
        tmp2 = tmp0 == tmp1
        tmp6 = (tmp3 / tmp5)
        tmp7 = tl.full([1], float("-inf"), tl.float32)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp12 = libdevice.exp(tmp11)
        tmp14 = tmp13 == tmp1
        tmp16 = (tmp15 / tmp5)
        tmp17 = tl.where(tmp14, tmp7, tmp16)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp21 = libdevice.exp(tmp20)
        tmp22 = tmp12 + tmp21
        tmp24 = tmp23 == tmp1
        tmp26 = (tmp25 / tmp5)
        tmp27 = tl.where(tmp24, tmp7, tmp26)
        tmp28 = tmp27.to(tl.float32)
        tmp30 = tmp28 - tmp29
        tmp31 = libdevice.exp(tmp30)
        tmp32 = tmp22 + tmp31
        tmp34 = tmp33 == tmp1
        tmp36 = (tmp35 / tmp5)
        tmp37 = tl.where(tmp34, tmp7, tmp36)
        tmp38 = tmp37.to(tl.float32)
        tmp40 = tmp38 - tmp39
        tmp41 = libdevice.exp(tmp40)
        tmp42 = tmp32 + tmp41
        tl.store(out_ptr0 + (x0 + 32*x1), tmp42, x0mask & x1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/attention-functional-20260914-unskip/functional-20260914T231936+0800/pattern-15/off/inductor-cache/tmpc20ax0jx/nx/cnxifor6c4fpk2bcodylxbabihov2je6untuvd2egtsxclmlkr4z.py
# Topologically Sorted Source Nodes: [eq, view, attn_mask, fill_value, scores, scores_1, masked_fill, softmax], Original ATen: [aten.eq, aten.view, aten.expand, aten.full, aten.matmul, aten.div, aten.masked_fill, aten._softmax]
# Source node to ATen node mapping:
#   attn_mask => expand_2
#   eq => eq
#   fill_value => full_default
#   masked_fill => where
#   scores => view_2
#   scores_1 => div
#   softmax => convert_element_type_2, convert_element_type_3, div_1, exp, sub
#   view => view_3
# Graph fragment:
#   %arg4_1 : Tensor "f16[2, 4][4, 1]npu:0" = PlaceHolder[target=arg4_1]
#   %bmm : Tensor "f16[16, 4, 4][16, 4, 1]npu:0" = PlaceHolder[target=bmm]
#   %arg3_1 : Tensor "f16[][]npu:0" = PlaceHolder[target=arg3_1]
#   %buf4 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0" = PlaceHolder[target=buf4]
#   %buf6 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0" = PlaceHolder[target=buf6]
#   %eq : Tensor "b8[2, 4][4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg4_1, 0), kwargs = {})
#   %view_3 : Tensor "b8[2, 1, 1, 4][4, 4, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%eq, [2, 1, 1, 4]), kwargs = {})
#   %expand_2 : Tensor "b8[2, 8, 4, 4][4, 0, 0, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_3, [2, 8, 4, 4]), kwargs = {})
#   %full_default : Tensor "f16[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: npu:0, pin_memory: False})
#   %view_2 : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2, 8, 4, 4]), kwargs = {})
#   %div : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_2, %arg3_1), kwargs = {})
#   %where : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand_2, %full_default, %div), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %sub : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %amax), kwargs = {})
#   %exp : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %div_1 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %convert_element_type_3 : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.float16), kwargs = {})
#   return %expand_3
triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_5 = async_compile.triton('triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [4, 32, 2, {'divisors': [1, 4, 128]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 32, 'divisor': 4, 'seed': 2}, {'name': 'x2', 'length': 2, 'divisor': 128, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + 4*x2), x0mask & x2mask, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_out_ptr0 + (x0 + 4*x1 + 128*x2), x0mask & x1mask & x2mask).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (0)).to(tl.float32)
        tmp5 = tmp4
        tmp10 = tl.load(in_ptr2 + (x0 + 4*x1 + 128*x2), x0mask & x1mask & x2mask)
        tmp13 = tl.load(in_ptr3 + (x0 + 4*x1 + 128*x2), x0mask & x1mask & x2mask)
        tmp1 = tl.full([1], 0.0, tl.float32)
        tmp2 = tmp0 == tmp1
        tmp6 = (tmp3 / tmp5)
        tmp7 = tl.full([1], float("-inf"), tl.float32)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp12 = libdevice.exp(tmp11)
        tmp14 = (tmp12 / tmp13)
        tmp15 = tmp14.to(tl.float32)
        tl.store(in_out_ptr0 + (x0 + 4*x1 + 128*x2), tmp15, x0mask & x1mask & x2mask)
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
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_npu((2, 8, 4, 16), (512, 64, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [q, scores], Original ATen: [aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_0.run(arg0_1, buf0, 1024, stream=raw_stream0)
            del arg0_1
            arg1_1 = copy_if_misaligned(arg1_1)
            buf1 = empty_strided_npu((2, 8, 16, 4), (512, 64, 4, 1), torch.float16)
            # Topologically Sorted Source Nodes: [k, transpose, scores], Original ATen: [aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_transpose_1.run(arg1_1, buf1, 256, 4, stream=raw_stream0)
            del arg1_1
            buf2 = empty_strided_npu((16, 4, 4), (16, 4, 1), torch.float16)
            # Topologically Sorted Source Nodes: [q, scores, k, transpose], Original ATen: [aten.permute, aten.matmul, aten.transpose]
            # [Provenance debug handles] extern_kernels.bmm:1
            extern_kernels.bmm(reinterpret_tensor(buf0, (16, 4, 16), (64, 16, 1), 0), reinterpret_tensor(buf1, (16, 16, 4), (64, 4, 1), 0), out=buf2)
            del buf0
            del buf1
            arg4_1 = copy_if_misaligned(arg4_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            buf3 = empty_strided_npu((2, 8, 4, 1), (32, 4, 1, 64), torch.float32)
            # Topologically Sorted Source Nodes: [eq, view, attn_mask, fill_value, scores, scores_1, masked_fill, softmax], Original ATen: [aten.eq, aten.view, aten.expand, aten.full, aten.matmul, aten.div, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_2.run(arg4_1, buf2, arg3_1, buf3, 64, stream=raw_stream0)
            buf4 = empty_strided_npu((2, 8, 4, 4), (128, 16, 4, 1), torch.float32)
            # Topologically Sorted Source Nodes: [eq, view, attn_mask, fill_value, scores, scores_1, masked_fill, softmax], Original ATen: [aten.eq, aten.view, aten.expand, aten.full, aten.matmul, aten.div, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_3.run(buf3, buf4, 256, stream=raw_stream0)
            del buf3
            buf5 = empty_strided_npu((2, 8, 4, 1), (32, 4, 1, 64), torch.float32)
            # Topologically Sorted Source Nodes: [eq, view, attn_mask, fill_value, scores, scores_1, masked_fill, softmax], Original ATen: [aten.eq, aten.view, aten.expand, aten.full, aten.matmul, aten.div, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_4.run(arg4_1, buf2, arg3_1, buf4, buf5, 64, stream=raw_stream0)
            buf6 = empty_strided_npu((2, 8, 4, 4), (128, 16, 4, 1), torch.float32)
            # Topologically Sorted Source Nodes: [eq, view, attn_mask, fill_value, scores, scores_1, masked_fill, softmax], Original ATen: [aten.eq, aten.view, aten.expand, aten.full, aten.matmul, aten.div, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_3.run(buf5, buf6, 256, stream=raw_stream0)
            del buf5
            buf7 = reinterpret_tensor(buf2, (2, 8, 4, 4), (128, 16, 4, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [eq, view, attn_mask, fill_value, scores, scores_1, masked_fill, softmax], Original ATen: [aten.eq, aten.view, aten.expand, aten.full, aten.matmul, aten.div, aten.masked_fill, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_div_eq_expand_full_masked_fill_matmul_view_5.run(buf7, arg4_1, arg3_1, buf4, buf6, 256, stream=raw_stream0)
            del arg3_1
            del arg4_1
            del buf4
            del buf6
            arg2_1 = copy_if_misaligned(arg2_1)
            buf8 = empty_strided_npu((2, 8, 4, 16), (512, 64, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [v, matmul_1], Original ATen: [aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_0.run(arg2_1, buf8, 1024, stream=raw_stream0)
            del arg2_1
            buf9 = empty_strided_npu((16, 4, 16), (64, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [eq, view, attn_mask, fill_value, scores, scores_1, masked_fill, softmax, matmul_1, v], Original ATen: [aten.eq, aten.view, aten.expand, aten.full, aten.matmul, aten.div, aten.masked_fill, aten._softmax, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:2
            extern_kernels.bmm(reinterpret_tensor(buf7, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf8, (16, 4, 16), (64, 16, 1), 0), out=buf9)
            del buf7
            del buf8
        return (reinterpret_tensor(buf9, (2, 8, 4, 16), (512, 64, 16, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float16)
    arg1_1 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float16)
    arg2_1 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float16)
    arg3_1 = rand_strided((), (), device='npu:0', dtype=torch.float16)
    arg4_1 = rand_strided((2, 4), (4, 1), device='npu:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
