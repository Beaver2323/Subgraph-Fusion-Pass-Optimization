# AOT ID: ['14_backward']
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


# kernel path: /home/z50063656/tmp/t102-training-boundary-installed-20260915/inductor-cache/tmpn078z59p/7v/c7vl3ooqnsk6fel6nmhjyukc4nt4t4xr6divxs2pt6woulze7gex.py
# Topologically Sorted Source Nodes: [add, mul, sum_2, mul_1, sub_1, div_2, matmul_backward_1], Original ATen: [aten.add, aten._softmax_backward_data, aten.div, aten.matmul_backward]
# Source node to ATen node mapping:
#   add => add
#   div_2 => div_2
#   matmul_backward_1 => matmul_backward_1
#   mul => mul
#   mul_1 => mul_1
#   sub_1 => sub_1
#   sum_2 => sum_2
# Graph fragment:
#   %tangents_2 : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0" = PlaceHolder[target=tangents_2]
#   %getitem : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0" = PlaceHolder[target=getitem]
#   %div_1 : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0" = PlaceHolder[target=div_1]
#   %sum_2 : Tensor "f32[2, 2, 8, 1][16, 8, 1, 32]npu:0" = PlaceHolder[target=sum_2]
#   %add : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_2, %getitem), kwargs = {})
#   %mul : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %div_1), kwargs = {})
#   %sum_2 : Tensor "f32[2, 2, 8, 1][16, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [-1], True), kwargs = {})
#   %mul_1 : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sum_2), kwargs = {})
#   %sub_1 : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %div_2 : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, 4.0), kwargs = {})
#   %matmul_backward_1 : [num_users=2] = call_function[target=torch.ops.aten.matmul_backward.default](args = (%div_2, %primals_2, %permute, [True, True]), kwargs = {})
#   return %sum_2,%buf4
triton_unk_fused__softmax_backward_data_add_div_matmul_backward_0 = async_compile.triton('triton_unk_fused__softmax_backward_data_add_div_matmul_backward_0', '''
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
    size_hints={'x': 32, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'r0_numel': 8}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [32, {'divisors': [1]}], 'R0_BLOCK_HINT': [8, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 32, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 8, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_backward_data_add_div_matmul_backward_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused__softmax_backward_data_add_div_matmul_backward_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 32
    r0_numel = 8
    x_g_tile0 : tl.constexpr = (32) if (32) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 32
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
        _tmp6 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_out_ptr0 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp3 = tl.load(in_ptr1 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tmp0 + tmp1
            tmp4 = tmp2 * tmp3
            tmp5 = tl.broadcast_to(tmp4, [real_block_x0, R0_BLOCK])
            tmp7 = _tmp6 + tmp5
            _tmp6 = tmp7
        tmp6 = tl.sum(_tmp6, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp8 = tl.load(in_ptr0 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp9 = tl.load(in_out_ptr0 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp11 = tl.load(in_ptr1 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp10 = tmp8 + tmp9
            tmp12 = tmp10 * tmp11
            tmp13 = tmp11 * tmp6
            tmp14 = tmp12 - tmp13
            tmp15 = tl.full([1, 1], 0.25, tl.float32)
            tmp16 = tmp14 * tmp15
            tl.store(in_out_ptr0 + (r0_1 + 8*x0), tmp16, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t102-training-boundary-installed-20260915/inductor-cache/tmpn078z59p/cg/ccgonuo3jgteh5unp6sla3frgsy2j7iw7b3ighcpumjtxdwjqu7l.py
# Topologically Sorted Source Nodes: [add, mul, mul_1, sub_1, div_2, matmul_backward_1], Original ATen: [aten.add, aten._softmax_backward_data, aten.div, aten.matmul_backward]
# Source node to ATen node mapping:
#   add => add
#   div_2 => div_2
#   matmul_backward_1 => matmul_backward_1
#   mul => mul
#   mul_1 => mul_1
#   sub_1 => sub_1
# Graph fragment:
#   %permute : Tensor "f32[2, 2, 16, 8][256, 128, 1, 16]npu:0" = PlaceHolder[target=permute]
#   %add : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_2, %getitem), kwargs = {})
#   %mul : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %div_1), kwargs = {})
#   %mul_1 : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sum_2), kwargs = {})
#   %sub_1 : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %div_2 : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, 4.0), kwargs = {})
#   %matmul_backward_1 : [num_users=2] = call_function[target=torch.ops.aten.matmul_backward.default](args = (%div_2, %primals_2, %permute, [True, True]), kwargs = {})
#   return %buf5
triton_unk_fused__softmax_backward_data_add_div_matmul_backward_1 = async_compile.triton('triton_unk_fused__softmax_backward_data_add_div_matmul_backward_1', '''
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
    size_hints={'y': 64, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'YBLOCK_HINT': [16, 4, {'divisors': [1, 16]}], 'XBLOCK_HINT': [8, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 16, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 4, 'divisor': 16, 'seed': 2}, {'name': 'x2', 'length': 8, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_backward_data_add_div_matmul_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__softmax_backward_data_add_div_matmul_backward_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    ynumel = 64
    xnumel = 8
    y_g_tile0 : tl.constexpr = (16) if (16) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (4) if (4) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 16
    y1numel : tl.constexpr = 4
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
        tl.store(out_ptr0 + (x2 + 8*y0 + 128*y1), tmp0, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t102-training-boundary-installed-20260915/inductor-cache/tmpn078z59p/mb/cmbht3dfqxqe6a3x6hdxpadljzpjxsq7mcvb2hor2l7xwlcgcmpy.py
# Topologically Sorted Source Nodes: [permute_1], Original ATen: [aten.transpose]
# Source node to ATen node mapping:
#   permute_1 => permute_1
# Graph fragment:
#   %getitem_3 : Tensor "f32[2, 2, 16, 8][256, 128, 8, 1]npu:0" = PlaceHolder[target=getitem_3]
#   %permute_1 : Tensor "f32[2, 2, 8, 16][256, 128, 1, 8]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_3, [0, 1, 3, 2]), kwargs = {})
#   return %permute_1
triton_unk_fused_transpose_2 = async_compile.triton('triton_unk_fused_transpose_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [512, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 512, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_transpose_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_transpose_2(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (x0), x0mask)
        tl.store(in_out_ptr0 + (x0), tmp0, x0mask)
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
        primals_2, primals_3, permute, div_1, tangents_1, tangents_2 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            tangents_1 = copy_if_misaligned(tangents_1)
            # Topologically Sorted Source Nodes: [matmul_backward], Original ATen: [aten.matmul_backward]
            # [Provenance debug handles] torch.ops.aten.matmul_backward.default:3
            buf0 = torch.ops.aten.matmul_backward.default(tangents_1, div_1, primals_3, [True, True])
            del primals_3
            del tangents_1
            buf1 = buf0[0]
            assert_alignment(buf1, 16, 'torch.ops.aten.matmul_backward.default')
            buf2 = buf0[1]
            assert_alignment(buf2, 16, 'torch.ops.aten.matmul_backward.default')
            del buf0
            tangents_2 = copy_if_misaligned(tangents_2)
            buf4 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [add, mul, sum_2, mul_1, sub_1, div_2, matmul_backward_1], Original ATen: [aten.add, aten._softmax_backward_data, aten.div, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_add_div_matmul_backward_0.run(buf4, tangents_2, div_1, 32, 8, stream=raw_stream0)
            del div_1
            del tangents_2
            buf5 = empty_strided_npu((2, 2, 16, 8), (256, 128, 8, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add, mul, mul_1, sub_1, div_2, matmul_backward_1], Original ATen: [aten.add, aten._softmax_backward_data, aten.div, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_add_div_matmul_backward_1.run(permute, buf5, 64, 8, stream=raw_stream0)
            del permute
            # Topologically Sorted Source Nodes: [add, mul, mul_1, sub_1, div_2, matmul_backward_1], Original ATen: [aten.add, aten._softmax_backward_data, aten.div, aten.matmul_backward]
            # [Provenance debug handles] torch.ops.aten.matmul_backward.default:4
            buf6 = torch.ops.aten.matmul_backward.default(buf4, primals_2, buf5, [True, True])
            del buf4
            del buf5
            del primals_2
            buf7 = buf6[0]
            assert_alignment(buf7, 16, 'torch.ops.aten.matmul_backward.default')
            buf8 = buf6[1]
            assert_alignment(buf8, 16, 'torch.ops.aten.matmul_backward.default')
            del buf6
            buf9 = reinterpret_tensor(buf8, (2, 2, 8, 16), (256, 128, 1, 8), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [permute_1], Original ATen: [aten.transpose]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_transpose_2.run(buf9, 512, stream=raw_stream0)
        return (buf9, buf7, buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_2 = rand_strided((2, 2, 8, 16), (256, 128, 16, 1), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((2, 2, 8, 16), (256, 128, 16, 1), device='npu:0', dtype=torch.float32)
    permute = rand_strided((2, 2, 16, 8), (256, 128, 1, 16), device='npu:0', dtype=torch.float32)
    div_1 = rand_strided((2, 2, 8, 8), (128, 64, 8, 1), device='npu:0', dtype=torch.float32)
    tangents_1 = rand_strided((2, 2, 8, 16), (256, 128, 16, 1), device='npu:0', dtype=torch.float32)
    tangents_2 = rand_strided((2, 2, 8, 8), (128, 64, 8, 1), device='npu:0', dtype=torch.float32)
    return [primals_2, primals_3, permute, div_1, tangents_1, tangents_2]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
