# AOT ID: ['16_forward']
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


# kernel path: /home/z50063656/tmp/t102-training-boundary-installed-20260915/inductor-cache/tmprrhaujga/eg/cegkiikqedj3uw5fez7bn23p3h76rhs7bijtzlbjytc7xet5xlmv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#    => mul_scalar
# Graph fragment:
#   %primals_2 : Tensor "f32[2, 2, 8, 16][256, 128, 16, 1]npu:0" = PlaceHolder[target=primals_2]
#   %mul_scalar : Tensor "f32[2, 2, 8, 16][256, 128, 16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Scalar](args = (%primals_2, 0.5), kwargs = {})
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
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [512, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 512, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0), x0mask)
        tmp1 = tl.full([1], 0.5, tl.float32)
        tmp2 = tmp0 * tmp1
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t102-training-boundary-installed-20260915/inductor-cache/tmprrhaujga/ep/cep356gkgly4gsajae3jbhjyrm7asgnenukuaycnslrggiajpxcg.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#    => add_tensor, eq_scalar, logical_not_default, view_default_2
# Graph fragment:
#   %bmm_default : Tensor "f32[4, 8, 8][64, 8, 1]npu:0" = PlaceHolder[target=bmm_default]
#   %primals_3 : Tensor "f32[8, 8][8, 1]npu:0" = PlaceHolder[target=primals_3]
#   %view_default_2 : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default, [2, 2, 8, 8]), kwargs = {})
#   %add_tensor : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %primals_3), kwargs = {})
#   %eq_scalar : Tensor "b8[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_tensor, -inf), kwargs = {})
#   %logical_not_default : Tensor "b8[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_scalar,), kwargs = {})
#   return %logical_not_default
triton_unk_fused__safe_softmax_add_matmul_1 = async_compile.triton('triton_unk_fused__safe_softmax_add_matmul_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [64, 4, {'divisors': [1, 64]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 64, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax_add_matmul_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__safe_softmax_add_matmul_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 256
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (4) if (4) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 64
    x1numel : tl.constexpr = 4
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
        tmp1 = tl.load(in_ptr1 + (x0), x0mask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1], float("-inf"), tl.float32)
        tmp4 = tmp2 == tmp3
        tmp5 = tmp4 == 0
        tl.store(out_ptr0 + (x0 + 64*x1), tmp5, x0mask & x1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t102-training-boundary-installed-20260915/inductor-cache/tmprrhaujga/3a/c3acugxgkz4f5qazthdmzrj3wz3piybvpht2n2tfqrtwhlb7azzr.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#    => add_tensor, amax_default, div_tensor, exp_default, full_default, logical_not_default_1, sub_tensor, sum_dim_int_list, view_default_2, where_self
# Graph fragment:
#   %bmm_default : Tensor "f32[4, 8, 8][64, 8, 1]npu:0" = PlaceHolder[target=bmm_default]
#   %primals_3 : Tensor "f32[8, 8][8, 1]npu:0" = PlaceHolder[target=primals_3]
#   %amax_default : Tensor "f32[2, 2, 8, 1][16, 8, 1, 32]npu:0" = PlaceHolder[target=amax_default]
#   %any_dim : Tensor "b8[2, 2, 8, 1][16, 8, 1, 1]npu:0" = PlaceHolder[target=any_dim]
#   %sum_dim_int_list : Tensor "f32[2, 2, 8, 1][16, 8, 1, 32]npu:0" = PlaceHolder[target=sum_dim_int_list]
#   %view_default_2 : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default, [2, 2, 8, 8]), kwargs = {})
#   %add_tensor : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %primals_3), kwargs = {})
#   %amax_default : Tensor "f32[2, 2, 8, 1][16, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_tensor, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_tensor, %amax_default), kwargs = {})
#   %exp_default : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %sum_dim_int_list : Tensor "f32[2, 2, 8, 1][16, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_default, [-1], True), kwargs = {})
#   %div_tensor : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %sum_dim_int_list), kwargs = {})
#   %logical_not_default_1 : Tensor "b8[2, 2, 8, 1][16, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dim,), kwargs = {})
#   %full_default : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 2, 8, 8], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_self : Tensor "f32[2, 2, 8, 8][128, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_1, %full_default, %div_tensor), kwargs = {})
#   return %amax_default,%sum_dim_int_list,%expand_default_2
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

@npu_triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'r0_numel': 8}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [8, 4, {'divisors': [1, 8]}], 'R0_BLOCK_HINT': [8, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 8, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4, 'divisor': 8, 'seed': 2}, {'name': 'r0_2', 'length': 8, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax_add_matmul_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused__safe_softmax_add_matmul_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 32
    r0_numel = 8
    x_g_tile0 : tl.constexpr = (8) if (8) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (4) if (4) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 8
    x1numel : tl.constexpr = 4
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
            tmp0 = tl.load(in_out_ptr0 + (r0_2 + 8*x0 + 64*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=float("-inf"))
            tmp1 = tl.load(in_ptr0 + (r0_2 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=float("-inf"))
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
            tmp6 = tl.load(in_out_ptr0 + (r0_2 + 8*x0 + 64*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=float("-inf"))
            tmp7 = tl.load(in_ptr0 + (r0_2 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=float("-inf"))
            tmp8 = tmp6 + tmp7
            tmp9 = tmp8 - tmp4
            tmp10 = libdevice.exp(tmp9)
            tmp11 = tl.broadcast_to(tmp10, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp13 = _tmp12 + tmp11
            _tmp12 = tmp13
        tmp12 = tl.sum(_tmp12, 2)[:, :, None]
        tmp14 = tl.load(in_ptr1 + (x0 + 8 * x1), x0mask & x1mask, eviction_policy='evict_last') != 0
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp16 = tl.load(in_out_ptr0 + (r0_2 + 8*x0 + 64*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_first', other=0.0)
            tmp17 = tl.load(in_ptr0 + (r0_2 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp15 = tmp14 == 0
            tmp18 = tmp16 + tmp17
            tmp19 = tmp18 - tmp4
            tmp20 = libdevice.exp(tmp19)
            tmp21 = (tmp20 / tmp12)
            tmp22 = tl.full([1, 1], 0.0, tl.float32)
            tmp23 = tl.where(tmp15, tmp22, tmp21)
            tl.store(in_out_ptr0 + (r0_2 + 8*x0 + 64*x1), tmp23, r0_mask & x0mask & x1mask)
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
            buf0 = empty_strided_npu((2, 2, 8, 16), (256, 128, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_mul_0.run(primals_2, buf0, 512, stream=raw_stream0)
            del primals_2
            primals_1 = copy_if_misaligned(primals_1)
            buf1 = empty_strided_npu((2, 2, 16, 8), (256, 128, 1, 16), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_mul_0.run(primals_1, buf1, 512, stream=raw_stream0)
            del primals_1
            buf2 = empty_strided_npu((4, 8, 8), (64, 8, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:1
            extern_kernels.bmm(reinterpret_tensor(buf0, (4, 8, 16), (128, 16, 1), 0), reinterpret_tensor(buf1, (4, 16, 8), (128, 1, 16), 0), out=buf2)
            primals_3 = copy_if_misaligned(primals_3)
            buf5 = empty_strided_npu((2, 2, 8, 8), (128, 64, 8, 1), torch.bool)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.add, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax_add_matmul_1.run(buf2, primals_3, buf5, 256, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.add, aten._safe_softmax]
            # [Provenance debug handles] torch.ops.aten.any.dim:2
            buf6 = torch.ops.aten.any.dim(buf5, -1, True)
            assert_alignment(buf6, 16, 'torch.ops.aten.any.dim')
            del buf5
            buf7 = reinterpret_tensor(buf2, (2, 2, 8, 8), (128, 64, 8, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul, aten.add, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax_add_matmul_2.run(buf7, primals_3, buf6, 32, 8, stream=raw_stream0)
            del buf6
            del primals_3
            primals_4 = copy_if_misaligned(primals_4)
            buf8 = empty_strided_npu((4, 8, 16), (128, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:3
            extern_kernels.bmm(reinterpret_tensor(buf7, (4, 8, 8), (64, 8, 1), 0), reinterpret_tensor(primals_4, (4, 8, 16), (128, 16, 1), 0), out=buf8)
        return (reinterpret_tensor(buf8, (2, 2, 8, 16), (256, 128, 16, 1), 0), primals_4, buf0, buf1, buf7, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((2, 2, 8, 16), (256, 128, 16, 1), device='npu:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 2, 8, 16), (256, 128, 16, 1), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((8, 8), (8, 1), device='npu:0', dtype=torch.float32)
    primals_4 = rand_strided((2, 2, 8, 16), (256, 128, 16, 1), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
