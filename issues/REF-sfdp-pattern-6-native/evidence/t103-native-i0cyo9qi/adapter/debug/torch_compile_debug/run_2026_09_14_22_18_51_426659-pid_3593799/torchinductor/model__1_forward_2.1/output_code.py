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


# kernel path: /home/z50063656/tmp/t103-native-i0cyo9qi/adapter/inductor-cache/tmpaxx4kzbr/hu/chul34wo73dosgps7a7lruccvpm4wkwam3tm6citkatbjnfze32e.py
# Topologically Sorted Source Nodes: [tril], Original ATen: [aten.tril]
# Source node to ATen node mapping:
#   tril => iota, le, sub, unsqueeze, unsqueeze_1
# Graph fragment:
#   %iota : Tensor "i64[16][1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: npu:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -2), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[16, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -1), kwargs = {})
#   %sub : Tensor "i64[16, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %le : Tensor "b8[16, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub, 0), kwargs = {})
#   return %le
triton_unk_fused_tril_0 = async_compile.triton('triton_unk_fused_tril_0', '''
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
    triton_meta={'signature': {'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [16, 16, {'divisors': [1, 16]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 16, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 16, 'divisor': 16, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_tril_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_tril_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 256
    x_g_tile0 : tl.constexpr = (16) if (16) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (16) if (16) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 16
    x1numel : tl.constexpr = 16
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
        tmp0 = (x0 + ((-1)*x1)).to(tl.int32)
        tmp1 = tl.full([1], 0, tl.int32)
        tmp2 = tmp0 <= tmp1
        tl.store(out_ptr0 + (x0 + 16*x1), tmp2, x0mask & x1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t103-native-i0cyo9qi/adapter/inductor-cache/tmpaxx4kzbr/vl/cvlf3izkjjpfy6vc7lrz4jzxcdypieyik3r2tklpqaizmdh46uzr.py
# Topologically Sorted Source Nodes: [ones], Original ATen: [aten.ones]
# Source node to ATen node mapping:
#   ones => full_default
# Graph fragment:
#   %full_default : Tensor "b8[16, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([16, 16], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: npu:0, pin_memory: False})
#   return %full_default
triton_unk_fused_ones_1 = async_compile.triton('triton_unk_fused_ones_1', '''
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
    triton_meta={'signature': {'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [256, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 256, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_ones_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_ones_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 256
    x_g_tile0 : tl.constexpr = (256) if (256) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 256
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
        tmp0 = tl.full([1], True, tl.int1)
        tl.store(out_ptr0 + (x0), tmp0, x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t103-native-i0cyo9qi/adapter/inductor-cache/tmpaxx4kzbr/dd/cddjk2ybvc372hqm5cpjb26pj7xbyvfcuw77pzubcru264gzob4u.py
# Topologically Sorted Source Nodes: [matmul, logical_not, masked_fill, truediv, add, softmax], Original ATen: [aten.matmul, aten.logical_not, aten.masked_fill, aten.div, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   add => add
#   logical_not => logical_not
#   masked_fill => full_default_1, where
#   matmul => view_2
#   softmax => amax, div_1, exp, sub_1, sum_1
#   truediv => div
# Graph fragment:
#   %bmm : Tensor "f32[8, 16, 16][256, 16, 1]npu:0" = PlaceHolder[target=bmm]
#   %logical_and : Tensor "b8[16, 16][16, 1]npu:0" = PlaceHolder[target=logical_and]
#   %amax : Tensor "f32[4, 2, 16, 1][32, 16, 1, 128]npu:0" = PlaceHolder[target=amax]
#   %sum_1 : Tensor "f32[4, 2, 16, 1][32, 16, 1, 128]npu:0" = PlaceHolder[target=sum_1]
#   %view_2 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [4, 2, 16, 16]), kwargs = {})
#   %logical_not : Tensor "b8[16, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and,), kwargs = {})
#   %full_default_1 : Tensor "b8[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "b8[16, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not, %full_default_1, %logical_and), kwargs = {})
#   %div : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_2, 5.656854249492381), kwargs = {})
#   %add : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %where), kwargs = {})
#   %amax : Tensor "f32[4, 2, 16, 1][32, 16, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add, [-1], True), kwargs = {})
#   %sub_1 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %amax), kwargs = {})
#   %exp : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : Tensor "f32[4, 2, 16, 1][32, 16, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   return %amax,%sum_1,%div_1
triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_2 = async_compile.triton('triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'r0_numel': 16}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [16, 8, {'divisors': [1, 16]}], 'R0_BLOCK_HINT': [16, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 16, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 8, 'divisor': 16, 'seed': 2}, {'name': 'r0_2', 'length': 16, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_2(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        _tmp10 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], float("-inf"), tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (r0_2 + 16*x0 + 256*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=float("-inf"))
            tmp3 = tl.load(in_ptr0 + (r0_2 + 16 * x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0) != 0
            tmp1 = tl.full([1, 1], 0.17677669529663687, tl.float32)
            tmp2 = tmp0 * tmp1
            tmp4 = tmp3 == 0
            tmp5 = tl.full([1, 1], True, tl.int1)
            tmp6 = tl.where(tmp4, tmp5, tmp3)
            tmp7 = tmp6.to(tl.float32)
            tmp8 = tmp2 + tmp7
            tmp9 = tl.broadcast_to(tmp8, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp11 = tl.maximum(_tmp10, tmp9)
            _tmp10 = tmp11
        tmp10 = triton_helpers.max2(_tmp10, 2)[:, :, None]
        _tmp24 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp12 = tl.load(in_out_ptr0 + (r0_2 + 16*x0 + 256*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=float("-inf"))
            tmp15 = tl.load(in_ptr0 + (r0_2 + 16 * x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0) != 0
            tmp13 = tl.full([1, 1], 0.17677669529663687, tl.float32)
            tmp14 = tmp12 * tmp13
            tmp16 = tmp15 == 0
            tmp17 = tl.full([1, 1], True, tl.int1)
            tmp18 = tl.where(tmp16, tmp17, tmp15)
            tmp19 = tmp18.to(tl.float32)
            tmp20 = tmp14 + tmp19
            tmp21 = tmp20 - tmp10
            tmp22 = libdevice.exp(tmp21)
            tmp23 = tl.broadcast_to(tmp22, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp25 = _tmp24 + tmp23
            _tmp24 = tmp25
        tmp24 = tl.sum(_tmp24, 2)[:, :, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp26 = tl.load(in_out_ptr0 + (r0_2 + 16*x0 + 256*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_first', other=0.0)
            tmp29 = tl.load(in_ptr0 + (r0_2 + 16 * x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0) != 0
            tmp27 = tl.full([1, 1], 0.17677669529663687, tl.float32)
            tmp28 = tmp26 * tmp27
            tmp30 = tmp29 == 0
            tmp31 = tl.full([1, 1], True, tl.int1)
            tmp32 = tl.where(tmp30, tmp31, tmp29)
            tmp33 = tmp32.to(tl.float32)
            tmp34 = tmp28 + tmp33
            tmp35 = tmp34 - tmp10
            tmp36 = libdevice.exp(tmp35)
            tmp37 = (tmp36 / tmp24)
            tl.store(in_out_ptr0 + (r0_2 + 16*x0 + 256*x1), tmp37, r0_mask & x0mask & x1mask)
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
        primals_1, primals_2, primals_3 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            primals_2 = copy_if_misaligned(primals_2)
            primals_1 = copy_if_misaligned(primals_1)
            buf0 = empty_strided_npu((8, 16, 16), (256, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [transpose, matmul], Original ATen: [aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:1
            extern_kernels.bmm(reinterpret_tensor(primals_2, (8, 16, 32), (512, 32, 1), 0), reinterpret_tensor(primals_1, (8, 32, 16), (512, 1, 32), 0), out=buf0)
            buf1 = empty_strided_npu((16, 16), (16, 1), torch.bool)
            # Topologically Sorted Source Nodes: [tril], Original ATen: [aten.tril]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_tril_0.run(buf1, 256, stream=raw_stream0)
            buf2 = empty_strided_npu((16, 16), (16, 1), torch.bool)
            # Topologically Sorted Source Nodes: [ones], Original ATen: [aten.ones]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_ones_1.run(buf2, 256, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [ones, tril], Original ATen: [aten.ones, aten.tril]
            # [Provenance debug handles] torch.ops.aten.logical_and.default:2
            buf3 = torch.ops.aten.logical_and.default(buf1, buf2)
            assert_alignment(buf3, 16, 'torch.ops.aten.logical_and.default')
            del buf1
            del buf2
            buf6 = reinterpret_tensor(buf0, (4, 2, 16, 16), (512, 256, 16, 1), 0); del buf0  # reuse
            # Topologically Sorted Source Nodes: [matmul, logical_not, masked_fill, truediv, add, softmax], Original ATen: [aten.matmul, aten.logical_not, aten.masked_fill, aten.div, aten.add, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_2.run(buf6, buf3, 128, 16, stream=raw_stream0)
            del buf3
            # Topologically Sorted Source Nodes: [dropout], Original ATen: [aten.native_dropout]
            # [Provenance debug handles] torch.ops.npu._npu_dropout.default:3
            buf7 = torch.ops.npu._npu_dropout.default(buf6, 0.5)
            buf8 = buf7[0]
            assert_alignment(buf8, 16, 'torch.ops.npu._npu_dropout.default')
            buf9 = buf7[1]
            assert_alignment(buf9, 16, 'torch.ops.npu._npu_dropout.default')
            del buf7
            primals_3 = copy_if_misaligned(primals_3)
            buf10 = empty_strided_npu((8, 16, 32), (512, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:4
            extern_kernels.bmm(reinterpret_tensor(buf8, (8, 16, 16), (256, 16, 1), 0), reinterpret_tensor(primals_3, (8, 16, 32), (512, 32, 1), 0), out=buf10)
        return (reinterpret_tensor(buf10, (4, 2, 16, 32), (1024, 512, 32, 1), 0), primals_2, primals_3, reinterpret_tensor(primals_1, (4, 2, 32, 16), (1024, 512, 1, 32), 0), buf6, buf8, buf9, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((4, 2, 16, 32), (1024, 512, 32, 1), device='npu:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 2, 16, 32), (1024, 512, 32, 1), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 2, 16, 32), (1024, 512, 32, 1), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
