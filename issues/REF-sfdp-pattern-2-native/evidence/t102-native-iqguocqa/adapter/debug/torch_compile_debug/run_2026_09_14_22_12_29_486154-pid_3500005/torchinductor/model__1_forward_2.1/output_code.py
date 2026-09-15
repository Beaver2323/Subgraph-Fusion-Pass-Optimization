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


# kernel path: /home/z50063656/tmp/t102-native-iqguocqa/adapter/inductor-cache/tmpqulijjzp/hl/chllglssfdcwbgrzvyk4ge7fag6oj4wj766hqbvlrtgrreaeadh4.py
# Topologically Sorted Source Nodes: [matmul, ], Original ATen: [aten.matmul, aten.mul, aten.isfinite, aten.all]
# Source node to ATen node mapping:
#    => abs_default, eq_tensor, logical_not_default, mul_tensor, mul_tensor_3, ne_scalar
#   matmul => view_2
# Graph fragment:
#   %bmm : Tensor "f32[8, 16, 16][256, 16, 1]npu:0" = PlaceHolder[target=bmm]
#   %view_2 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [4, 2, 16, 16]), kwargs = {})
#   %mul_tensor : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, 0.17677669529663687), kwargs = {})
#   %eq_tensor : Tensor "b8[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%mul_tensor, %mul_tensor), kwargs = {})
#   %abs_default : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_tensor,), kwargs = {})
#   %ne_scalar : Tensor "b8[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%abs_default, inf), kwargs = {})
#   %mul_tensor_3 : Tensor "b8[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_tensor, %ne_scalar), kwargs = {})
#   %logical_not_default : Tensor "b8[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%mul_tensor_3,), kwargs = {})
#   return %logical_not_default
triton_unk_fused_all_isfinite_matmul_mul_0 = async_compile.triton('triton_unk_fused_all_isfinite_matmul_mul_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [2048, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 2048, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_all_isfinite_matmul_mul_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_all_isfinite_matmul_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 2048
    x_g_tile0 : tl.constexpr = (2048) if (2048) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 2048
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
        tmp1 = tl.full([1], 0.17677669529663687, tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tmp2 == tmp2
        tmp4 = tl_math.abs(tmp2)
        tmp5 = tl.full([1], float("inf"), tl.float32)
        tmp6 = tmp4 != tmp5
        tmp7 = tmp3 & tmp6
        tmp8 = tmp7 == 0
        tl.store(out_ptr0 + (x0), tmp8, x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t102-native-iqguocqa/adapter/inductor-cache/tmpqulijjzp/2p/c2pcjjn3rjhs6szdpumijgaeg5smhuyybmwewrhdkvvenm6qdcqx.py
# Topologically Sorted Source Nodes: [matmul, , softmax], Original ATen: [aten.matmul, aten.mul, aten.amax, aten.sub, aten.all, aten._softmax]
# Source node to ATen node mapping:
#    => amax_default, amax_default_1, logical_not_default_1, mul_tensor, mul_tensor_1, mul_tensor_2, sub_tensor, sub_tensor_1, where_self
#   matmul => view_2
#   softmax => div, exp, sum_1
# Graph fragment:
#   %bmm : Tensor "f32[8, 16, 16][256, 16, 1]npu:0" = PlaceHolder[target=bmm]
#   %any_dims : Tensor "b8[4, 2, 16, 1][32, 16, 1, 1]npu:0" = PlaceHolder[target=any_dims]
#   %amax_default : Tensor "f32[4, 2, 16, 1][32, 16, 1, 128]npu:0" = PlaceHolder[target=amax_default]
#   %amax_default_1 : Tensor "f32[4, 2, 16, 1][32, 16, 1, 128]npu:0" = PlaceHolder[target=amax_default_1]
#   %sum_1 : Tensor "f32[4, 2, 16, 1][32, 16, 1, 128]npu:0" = PlaceHolder[target=sum_1]
#   %view_2 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [4, 2, 16, 16]), kwargs = {})
#   %mul_tensor : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, 0.17677669529663687), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, 1), kwargs = {})
#   %amax_default : Tensor "f32[4, 2, 16, 1][32, 16, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_1, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %amax_default), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor, 0.17677669529663687), kwargs = {})
#   %amax_default_1 : Tensor "f32[4, 2, 16, 1][32, 16, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [-1], True), kwargs = {})
#   %sub_tensor_1 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default_1), kwargs = {})
#   %logical_not_default_1 : Tensor "b8[4, 2, 16, 1][32, 16, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dims,), kwargs = {})
#   %where_self : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_1, %mul_tensor_2, %sub_tensor_1), kwargs = {})
#   %exp : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%where_self,), kwargs = {})
#   %sum_1 : Tensor "f32[4, 2, 16, 1][32, 16, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   return %amax_default,%amax_default_1,%sum_1,%expand_2
triton_unk_fused__softmax_all_amax_matmul_mul_sub_1 = async_compile.triton('triton_unk_fused__softmax_all_amax_matmul_mul_sub_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'r0_numel': 16}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [16, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 16, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_all_amax_matmul_mul_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused__softmax_all_amax_matmul_mul_sub_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 128
    r0_numel = 16
    x_g_tile0 : tl.constexpr = (128) if (128) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 128
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
        _tmp4 = tl.full([real_block_x0, R0_BLOCK], float("-inf"), tl.float32)
        _tmp9 = tl.full([real_block_x0, R0_BLOCK], float("-inf"), tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (r0_1 + 16*x0), r0_mask & x0mask, eviction_policy='evict_last', other=float("-inf"))
            tmp1 = tl.full([1, 1], 1.0, tl.float32)
            tmp2 = tmp0 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [real_block_x0, R0_BLOCK])
            tmp5 = tl.maximum(_tmp4, tmp3)
            _tmp4 = tmp5
            tmp6 = tl.full([1, 1], 0.17677669529663687, tl.float32)
            tmp7 = tmp0 * tmp6
            tmp8 = tl.broadcast_to(tmp7, [real_block_x0, R0_BLOCK])
            tmp10 = tl.maximum(_tmp9, tmp8)
            _tmp9 = tmp10
        tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
        tmp9 = triton_helpers.max2(_tmp9, 1)[:, None]
        tmp11 = tl.load(in_ptr0 + x0, x0mask, eviction_policy='evict_last') != 0
        _tmp24 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp13 = tl.load(in_out_ptr0 + (r0_1 + 16*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp12 = tmp11 == 0
            tmp14 = tl.full([1, 1], 1.0, tl.float32)
            tmp15 = tmp13 * tmp14
            tmp16 = tmp15 - tmp4
            tmp17 = tl.full([1, 1], 0.17677669529663687, tl.float32)
            tmp18 = tmp16 * tmp17
            tmp19 = tmp13 * tmp17
            tmp20 = tmp19 - tmp9
            tmp21 = tl.where(tmp12, tmp18, tmp20)
            tmp22 = libdevice.exp(tmp21)
            tmp23 = tl.broadcast_to(tmp22, [real_block_x0, R0_BLOCK])
            tmp25 = _tmp24 + tmp23
            _tmp24 = tl.where(r0_mask & xmask, tmp25, _tmp24)
        tmp24 = tl.sum(_tmp24, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp27 = tl.load(in_out_ptr0 + (r0_1 + 16*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp26 = tmp11 == 0
            tmp28 = tl.full([1, 1], 1.0, tl.float32)
            tmp29 = tmp27 * tmp28
            tmp30 = tmp29 - tmp4
            tmp31 = tl.full([1, 1], 0.17677669529663687, tl.float32)
            tmp32 = tmp30 * tmp31
            tmp33 = tmp27 * tmp31
            tmp34 = tmp33 - tmp9
            tmp35 = tl.where(tmp26, tmp32, tmp34)
            tmp36 = libdevice.exp(tmp35)
            tmp37 = (tmp36 / tmp24)
            tl.store(in_out_ptr0 + (r0_1 + 16*x0), tmp37, r0_mask & x0mask)
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
            buf3 = empty_strided_npu((4, 2, 16, 16), (512, 256, 16, 1), torch.bool)
            # Topologically Sorted Source Nodes: [matmul, ], Original ATen: [aten.matmul, aten.mul, aten.isfinite, aten.all]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_all_isfinite_matmul_mul_0.run(buf0, buf3, 2048, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [matmul, ], Original ATen: [aten.matmul, aten.mul, aten.isfinite, aten.all]
            # [Provenance debug handles] torch.ops.aten.any.dims:2
            buf4 = torch.ops.aten.any.dims(buf3, [-1], True)
            assert_alignment(buf4, 16, 'torch.ops.aten.any.dims')
            del buf3
            buf6 = reinterpret_tensor(buf0, (4, 2, 16, 16), (512, 256, 16, 1), 0); del buf0  # reuse
            # Topologically Sorted Source Nodes: [matmul, , softmax], Original ATen: [aten.matmul, aten.mul, aten.amax, aten.sub, aten.all, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_all_amax_matmul_mul_sub_1.run(buf6, buf4, 128, 16, stream=raw_stream0)
            del buf4
            primals_3 = copy_if_misaligned(primals_3)
            buf7 = empty_strided_npu((8, 16, 32), (512, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:3
            extern_kernels.bmm(reinterpret_tensor(buf6, (8, 16, 16), (256, 16, 1), 0), reinterpret_tensor(primals_3, (8, 16, 32), (512, 32, 1), 0), out=buf7)
        return (reinterpret_tensor(buf7, (4, 2, 16, 32), (1024, 512, 32, 1), 0), primals_2, primals_3, reinterpret_tensor(primals_1, (4, 2, 32, 16), (1024, 512, 1, 32), 0), buf6, )

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
