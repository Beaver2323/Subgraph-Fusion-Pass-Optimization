# AOT ID: ['23_backward']
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


# kernel path: /home/z50063656/tmp/t102-native-0gliytfd/adapter/inductor-cache/tmpqwr7zmat/wu/cwucsawf2pkq74getoqioxqjhqgdngw2nz5i7dimabye7vjbwuqw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.matmul_backward]
# Source node to ATen node mapping:
#    => matmul_backward_default_1, mul_tensor, mul_tensor_1, sub_tensor_1, sum_dim_int_list_1
# Graph fragment:
#   %getitem_4 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0" = PlaceHolder[target=getitem_4]
#   %where_self : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0" = PlaceHolder[target=where_self]
#   %sum_dim_int_list_1 : Tensor "f32[4, 2, 16, 1][32, 16, 1, 128]npu:0" = PlaceHolder[target=sum_dim_int_list_1]
#   %mul_tensor : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_4, %where_self), kwargs = {})
#   %sum_dim_int_list_1 : Tensor "f32[4, 2, 16, 1][32, 16, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_tensor, [-1], True), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self, %sum_dim_int_list_1), kwargs = {})
#   %sub_tensor_1 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %mul_tensor_1), kwargs = {})
#   %matmul_backward_default_1 : [num_users=2] = call_function[target=torch.ops.aten.matmul_backward.default](args = (%sub_tensor_1, %mul_scalar, %mul_scalar_1, [True, True]), kwargs = {})
#   return %sum_dim_int_list_1,%buf4
triton_unk_fused__softmax_backward_data_matmul_backward_0 = async_compile.triton('triton_unk_fused__softmax_backward_data_matmul_backward_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'r0_numel': 16}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [128, {'divisors': [1]}], 'R0_BLOCK_HINT': [16, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 128, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 16, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_backward_data_matmul_backward_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused__softmax_backward_data_matmul_backward_0(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        _tmp4 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (r0_1 + 16*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp1 = tl.load(in_ptr0 + (r0_1 + 16*x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0)
            tmp2 = tmp0 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [real_block_x0, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tmp5
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp6 = tl.load(in_out_ptr0 + (r0_1 + 16*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp7 = tl.load(in_ptr0 + (r0_1 + 16*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp8 = tmp6 * tmp7
            tmp9 = tmp7 * tmp4
            tmp10 = tmp8 - tmp9
            tl.store(in_out_ptr0 + (r0_1 + 16*x0), tmp10, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t102-native-0gliytfd/adapter/inductor-cache/tmpqwr7zmat/6q/c6ql7j6xtmelapamcdciokxxrebibjzxmplw4vdk3t2nyrm5eupg.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.matmul_backward]
# Source node to ATen node mapping:
#    => matmul_backward_default_1, mul_tensor, mul_tensor_1, sub_tensor_1
# Graph fragment:
#   %mul_scalar_1 : Tensor "f32[4, 2, 32, 16][1024, 512, 1, 32]npu:0" = PlaceHolder[target=mul_scalar_1]
#   %mul_tensor : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_4, %where_self), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self, %sum_dim_int_list_1), kwargs = {})
#   %sub_tensor_1 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %mul_tensor_1), kwargs = {})
#   %matmul_backward_default_1 : [num_users=2] = call_function[target=torch.ops.aten.matmul_backward.default](args = (%sub_tensor_1, %mul_scalar, %mul_scalar_1, [True, True]), kwargs = {})
#   return %buf5
triton_unk_fused__softmax_backward_data_matmul_backward_1 = async_compile.triton('triton_unk_fused__softmax_backward_data_matmul_backward_1', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_backward_data_matmul_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__softmax_backward_data_matmul_backward_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
        tl.store(out_ptr0 + (x2 + 16*y0 + 512*y1), tmp0, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t102-native-0gliytfd/adapter/inductor-cache/tmpqwr7zmat/mc/cmcms4khzrboy3iibjmnlhndcyux4xfecddnkcno5j33fvhamzwi.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#    => mul_scalar_2
# Graph fragment:
#   %getitem_7 : Tensor "f32[4, 2, 32, 16][1024, 512, 16, 1]npu:0" = PlaceHolder[target=getitem_7]
#   %mul_scalar_2 : Tensor "f32[4, 2, 32, 16][1024, 512, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%getitem_7, 0.4167804918837871), kwargs = {})
#   return %mul_scalar_2
triton_unk_fused_mul_2 = async_compile.triton('triton_unk_fused_mul_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_mul_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_mul_2(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (x0), x0mask)
        tmp1 = tl.full([1], 0.4167804918837871, tl.float32)
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
        primals_3, mul_scalar, mul_scalar_1, where_self, tangents_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            tangents_1 = copy_if_misaligned(tangents_1)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul_backward]
            # [Provenance debug handles] torch.ops.aten.matmul_backward.default:11
            buf0 = torch.ops.aten.matmul_backward.default(tangents_1, where_self, primals_3, [True, True])
            del primals_3
            del tangents_1
            buf1 = buf0[0]
            assert_alignment(buf1, 16, 'torch.ops.aten.matmul_backward.default')
            buf2 = buf0[1]
            assert_alignment(buf2, 16, 'torch.ops.aten.matmul_backward.default')
            del buf0
            buf4 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_matmul_backward_0.run(buf4, where_self, 128, 16, stream=raw_stream0)
            del where_self
            buf5 = empty_strided_npu((4, 2, 32, 16), (1024, 512, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_backward_data_matmul_backward_1.run(mul_scalar_1, buf5, 256, 16, stream=raw_stream0)
            del mul_scalar_1
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.matmul_backward]
            # [Provenance debug handles] torch.ops.aten.matmul_backward.default:12
            buf6 = torch.ops.aten.matmul_backward.default(buf4, mul_scalar, buf5, [True, True])
            del buf4
            del buf5
            del mul_scalar
            buf7 = buf6[0]
            assert_alignment(buf7, 16, 'torch.ops.aten.matmul_backward.default')
            buf8 = buf6[1]
            assert_alignment(buf8, 16, 'torch.ops.aten.matmul_backward.default')
            del buf6
            buf9 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_mul_2.run(buf9, 4096, stream=raw_stream0)
            buf10 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_mul_2.run(buf10, 4096, stream=raw_stream0)
        return (reinterpret_tensor(buf9, (4, 2, 16, 32), (1024, 512, 1, 16), 0), buf10, buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_3 = rand_strided((4, 2, 16, 32), (1024, 512, 32, 1), device='npu:0', dtype=torch.float32)
    mul_scalar = rand_strided((4, 2, 16, 32), (1024, 512, 32, 1), device='npu:0', dtype=torch.float32)
    mul_scalar_1 = rand_strided((4, 2, 32, 16), (1024, 512, 1, 32), device='npu:0', dtype=torch.float32)
    where_self = rand_strided((4, 2, 16, 16), (512, 256, 16, 1), device='npu:0', dtype=torch.float32)
    tangents_1 = rand_strided((4, 2, 16, 32), (1024, 512, 32, 1), device='npu:0', dtype=torch.float32)
    return [primals_3, mul_scalar, mul_scalar_1, where_self, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
