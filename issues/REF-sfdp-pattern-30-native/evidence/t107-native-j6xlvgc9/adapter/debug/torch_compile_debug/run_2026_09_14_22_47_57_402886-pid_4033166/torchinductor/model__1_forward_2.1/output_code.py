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


# kernel path: /home/z50063656/tmp/t107-native-j6xlvgc9/adapter/inductor-cache/tmpn7rsjodp/cv/ccvg224hwbcfnzwihocbn27qvvz2qisztvle3gu5e6i5wcxyjqro.py
# Topologically Sorted Source Nodes: [permute_1, mul_scalar, matmul], Original ATen: [aten.permute, aten.mul, aten.matmul]
# Source node to ATen node mapping:
#   matmul => clone
#   mul_scalar => mul
#   permute_1 => permute_1
# Graph fragment:
#   %primals_2 : Tensor "f16[2, 8, 4, 16][512, 64, 16, 1]npu:0" = PlaceHolder[target=primals_2]
#   %expand : Tensor "f16[2, 4, 8, 16][512, 16, 64, 1]npu:0" = PlaceHolder[target=expand]
#   %permute_1 : Tensor "f16[2, 4, 8, 16][512, 16, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_2, [0, 2, 1, 3]), kwargs = {})
#   %mul : Tensor "f16[2, 4, 8, 16][512, 16, 64, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_1, 0.25), kwargs = {})
#   %clone : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %expand,%clone
triton_unk_fused_matmul_mul_permute_0 = async_compile.triton('triton_unk_fused_matmul_mul_permute_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [2, 64, 8, {'divisors': [512, 1, 64]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x2', 'length': 2, 'divisor': 512, 'seed': 2}, {'name': 'x3', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x4', 'length': 8, 'divisor': 64, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_mul_permute_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_mul_permute_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 1024
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (8) if (8) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (2) if (2) < (x_g_rem2) else (x_g_rem2)
    x2numel : tl.constexpr = 2
    x3numel : tl.constexpr = 64
    x4numel : tl.constexpr = 8
    real_block_x2 : tl.constexpr = x_g_tile2
    real_block_x3 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x4 : tl.constexpr = x_g_tile1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x4_blocks : tl.constexpr = (x4numel + real_block_x4 - 1) // real_block_x4
    x_cumblk_1 = x2_blocks
    x_cumblk_2 = x2_blocks * x3_blocks
    total_blocks = x2_blocks * x3_blocks * x4_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x2offset = (group_base + i) % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_1 % x3_blocks * real_block_x3
        x4offset = (group_base + i) // x_cumblk_2 % x4_blocks * real_block_x4
        x2index = x2offset + tl.arange(0, real_block_x2)[:, None, None]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[None, None, :]
        x3 = x3index
        x3mask = x3index < x3numel
        x4index = x4offset + tl.arange(0, real_block_x4)[None, :, None]
        x4 = x4index
        x4mask = x4index < x4numel
        xmask = x2mask & x3mask & x4mask
        x1 = x3 + 64*x4
        x1mask = x3mask & x4mask
        x0 = x1 + 512*x2
        x0mask = x1mask & x2mask
        tmp0 = tl.load(in_ptr0 + (x0), x0mask).to(tl.float32)
        tmp1 = tl.full([1], 0.25, tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tmp2.to(tl.float16)
        tmp4 = tmp3.to(tl.float32)
        tl.store(out_ptr0 + (x0), tmp4, x0mask)
        tl.store(out_ptr1 + (16*x4 + 128*(x3 // 16) + 512*x2 + ((x3 % 16))), tmp4, x2mask & x3mask & x4mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t107-native-j6xlvgc9/adapter/inductor-cache/tmpn7rsjodp/vm/cvmhms6ucrq53kq2e4vckbz2zy7cjmwh7xr5znewewaclbe67hcr.py
# Topologically Sorted Source Nodes: [permute, transpose, mul_scalar_1, matmul], Original ATen: [aten.permute, aten.transpose, aten.mul, aten.matmul]
# Source node to ATen node mapping:
#   matmul => clone_1
#   mul_scalar_1 => mul_1
#   permute => permute
#   transpose => permute_3
# Graph fragment:
#   %primals_1 : Tensor "f16[2, 8, 4, 16][512, 64, 16, 1]npu:0" = PlaceHolder[target=primals_1]
#   %expand_1 : Tensor "f16[2, 4, 16, 8][512, 16, 1, 64]npu:0" = PlaceHolder[target=expand_1]
#   %permute : Tensor "f16[2, 4, 8, 16][512, 16, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_1, [0, 2, 1, 3]), kwargs = {})
#   %permute_3 : Tensor "f16[2, 4, 16, 8][512, 16, 1, 64]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute, [0, 1, 3, 2]), kwargs = {})
#   %mul_1 : Tensor "f16[2, 4, 16, 8][512, 16, 1, 64]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_3, 0.25), kwargs = {})
#   %clone_1 : Tensor "f16[2, 4, 16, 8][512, 128, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %expand_1,%clone_1
triton_unk_fused_matmul_mul_permute_transpose_1 = async_compile.triton('triton_unk_fused_matmul_mul_permute_transpose_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [2, 64, 8, {'divisors': [512, 1, 64]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x2', 'length': 2, 'divisor': 512, 'seed': 2}, {'name': 'x3', 'length': 64, 'divisor': 1, 'seed': 1}, {'name': 'x4', 'length': 8, 'divisor': 64, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_mul_permute_transpose_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_mul_permute_transpose_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 1024
    x_g_tile0 : tl.constexpr = (64) if (64) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (8) if (8) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (2) if (2) < (x_g_rem2) else (x_g_rem2)
    x2numel : tl.constexpr = 2
    x3numel : tl.constexpr = 64
    x4numel : tl.constexpr = 8
    real_block_x2 : tl.constexpr = x_g_tile2
    real_block_x3 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x4 : tl.constexpr = x_g_tile1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x4_blocks : tl.constexpr = (x4numel + real_block_x4 - 1) // real_block_x4
    x_cumblk_1 = x2_blocks
    x_cumblk_2 = x2_blocks * x3_blocks
    total_blocks = x2_blocks * x3_blocks * x4_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x2offset = (group_base + i) % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_1 % x3_blocks * real_block_x3
        x4offset = (group_base + i) // x_cumblk_2 % x4_blocks * real_block_x4
        x2index = x2offset + tl.arange(0, real_block_x2)[:, None, None]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[None, None, :]
        x3 = x3index
        x3mask = x3index < x3numel
        x4index = x4offset + tl.arange(0, real_block_x4)[None, :, None]
        x4 = x4index
        x4mask = x4index < x4numel
        xmask = x2mask & x3mask & x4mask
        x1 = x3 + 64*x4
        x1mask = x3mask & x4mask
        x0 = x1 + 512*x2
        x0mask = x1mask & x2mask
        tmp0 = tl.load(in_ptr0 + (x0), x0mask).to(tl.float32)
        tmp1 = tl.full([1], 0.25, tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tmp2.to(tl.float16)
        tmp4 = tmp3.to(tl.float32)
        tl.store(out_ptr0 + (x0), tmp4, x0mask)
        tl.store(out_ptr1 + (x4 + 8*x3 + 512*x2), tmp4, x2mask & x3mask & x4mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t107-native-j6xlvgc9/adapter/inductor-cache/tmpn7rsjodp/ac/cacdg6xtkqpibgo46upw4g4bl7uj26dntfrzlvqkihzowawnxpn4.py
# Topologically Sorted Source Nodes: [matmul, _safe_softmax], Original ATen: [aten.matmul, aten._safe_softmax]
# Source node to ATen node mapping:
#   _safe_softmax => eq, logical_not
#   matmul => view_2
# Graph fragment:
#   %bmm : Tensor "f16[8, 8, 8][64, 8, 1]npu:0" = PlaceHolder[target=bmm]
#   %view_2 : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2, 4, 8, 8]), kwargs = {})
#   %eq : Tensor "b8[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%view_2, -inf), kwargs = {})
#   %logical_not : Tensor "b8[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   return %logical_not
triton_unk_fused__safe_softmax_matmul_2 = async_compile.triton('triton_unk_fused__safe_softmax_matmul_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax_matmul_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__safe_softmax_matmul_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        tmp1 = tl.full([1], float("-inf"), tl.float32)
        tmp2 = tmp0 == tmp1
        tmp3 = tmp2 == 0
        tl.store(out_ptr0 + (x0), tmp3, x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t107-native-j6xlvgc9/adapter/inductor-cache/tmpn7rsjodp/wi/cwi2uthyrxdqvthmccfd3a2amqwg2qbfmc774h7cali4zc6u5c4d.py
# Topologically Sorted Source Nodes: [matmul, _safe_softmax], Original ATen: [aten.matmul, aten._safe_softmax]
# Source node to ATen node mapping:
#   _safe_softmax => amax, convert_element_type_2, convert_element_type_3, div, exp, full_default, logical_not_1, sub, sum_1, where
#   matmul => view_2
# Graph fragment:
#   %bmm : Tensor "f16[8, 8, 8][64, 8, 1]npu:0" = PlaceHolder[target=bmm]
#   %amax : Tensor "f32[2, 4, 8, 1][32, 8, 1, 64]npu:0" = PlaceHolder[target=amax]
#   %any_1 : Tensor "b8[2, 4, 8, 1][32, 8, 1, 1]npu:0" = PlaceHolder[target=any_1]
#   %sum_1 : Tensor "f32[2, 4, 8, 1][32, 8, 1, 64]npu:0" = PlaceHolder[target=sum_1]
#   %view_2 : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2, 4, 8, 8]), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %amax : Tensor "f32[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_2, [-1], True), kwargs = {})
#   %sub : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %amax), kwargs = {})
#   %exp : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : Tensor "f32[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : Tensor "f32[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %convert_element_type_3 : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
#   %logical_not_1 : Tensor "b8[2, 4, 8, 1][32, 8, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %full_default : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 4, 8, 8], 0), kwargs = {dtype: torch.float16, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "f16[2, 4, 8, 8][256, 64, 8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_1, %full_default, %convert_element_type_3), kwargs = {})
#   return %amax,%sum_1,%expand_2
triton_unk_fused__safe_softmax_matmul_3 = async_compile.triton('triton_unk_fused__safe_softmax_matmul_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax_matmul_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused__safe_softmax_matmul_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        _tmp3 = tl.full([real_block_x0, R0_BLOCK], float("-inf"), tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tl.broadcast_to(tmp1, [real_block_x0, R0_BLOCK])
            tmp4 = tl.maximum(_tmp3, tmp2)
            _tmp3 = tmp4
        tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
        _tmp10 = tl.full([real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp5 = tl.load(in_out_ptr0 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
            tmp6 = tmp5.to(tl.float32)
            tmp7 = tmp6 - tmp3
            tmp8 = libdevice.exp(tmp7)
            tmp9 = tl.broadcast_to(tmp8, [real_block_x0, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tmp11
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tmp12 = tl.load(in_ptr0 + x0, x0mask, eviction_policy='evict_last') != 0
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp14 = tl.load(in_out_ptr0 + (r0_1 + 8*x0), r0_mask & x0mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp13 = tmp12 == 0
            tmp15 = tmp14.to(tl.float32)
            tmp16 = tmp15 - tmp3
            tmp17 = libdevice.exp(tmp16)
            tmp18 = (tmp17 / tmp10)
            tmp19 = tmp18.to(tl.float32)
            tmp20 = tl.full([1, 1], 0.0, tl.float32)
            tmp21 = tl.where(tmp13, tmp20, tmp19)
            tmp22 = tmp21.to(tl.float16)
            tmp23 = tmp22.to(tl.float32)
            tl.store(in_out_ptr0 + (r0_1 + 8*x0), tmp23, r0_mask & x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t107-native-j6xlvgc9/adapter/inductor-cache/tmpn7rsjodp/sm/csmbaye3icl2dqxnpsiddtf4yxqpr2mj3d5y5duch34e2d5vkp22.py
# Topologically Sorted Source Nodes: [permute_2, matmul_1], Original ATen: [aten.permute, aten.matmul]
# Source node to ATen node mapping:
#   matmul_1 => clone_2
#   permute_2 => permute_2
# Graph fragment:
#   %primals_3 : Tensor "f16[2, 8, 4, 16][512, 64, 16, 1]npu:0" = PlaceHolder[target=primals_3]
#   %permute_2 : Tensor "f16[2, 4, 8, 16][512, 16, 64, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%primals_3, [0, 2, 1, 3]), kwargs = {})
#   %clone_2 : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_unk_fused_matmul_permute_4 = async_compile.triton('triton_unk_fused_matmul_permute_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [16, 8, 4, 2, {'divisors': [1, 16, 128, 512]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 16, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 8, 'divisor': 16, 'seed': 2}, {'name': 'x2', 'length': 4, 'divisor': 128, 'seed': 2}, {'name': 'x3', 'length': 2, 'divisor': 512, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 1024
    x_g_tile0 : tl.constexpr = (16) if (16) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (8) if (8) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (4) if (4) < (x_g_rem2) else (x_g_rem2)
    x_g_rem3 : tl.constexpr = ((x_g_rem2) // x_g_tile2) if ((x_g_rem2) // x_g_tile2) > 1 else 1
    x_g_tile3 : tl.constexpr = (2) if (2) < (x_g_rem3) else (x_g_rem3)
    x0numel : tl.constexpr = 16
    x1numel : tl.constexpr = 8
    x2numel : tl.constexpr = 4
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
        tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 64*x1 + 512*x3), x0mask & x1mask & x2mask & x3mask).to(tl.float32)
        tl.store(out_ptr0 + (x0 + 16*x1 + 128*x2 + 512*x3), tmp0, x0mask & x1mask & x2mask & x3mask)
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
            buf0 = empty_strided_npu((2, 4, 8, 16), (512, 16, 64, 1), torch.float16)
            buf2 = empty_strided_npu((2, 4, 8, 16), (512, 128, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [permute_1, mul_scalar, matmul], Original ATen: [aten.permute, aten.mul, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_mul_permute_0.run(primals_2, buf0, buf2, 1024, stream=raw_stream0)
            del primals_2
            primals_1 = copy_if_misaligned(primals_1)
            buf1 = empty_strided_npu((2, 4, 16, 8), (512, 16, 1, 64), torch.float16)
            buf3 = empty_strided_npu((2, 4, 16, 8), (512, 128, 8, 1), torch.float16)
            # Topologically Sorted Source Nodes: [permute, transpose, mul_scalar_1, matmul], Original ATen: [aten.permute, aten.transpose, aten.mul, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_mul_permute_transpose_1.run(primals_1, buf1, buf3, 1024, stream=raw_stream0)
            del primals_1
            buf4 = empty_strided_npu((8, 8, 8), (64, 8, 1), torch.float16)
            # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:1
            extern_kernels.bmm(reinterpret_tensor(buf2, (8, 8, 16), (128, 16, 1), 0), reinterpret_tensor(buf3, (8, 16, 8), (128, 8, 1), 0), out=buf4)
            del buf2
            del buf3
            buf7 = empty_strided_npu((2, 4, 8, 8), (256, 64, 8, 1), torch.bool)
            # Topologically Sorted Source Nodes: [matmul, _safe_softmax], Original ATen: [aten.matmul, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax_matmul_2.run(buf4, buf7, 512, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [matmul, _safe_softmax], Original ATen: [aten.matmul, aten._safe_softmax]
            # [Provenance debug handles] torch.ops.aten.any.dim:2
            buf8 = torch.ops.aten.any.dim(buf7, -1, True)
            assert_alignment(buf8, 16, 'torch.ops.aten.any.dim')
            del buf7
            buf9 = reinterpret_tensor(buf4, (2, 4, 8, 8), (256, 64, 8, 1), 0); del buf4  # reuse
            # Topologically Sorted Source Nodes: [matmul, _safe_softmax], Original ATen: [aten.matmul, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax_matmul_3.run(buf9, buf8, 64, 8, stream=raw_stream0)
            del buf8
            primals_3 = copy_if_misaligned(primals_3)
            buf10 = empty_strided_npu((2, 4, 8, 16), (512, 128, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [permute_2, matmul_1], Original ATen: [aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_4.run(primals_3, buf10, 1024, stream=raw_stream0)
            buf11 = empty_strided_npu((8, 8, 16), (128, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [permute_2, matmul_1], Original ATen: [aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:3
            extern_kernels.bmm(reinterpret_tensor(buf9, (8, 8, 8), (64, 8, 1), 0), reinterpret_tensor(buf10, (8, 8, 16), (128, 16, 1), 0), out=buf11)
            del buf10
        return (reinterpret_tensor(buf11, (2, 4, 8, 16), (512, 128, 16, 1), 0), buf0, reinterpret_tensor(primals_3, (2, 4, 8, 16), (512, 16, 64, 1), 0), buf1, buf9, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((2, 8, 4, 16), (512, 64, 16, 1), device='npu:0', dtype=torch.float16)
    primals_2 = rand_strided((2, 8, 4, 16), (512, 64, 16, 1), device='npu:0', dtype=torch.float16)
    primals_3 = rand_strided((2, 8, 4, 16), (512, 64, 16, 1), device='npu:0', dtype=torch.float16)
    return [primals_1, primals_2, primals_3]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
