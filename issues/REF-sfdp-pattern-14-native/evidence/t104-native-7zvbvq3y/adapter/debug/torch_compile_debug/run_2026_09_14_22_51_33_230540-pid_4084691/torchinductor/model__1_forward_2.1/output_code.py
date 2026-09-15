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


# kernel path: /home/z50063656/tmp/t104-native-7zvbvq3y/adapter/inductor-cache/tmp51wu213h/rj/crjadnldct6zlydxc2zsqcxc2ftevfl7kgexhnnarv3vylf2lvwo.py
# Topologically Sorted Source Nodes: [permute_1, matmul], Original ATen: [aten.permute, aten.matmul]
# Source node to ATen node mapping:
#   matmul => clone
#   permute_1 => permute_1
# Graph fragment:
#   %primals_2 : Tensor "f32[4, 2, 16, 32][1024, 512, 32, 1]npu:0" = PlaceHolder[target=primals_2]
#   %permute_1 : Tensor "f32[4, 16, 2, 32][1024, 32, 512, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%primals_2, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "f32[4, 16, 2, 32][1024, 64, 32, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
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
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [32, 2, 16, 4, {'divisors': [1, 32, 64, 1024]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 32, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 2, 'divisor': 32, 'seed': 2}, {'name': 'x2', 'length': 16, 'divisor': 64, 'seed': 2}, {'name': 'x3', 'length': 4, 'divisor': 1024, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 4096
    x_g_tile0 : tl.constexpr = (32) if (32) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (2) if (2) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (16) if (16) < (x_g_rem2) else (x_g_rem2)
    x_g_rem3 : tl.constexpr = ((x_g_rem2) // x_g_tile2) if ((x_g_rem2) // x_g_tile2) > 1 else 1
    x_g_tile3 : tl.constexpr = (4) if (4) < (x_g_rem3) else (x_g_rem3)
    x0numel : tl.constexpr = 32
    x1numel : tl.constexpr = 2
    x2numel : tl.constexpr = 16
    x3numel : tl.constexpr = 4
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
        tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 512*x1 + 1024*x3), x0mask & x1mask & x2mask & x3mask)
        tl.store(out_ptr0 + (x0 + 32*x1 + 64*x2 + 1024*x3), tmp0, x0mask & x1mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t104-native-7zvbvq3y/adapter/inductor-cache/tmp51wu213h/ai/caihzeylyt4onfjvdeqai6cd2bdmebd4qidrf4wo653h7efig3qm.py
# Topologically Sorted Source Nodes: [permute, transpose, matmul], Original ATen: [aten.permute, aten.transpose, aten.matmul]
# Source node to ATen node mapping:
#   matmul => clone_1
#   permute => permute
#   transpose => permute_3
# Graph fragment:
#   %primals_1 : Tensor "f32[4, 2, 16, 32][1024, 512, 32, 1]npu:0" = PlaceHolder[target=primals_1]
#   %permute : Tensor "f32[4, 16, 2, 32][1024, 32, 512, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_1, [0, 2, 1, 3]), kwargs = {})
#   %permute_3 : Tensor "f32[4, 16, 32, 2][1024, 32, 1, 512]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute, [0, 1, 3, 2]), kwargs = {})
#   %clone_1 : Tensor "f32[4, 16, 32, 2][1024, 64, 2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
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
    size_hints={'y': 2048, 'x': 2}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'YBLOCK_HINT': [512, 4, {'divisors': [1, 512]}], 'XBLOCK_HINT': [2, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'y0', 'length': 512, 'divisor': 1, 'seed': 1}, {'name': 'y1', 'length': 4, 'divisor': 512, 'seed': 2}, {'name': 'x2', 'length': 2, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_transpose_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_transpose_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    ynumel = 2048
    xnumel = 2
    y_g_tile0 : tl.constexpr = (512) if (512) < (YBLOCK) else (YBLOCK)
    y_g_rem1 : tl.constexpr = ((YBLOCK) // y_g_tile0) if ((YBLOCK) // y_g_tile0) > 1 else 1
    y_g_tile1 : tl.constexpr = (4) if (4) < (y_g_rem1) else (y_g_rem1)
    y0numel : tl.constexpr = 512
    y1numel : tl.constexpr = 4
    real_block_y0 : tl.constexpr = (((y_g_tile0) + 7) // 8) * 8
    real_block_y1 : tl.constexpr = y_g_tile1
    y0_blocks : tl.constexpr = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks : tl.constexpr = (y1numel + real_block_y1 - 1) // real_block_y1
    y_cumblk_1 = y0_blocks
    x_g_tile0 : tl.constexpr = (2) if (2) < (XBLOCK) else (XBLOCK)
    x2numel : tl.constexpr = 2
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
        tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 1024*y1), x2mask & y0mask & y1mask)
        tl.store(out_ptr0 + (x2 + 2*y0 + 1024*y1), tmp0, x2mask & y0mask & y1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t104-native-7zvbvq3y/adapter/inductor-cache/tmp51wu213h/kq/ckqxqakogvhlnb7j3st6rwbevgq5p2qwgskrzqk2zm2m4e6x6nwg.py
# Topologically Sorted Source Nodes: [tril], Original ATen: [aten.tril]
# Source node to ATen node mapping:
#   tril => iota, le, sub, unsqueeze, unsqueeze_1
# Graph fragment:
#   %iota : Tensor "i64[2][1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (2,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: npu:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -2), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[2, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -1), kwargs = {})
#   %sub : Tensor "i64[2, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %le : Tensor "b8[2, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub, 0), kwargs = {})
#   return %le
triton_unk_fused_tril_2 = async_compile.triton('triton_unk_fused_tril_2', '''
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
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [2, 2, {'divisors': [1, 2]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 2, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 2, 'divisor': 2, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_tril_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_tril_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 4
    x_g_tile0 : tl.constexpr = (2) if (2) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (2) if (2) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 2
    x1numel : tl.constexpr = 2
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
        tmp0 = (x0 + ((-1)*x1)).to(tl.int32)
        tmp1 = tl.full([1], 0, tl.int32)
        tmp2 = tmp0 <= tmp1
        tl.store(out_ptr0 + (x0 + 2*x1), tmp2, x0mask & x1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t104-native-7zvbvq3y/adapter/inductor-cache/tmp51wu213h/z3/cz3mxjkmkx6ytsw557zopinhgaipdvcoyeoqb4eglbt5ojpxkpsm.py
# Topologically Sorted Source Nodes: [ones], Original ATen: [aten.ones]
# Source node to ATen node mapping:
#   ones => full_default
# Graph fragment:
#   %full_default : Tensor "b8[2, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 2], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: npu:0, pin_memory: False})
#   return %full_default
triton_unk_fused_ones_3 = async_compile.triton('triton_unk_fused_ones_3', '''
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
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [4, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_ones_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_ones_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 4
    x_g_tile0 : tl.constexpr = (4) if (4) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 4
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
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x0index = x0offset + tl.arange(0, real_block_x0)[:]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        tmp0 = tl.full([1], True, tl.int1)
        tl.store(out_ptr0 + (x0), tmp0, x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t104-native-7zvbvq3y/adapter/inductor-cache/tmp51wu213h/vj/cvjd3cr6k6sjpdpdykvobroxwt7wqroxjs2qeyua62qsk4lt7pz4.py
# Topologically Sorted Source Nodes: [matmul, div, logical_not, masked_fill, add, softmax], Original ATen: [aten.matmul, aten.div, aten.logical_not, aten.masked_fill, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   add => add
#   div => div
#   logical_not => logical_not
#   masked_fill => full_default_1, where
#   matmul => view_2
#   softmax => amax, sub_1
# Graph fragment:
#   %bmm : Tensor "f32[64, 2, 2][4, 2, 1]npu:0" = PlaceHolder[target=bmm]
#   %logical_and : Tensor "b8[2, 2][2, 1]npu:0" = PlaceHolder[target=logical_and]
#   %view_2 : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [4, 16, 2, 2]), kwargs = {})
#   %div : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_2, 3.0), kwargs = {})
#   %logical_not : Tensor "b8[2, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and,), kwargs = {})
#   %full_default_1 : Tensor "b8[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "b8[2, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not, %full_default_1, %logical_and), kwargs = {})
#   %add : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %where), kwargs = {})
#   %amax : Tensor "f32[4, 16, 2, 1][32, 2, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add, [-1], True), kwargs = {})
#   %sub_1 : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %amax), kwargs = {})
#   return %buf7
triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_4 = async_compile.triton('triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [2, 2, 64, {'divisors': [1, 2, 4]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 2, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 2, 'divisor': 2, 'seed': 2}, {'name': 'x2', 'length': 64, 'divisor': 4, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 256
    x_g_tile0 : tl.constexpr = (2) if (2) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (2) if (2) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (64) if (64) < (x_g_rem2) else (x_g_rem2)
    x0numel : tl.constexpr = 2
    x1numel : tl.constexpr = 2
    x2numel : tl.constexpr = 64
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
        _es_lane0 = tl.arange(0, 2)[None, None, :]
        _es_full0 = tl.load(in_ptr0 + (2 * x1 + 4 * x2) + _es_lane0, x1mask & x2mask, eviction_policy='evict_last')
        tmp0 = extract_slice(_es_full0, [0, 0, 0], [real_block_x2, real_block_x1, 1], [1, 1, 1])
        tmp3 = tl.load(in_ptr1 + 2 * x1, x1mask, eviction_policy='evict_last') != 0
        _es_lane1 = tl.arange(0, 2)[None, None, :]
        _es_full1 = tl.load(in_ptr0 + (1 + 2 * x1 + 4 * x2) + _es_lane1, x1mask & x2mask, eviction_policy='evict_last')
        tmp9 = extract_slice(_es_full1, [0, 0, 0], [real_block_x2, real_block_x1, 1], [1, 1, 1])
        tmp11 = tl.load(in_ptr1 + (1 + 2 * x1), x1mask, eviction_policy='evict_last') != 0
        tmp1 = tl.full([1], 0.3333333333333333, tl.float32)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp3 == 0
        tmp5 = tl.full([1], True, tl.int1)
        tmp6 = tl.where(tmp4, tmp5, tmp3)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp2 + tmp7
        tmp10 = tmp9 * tmp1
        tmp12 = tmp11 == 0
        tmp13 = tl.where(tmp12, tmp5, tmp11)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp10 + tmp14
        tmp16 = tl.maximum(tmp8, tmp15)
        tl.store(out_ptr0 + (x0 + 2*x1 + 4*x2), tmp16, x0mask & x1mask & x2mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t104-native-7zvbvq3y/adapter/inductor-cache/tmp51wu213h/33/c33vbpdvqburtawrned5owayxwg2rx6ch2zjraejgkhjnnqccqtn.py
# Topologically Sorted Source Nodes: [matmul, div, logical_not, masked_fill, add, softmax], Original ATen: [aten.matmul, aten.div, aten.logical_not, aten.masked_fill, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   add => add
#   div => div
#   logical_not => logical_not
#   masked_fill => full_default_1, where
#   matmul => view_2
#   softmax => amax, div_1, exp, sub_1, sum_1
# Graph fragment:
#   %bmm : Tensor "f32[64, 2, 2][4, 2, 1]npu:0" = PlaceHolder[target=bmm]
#   %logical_and : Tensor "b8[2, 2][2, 1]npu:0" = PlaceHolder[target=logical_and]
#   %buf7 : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0" = PlaceHolder[target=buf7]
#   %view_2 : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [4, 16, 2, 2]), kwargs = {})
#   %div : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_2, 3.0), kwargs = {})
#   %logical_not : Tensor "b8[2, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and,), kwargs = {})
#   %full_default_1 : Tensor "b8[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "b8[2, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not, %full_default_1, %logical_and), kwargs = {})
#   %add : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %where), kwargs = {})
#   %amax : Tensor "f32[4, 16, 2, 1][32, 2, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add, [-1], True), kwargs = {})
#   %sub_1 : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %amax), kwargs = {})
#   %exp : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : Tensor "f32[4, 16, 2, 1][32, 2, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   return %buf8
triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_5 = async_compile.triton('triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [2, 2, 64, {'divisors': [1, 2, 4]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 2, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 2, 'divisor': 2, 'seed': 2}, {'name': 'x2', 'length': 64, 'divisor': 4, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 256
    x_g_tile0 : tl.constexpr = (2) if (2) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (2) if (2) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (64) if (64) < (x_g_rem2) else (x_g_rem2)
    x0numel : tl.constexpr = 2
    x1numel : tl.constexpr = 2
    x2numel : tl.constexpr = 64
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
        _es_lane0 = tl.arange(0, 2)[None, None, :]
        _es_full0 = tl.load(in_ptr0 + (2 * x1 + 4 * x2) + _es_lane0, x1mask & x2mask, eviction_policy='evict_last')
        tmp0 = extract_slice(_es_full0, [0, 0, 0], [real_block_x2, real_block_x1, 1], [1, 1, 1])
        tmp3 = tl.load(in_ptr1 + 2 * x1, x1mask, eviction_policy='evict_last') != 0
        _es_lane1 = tl.arange(0, 2)[None, None, :]
        _es_full1 = tl.load(in_ptr2 + (2 * x1 + 4 * x2) + _es_lane1, x1mask & x2mask, eviction_policy='evict_last')
        tmp9 = extract_slice(_es_full1, [0, 0, 0], [real_block_x2, real_block_x1, 1], [1, 1, 1])
        _es_lane2 = tl.arange(0, 2)[None, None, :]
        _es_full2 = tl.load(in_ptr0 + (1 + 2 * x1 + 4 * x2) + _es_lane2, x1mask & x2mask, eviction_policy='evict_last')
        tmp12 = extract_slice(_es_full2, [0, 0, 0], [real_block_x2, real_block_x1, 1], [1, 1, 1])
        tmp14 = tl.load(in_ptr1 + (1 + 2 * x1), x1mask, eviction_policy='evict_last') != 0
        _es_lane3 = tl.arange(0, 2)[None, None, :]
        _es_full3 = tl.load(in_ptr2 + (1 + 2 * x1 + 4 * x2) + _es_lane3, x1mask & x2mask, eviction_policy='evict_last')
        tmp19 = extract_slice(_es_full3, [0, 0, 0], [real_block_x2, real_block_x1, 1], [1, 1, 1])
        tmp1 = tl.full([1], 0.3333333333333333, tl.float32)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp3 == 0
        tmp5 = tl.full([1], True, tl.int1)
        tmp6 = tl.where(tmp4, tmp5, tmp3)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp2 + tmp7
        tmp10 = tmp8 - tmp9
        tmp11 = libdevice.exp(tmp10)
        tmp13 = tmp12 * tmp1
        tmp15 = tmp14 == 0
        tmp16 = tl.where(tmp15, tmp5, tmp14)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp13 + tmp17
        tmp20 = tmp18 - tmp19
        tmp21 = libdevice.exp(tmp20)
        tmp22 = tmp11 + tmp21
        tl.store(out_ptr0 + (x0 + 2*x1 + 4*x2), tmp22, x0mask & x1mask & x2mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t104-native-7zvbvq3y/adapter/inductor-cache/tmp51wu213h/he/chesnegjesrupsepzmz5m3mn4byj3y2bdfirjo5ttjm2ermnwgbw.py
# Topologically Sorted Source Nodes: [matmul, div, logical_not, masked_fill, add, softmax], Original ATen: [aten.matmul, aten.div, aten.logical_not, aten.masked_fill, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   add => add
#   div => div
#   logical_not => logical_not
#   masked_fill => full_default_1, where
#   matmul => view_2
#   softmax => amax, div_1, exp, sub_1, sum_1
# Graph fragment:
#   %bmm : Tensor "f32[64, 2, 2][4, 2, 1]npu:0" = PlaceHolder[target=bmm]
#   %logical_and : Tensor "b8[2, 2][2, 1]npu:0" = PlaceHolder[target=logical_and]
#   %buf7 : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0" = PlaceHolder[target=buf7]
#   %buf8 : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0" = PlaceHolder[target=buf8]
#   %view_2 : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [4, 16, 2, 2]), kwargs = {})
#   %div : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_2, 3.0), kwargs = {})
#   %logical_not : Tensor "b8[2, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and,), kwargs = {})
#   %full_default_1 : Tensor "b8[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "b8[2, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not, %full_default_1, %logical_and), kwargs = {})
#   %add : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %where), kwargs = {})
#   %amax : Tensor "f32[4, 16, 2, 1][32, 2, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add, [-1], True), kwargs = {})
#   %sub_1 : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %amax), kwargs = {})
#   %exp : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : Tensor "f32[4, 16, 2, 1][32, 2, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : Tensor "f32[4, 16, 2, 2][64, 4, 2, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   return %expand_2
triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_6 = async_compile.triton('triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [4, 64, {'divisors': [1, 4]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 64, 'divisor': 4, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (x0 + 4*x1), x0mask & x1mask)
        tmp3 = tl.load(in_ptr0 + x0, x0mask, eviction_policy='evict_last') != 0
        tmp9 = tl.load(in_ptr1 + (x0 + 4*x1), x0mask & x1mask)
        tmp12 = tl.load(in_ptr2 + (x0 + 4*x1), x0mask & x1mask)
        tmp1 = tl.full([1], 0.3333333333333333, tl.float32)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp3 == 0
        tmp5 = tl.full([1], True, tl.int1)
        tmp6 = tl.where(tmp4, tmp5, tmp3)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp2 + tmp7
        tmp10 = tmp8 - tmp9
        tmp11 = libdevice.exp(tmp10)
        tmp13 = (tmp11 / tmp12)
        tl.store(in_out_ptr0 + (x0 + 4*x1), tmp13, x0mask & x1mask)
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
            buf0 = empty_strided_npu((4, 16, 2, 32), (1024, 64, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_1, matmul], Original ATen: [aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_0.run(primals_2, buf0, 4096, stream=raw_stream0)
            primals_1 = copy_if_misaligned(primals_1)
            buf1 = empty_strided_npu((4, 16, 32, 2), (1024, 64, 2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute, transpose, matmul], Original ATen: [aten.permute, aten.transpose, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_transpose_1.run(primals_1, buf1, 2048, 2, stream=raw_stream0)
            buf2 = empty_strided_npu((64, 2, 2), (4, 2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute, permute_1, transpose, matmul], Original ATen: [aten.permute, aten.transpose, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:1
            extern_kernels.bmm(reinterpret_tensor(buf0, (64, 2, 32), (64, 32, 1), 0), reinterpret_tensor(buf1, (64, 32, 2), (64, 2, 1), 0), out=buf2)
            del buf0
            del buf1
            buf4 = empty_strided_npu((2, 2), (2, 1), torch.bool)
            # Topologically Sorted Source Nodes: [tril], Original ATen: [aten.tril]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_tril_2.run(buf4, 4, stream=raw_stream0)
            buf5 = empty_strided_npu((2, 2), (2, 1), torch.bool)
            # Topologically Sorted Source Nodes: [ones], Original ATen: [aten.ones]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_ones_3.run(buf5, 4, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [ones, tril], Original ATen: [aten.ones, aten.tril]
            # [Provenance debug handles] torch.ops.aten.logical_and.default:2
            buf6 = torch.ops.aten.logical_and.default(buf4, buf5)
            assert_alignment(buf6, 16, 'torch.ops.aten.logical_and.default')
            del buf4
            del buf5
            buf7 = empty_strided_npu((4, 16, 2, 2), (64, 4, 2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [matmul, div, logical_not, masked_fill, add, softmax], Original ATen: [aten.matmul, aten.div, aten.logical_not, aten.masked_fill, aten.add, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_4.run(buf2, buf6, buf7, 256, stream=raw_stream0)
            buf8 = empty_strided_npu((4, 16, 2, 2), (64, 4, 2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [matmul, div, logical_not, masked_fill, add, softmax], Original ATen: [aten.matmul, aten.div, aten.logical_not, aten.masked_fill, aten.add, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_5.run(buf2, buf6, buf7, buf8, 256, stream=raw_stream0)
            buf9 = reinterpret_tensor(buf2, (4, 16, 2, 2), (64, 4, 2, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [matmul, div, logical_not, masked_fill, add, softmax], Original ATen: [aten.matmul, aten.div, aten.logical_not, aten.masked_fill, aten.add, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_add_div_logical_not_masked_fill_matmul_6.run(buf9, buf6, buf7, buf8, 256, stream=raw_stream0)
            del buf6
            del buf7
            del buf8
            primals_3 = copy_if_misaligned(primals_3)
            buf10 = empty_strided_npu((4, 16, 2, 32), (1024, 64, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_2, matmul_1], Original ATen: [aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_0.run(primals_3, buf10, 4096, stream=raw_stream0)
            buf11 = empty_strided_npu((64, 2, 32), (64, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_2, matmul_1], Original ATen: [aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:3
            extern_kernels.bmm(reinterpret_tensor(buf9, (64, 2, 2), (4, 2, 1), 0), reinterpret_tensor(buf10, (64, 2, 32), (64, 32, 1), 0), out=buf11)
            del buf10
        return (reinterpret_tensor(buf11, (4, 16, 2, 32), (1024, 64, 32, 1), 0), reinterpret_tensor(primals_2, (4, 16, 2, 32), (1024, 32, 512, 1), 0), reinterpret_tensor(primals_3, (4, 16, 2, 32), (1024, 32, 512, 1), 0), reinterpret_tensor(primals_1, (4, 16, 32, 2), (1024, 32, 1, 512), 0), buf9, )

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
