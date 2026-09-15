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


# kernel path: /home/z50063656/tmp/attention-functional-20260914-unskip/functional-20260914T233011+0800/pattern-23/off/inductor-cache/tmp5l1ssgz1/k6/ck674jpirmxolj5h635ld7mjvb67g23yjc4qxg5huoxntqwugl3f.py
# Topologically Sorted Source Nodes: [query, score], Original ATen: [aten.permute, aten.matmul]
# Source node to ATen node mapping:
#   query => permute
#   score => clone
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


# kernel path: /home/z50063656/tmp/attention-functional-20260914-unskip/functional-20260914T233011+0800/pattern-23/off/inductor-cache/tmp5l1ssgz1/ji/cji2sv4xvaftgl3viy6aydlrlgchnbkkgq35gd3hlbbrkknt6lwp.py
# Topologically Sorted Source Nodes: [key, permute_3, score], Original ATen: [aten.permute, aten.matmul]
# Source node to ATen node mapping:
#   key => permute_1
#   permute_3 => permute_3
#   score => clone_1
# Graph fragment:
#   %arg1_1 : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %permute_1 : Tensor "f16[2, 8, 4, 16][512, 16, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%arg1_1, [0, 2, 1, 3]), kwargs = {})
#   %permute_3 : Tensor "f16[2, 8, 16, 4][512, 16, 1, 128]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_1 : Tensor "f16[2, 8, 16, 4][512, 64, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_unk_fused_matmul_permute_1 = async_compile.triton('triton_unk_fused_matmul_permute_1', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_matmul_permute_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_matmul_permute_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /home/z50063656/tmp/attention-functional-20260914-unskip/functional-20260914T233011+0800/pattern-23/off/inductor-cache/tmp5l1ssgz1/7l/c7l6ynhwx7zr25c42uabjuwmu34mquwucmaiojfirrl5urmdm6fq.py
# Topologically Sorted Source Nodes: [score, float_2, softmax], Original ATen: [aten.matmul, aten._to_copy, aten._softmax]
# Source node to ATen node mapping:
#   float_2 => convert_element_type_4
#   score => view_2
#   softmax => amax, sub
# Graph fragment:
#   %bmm : Tensor "f16[16, 4, 4][16, 4, 1]npu:0" = PlaceHolder[target=bmm]
#   %view_2 : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2, 8, 4, 4]), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %amax : Tensor "f32[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_4, [-1], True), kwargs = {})
#   %sub : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_4, %amax), kwargs = {})
#   return %buf3
triton_unk_fused__softmax__to_copy_matmul_2 = async_compile.triton('triton_unk_fused__softmax__to_copy_matmul_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [4, 32, 2, {'divisors': [1, 4, 128]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}, {'name': 'x2', 'length': 32, 'divisor': 4, 'seed': 2}, {'name': 'x3', 'length': 2, 'divisor': 128, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax__to_copy_matmul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__softmax__to_copy_matmul_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 256
    x_g_tile0 : tl.constexpr = (4) if (4) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (32) if (32) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (2) if (2) < (x_g_rem2) else (x_g_rem2)
    x0numel : tl.constexpr = 4
    x2numel : tl.constexpr = 32
    x3numel : tl.constexpr = 2
    real_block_x0 : tl.constexpr = x_g_tile0
    real_block_x2 : tl.constexpr = x_g_tile1
    real_block_x3 : tl.constexpr = x_g_tile2
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_1 = x0_blocks
    x_cumblk_2 = x0_blocks * x2_blocks
    total_blocks = x0_blocks * x2_blocks * x3_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x2offset = (group_base + i) // x_cumblk_1 % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_2 % x3_blocks * real_block_x3
        x0index = x0offset + tl.arange(0, real_block_x0)[None, None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x2index = x2offset + tl.arange(0, real_block_x2)[None, :, None]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[:, None, None]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x0mask & x2mask & x3mask
        x1 = x2 + 32*x3
        x1mask = x2mask & x3mask
        _es_lane0 = tl.arange(0, 4)[None, None, :]
        _es_full0 = tl.load(in_ptr0 + (4 * x2 + 128 * x3) + _es_lane0, x2mask & x3mask, eviction_policy='evict_last').to(tl.float32)
        tmp0 = extract_slice(_es_full0, [0, 0, 0], [real_block_x3, real_block_x2, 1], [1, 1, 1])
        _es_lane1 = tl.arange(0, 4)[None, None, :]
        _es_full1 = tl.load(in_ptr0 + (1 + 4 * x2 + 128 * x3) + _es_lane1, x2mask & x3mask, eviction_policy='evict_last').to(tl.float32)
        tmp2 = extract_slice(_es_full1, [0, 0, 0], [real_block_x3, real_block_x2, 1], [1, 1, 1])
        _es_lane2 = tl.arange(0, 4)[None, None, :]
        _es_full2 = tl.load(in_ptr0 + (2 + 4 * x2 + 128 * x3) + _es_lane2, x2mask & x3mask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = extract_slice(_es_full2, [0, 0, 0], [real_block_x3, real_block_x2, 1], [1, 1, 1])
        _es_lane3 = tl.arange(0, 4)[None, None, :]
        _es_full3 = tl.load(in_ptr0 + (3 + 4 * x2 + 128 * x3) + _es_lane3, x2mask & x3mask, eviction_policy='evict_last').to(tl.float32)
        tmp8 = extract_slice(_es_full3, [0, 0, 0], [real_block_x3, real_block_x2, 1], [1, 1, 1])
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.maximum(tmp1, tmp3)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.maximum(tmp4, tmp6)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.maximum(tmp7, tmp9)
        tl.store(out_ptr0 + (x0 + 4*x2 + 128*x3), tmp10, x0mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/attention-functional-20260914-unskip/functional-20260914T233011+0800/pattern-23/off/inductor-cache/tmp5l1ssgz1/oj/cojfbobsmnpj2esm534er7muuki74jun4wyow2dx5mtu6oityhym.py
# Topologically Sorted Source Nodes: [score, float_2, softmax], Original ATen: [aten.matmul, aten._to_copy, aten._softmax]
# Source node to ATen node mapping:
#   float_2 => convert_element_type_4
#   score => view_2
#   softmax => amax, div, exp, sub, sum_1
# Graph fragment:
#   %bmm : Tensor "f16[16, 4, 4][16, 4, 1]npu:0" = PlaceHolder[target=bmm]
#   %buf3 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0" = PlaceHolder[target=buf3]
#   %view_2 : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2, 8, 4, 4]), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %amax : Tensor "f32[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_4, [-1], True), kwargs = {})
#   %sub : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_4, %amax), kwargs = {})
#   %exp : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : Tensor "f32[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   return %buf4
triton_unk_fused__softmax__to_copy_matmul_3 = async_compile.triton('triton_unk_fused__softmax__to_copy_matmul_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [4, 32, 2, {'divisors': [1, 4, 128]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4, 'divisor': 1, 'seed': 1}, {'name': 'x2', 'length': 32, 'divisor': 4, 'seed': 2}, {'name': 'x3', 'length': 2, 'divisor': 128, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax__to_copy_matmul_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__softmax__to_copy_matmul_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 256
    x_g_tile0 : tl.constexpr = (4) if (4) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (32) if (32) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (2) if (2) < (x_g_rem2) else (x_g_rem2)
    x0numel : tl.constexpr = 4
    x2numel : tl.constexpr = 32
    x3numel : tl.constexpr = 2
    real_block_x0 : tl.constexpr = x_g_tile0
    real_block_x2 : tl.constexpr = x_g_tile1
    real_block_x3 : tl.constexpr = x_g_tile2
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_1 = x0_blocks
    x_cumblk_2 = x0_blocks * x2_blocks
    total_blocks = x0_blocks * x2_blocks * x3_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x2offset = (group_base + i) // x_cumblk_1 % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_2 % x3_blocks * real_block_x3
        x0index = x0offset + tl.arange(0, real_block_x0)[None, None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        x2index = x2offset + tl.arange(0, real_block_x2)[None, :, None]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[:, None, None]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x0mask & x2mask & x3mask
        x1 = x2 + 32*x3
        x1mask = x2mask & x3mask
        _es_lane0 = tl.arange(0, 4)[None, None, :]
        _es_full0 = tl.load(in_ptr0 + (4 * x2 + 128 * x3) + _es_lane0, x2mask & x3mask, eviction_policy='evict_last').to(tl.float32)
        tmp0 = extract_slice(_es_full0, [0, 0, 0], [real_block_x3, real_block_x2, 1], [1, 1, 1])
        _es_lane1 = tl.arange(0, 4)[None, None, :]
        _es_full1 = tl.load(in_ptr1 + (4 * x2 + 128 * x3) + _es_lane1, x2mask & x3mask, eviction_policy='evict_last')
        tmp2 = extract_slice(_es_full1, [0, 0, 0], [real_block_x3, real_block_x2, 1], [1, 1, 1])
        _es_lane2 = tl.arange(0, 4)[None, None, :]
        _es_full2 = tl.load(in_ptr0 + (1 + 4 * x2 + 128 * x3) + _es_lane2, x2mask & x3mask, eviction_policy='evict_last').to(tl.float32)
        tmp5 = extract_slice(_es_full2, [0, 0, 0], [real_block_x3, real_block_x2, 1], [1, 1, 1])
        _es_lane3 = tl.arange(0, 4)[None, None, :]
        _es_full3 = tl.load(in_ptr1 + (1 + 4 * x2 + 128 * x3) + _es_lane3, x2mask & x3mask, eviction_policy='evict_last')
        tmp7 = extract_slice(_es_full3, [0, 0, 0], [real_block_x3, real_block_x2, 1], [1, 1, 1])
        _es_lane4 = tl.arange(0, 4)[None, None, :]
        _es_full4 = tl.load(in_ptr0 + (2 + 4 * x2 + 128 * x3) + _es_lane4, x2mask & x3mask, eviction_policy='evict_last').to(tl.float32)
        tmp11 = extract_slice(_es_full4, [0, 0, 0], [real_block_x3, real_block_x2, 1], [1, 1, 1])
        _es_lane5 = tl.arange(0, 4)[None, None, :]
        _es_full5 = tl.load(in_ptr1 + (2 + 4 * x2 + 128 * x3) + _es_lane5, x2mask & x3mask, eviction_policy='evict_last')
        tmp13 = extract_slice(_es_full5, [0, 0, 0], [real_block_x3, real_block_x2, 1], [1, 1, 1])
        _es_lane6 = tl.arange(0, 4)[None, None, :]
        _es_full6 = tl.load(in_ptr0 + (3 + 4 * x2 + 128 * x3) + _es_lane6, x2mask & x3mask, eviction_policy='evict_last').to(tl.float32)
        tmp17 = extract_slice(_es_full6, [0, 0, 0], [real_block_x3, real_block_x2, 1], [1, 1, 1])
        _es_lane7 = tl.arange(0, 4)[None, None, :]
        _es_full7 = tl.load(in_ptr1 + (3 + 4 * x2 + 128 * x3) + _es_lane7, x2mask & x3mask, eviction_policy='evict_last')
        tmp19 = extract_slice(_es_full7, [0, 0, 0], [real_block_x3, real_block_x2, 1], [1, 1, 1])
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 - tmp2
        tmp4 = libdevice.exp(tmp3)
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 - tmp7
        tmp9 = libdevice.exp(tmp8)
        tmp10 = tmp4 + tmp9
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp12 - tmp13
        tmp15 = libdevice.exp(tmp14)
        tmp16 = tmp10 + tmp15
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 - tmp19
        tmp21 = libdevice.exp(tmp20)
        tmp22 = tmp16 + tmp21
        tl.store(out_ptr0 + (x0 + 4*x2 + 128*x3), tmp22, x0mask & x2mask & x3mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/attention-functional-20260914-unskip/functional-20260914T233011+0800/pattern-23/off/inductor-cache/tmp5l1ssgz1/ss/cssrhkmfrzpevorkbvjyvnatfvnabpd6qejoz6efygnii44jnfhp.py
# Topologically Sorted Source Nodes: [score, float_2, softmax, type_as_1], Original ATen: [aten.matmul, aten._to_copy, aten._softmax]
# Source node to ATen node mapping:
#   float_2 => convert_element_type_4
#   score => view_2
#   softmax => amax, div, exp, sub, sum_1
#   type_as_1 => convert_element_type_5
# Graph fragment:
#   %bmm : Tensor "f16[16, 4, 4][16, 4, 1]npu:0" = PlaceHolder[target=bmm]
#   %buf3 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0" = PlaceHolder[target=buf3]
#   %buf4 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0" = PlaceHolder[target=buf4]
#   %view_2 : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [2, 8, 4, 4]), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %amax : Tensor "f32[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_4, [-1], True), kwargs = {})
#   %sub : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_4, %amax), kwargs = {})
#   %exp : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : Tensor "f32[2, 8, 4, 1][32, 4, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : Tensor "f32[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %convert_element_type_5 : Tensor "f16[2, 8, 4, 4][128, 16, 4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
#   return %expand_2
triton_unk_fused__softmax__to_copy_matmul_4 = async_compile.triton('triton_unk_fused__softmax__to_copy_matmul_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [256, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 256, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__softmax__to_copy_matmul_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__softmax__to_copy_matmul_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (x0), x0mask).to(tl.float32)
        tmp2 = tl.load(in_ptr0 + (x0), x0mask)
        tmp5 = tl.load(in_ptr1 + (x0), x0mask)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 - tmp2
        tmp4 = libdevice.exp(tmp3)
        tmp6 = (tmp4 / tmp5)
        tmp7 = tmp6.to(tl.float32)
        tl.store(in_out_ptr0 + (x0), tmp7, x0mask)
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
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_npu((2, 8, 4, 16), (512, 64, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [query, score], Original ATen: [aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_0.run(arg0_1, buf0, 1024, stream=raw_stream0)
            del arg0_1
            arg1_1 = copy_if_misaligned(arg1_1)
            buf1 = empty_strided_npu((2, 8, 16, 4), (512, 64, 4, 1), torch.float16)
            # Topologically Sorted Source Nodes: [key, permute_3, score], Original ATen: [aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_1.run(arg1_1, buf1, 256, 4, stream=raw_stream0)
            buf2 = empty_strided_npu((16, 4, 4), (16, 4, 1), torch.float16)
            # Topologically Sorted Source Nodes: [query, score, key, permute_3], Original ATen: [aten.permute, aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:1
            extern_kernels.bmm(reinterpret_tensor(buf0, (16, 4, 16), (64, 16, 1), 0), reinterpret_tensor(buf1, (16, 16, 4), (64, 4, 1), 0), out=buf2)
            del buf0
            del buf1
            buf3 = empty_strided_npu((2, 8, 4, 4), (128, 16, 4, 1), torch.float32)
            # Topologically Sorted Source Nodes: [score, float_2, softmax], Original ATen: [aten.matmul, aten._to_copy, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax__to_copy_matmul_2.run(buf2, buf3, 256, stream=raw_stream0)
            buf4 = empty_strided_npu((2, 8, 4, 4), (128, 16, 4, 1), torch.float32)
            # Topologically Sorted Source Nodes: [score, float_2, softmax], Original ATen: [aten.matmul, aten._to_copy, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax__to_copy_matmul_3.run(buf2, buf3, buf4, 256, stream=raw_stream0)
            buf5 = reinterpret_tensor(buf2, (2, 8, 4, 4), (128, 16, 4, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [score, float_2, softmax, type_as_1], Original ATen: [aten.matmul, aten._to_copy, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax__to_copy_matmul_4.run(buf5, buf3, buf4, 256, stream=raw_stream0)
            del buf3
            del buf4
            arg2_1 = copy_if_misaligned(arg2_1)
            buf6 = empty_strided_npu((2, 8, 4, 16), (512, 64, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [value, matmul_1], Original ATen: [aten.permute, aten.matmul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_matmul_permute_0.run(arg2_1, buf6, 1024, stream=raw_stream0)
            buf7 = empty_strided_npu((16, 4, 16), (64, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [score, float_2, softmax, type_as_1, matmul_1, value], Original ATen: [aten.matmul, aten._to_copy, aten._softmax, aten.permute]
            # [Provenance debug handles] extern_kernels.bmm:2
            extern_kernels.bmm(reinterpret_tensor(buf5, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf6, (16, 4, 16), (64, 16, 1), 0), out=buf7)
            del buf5
            del buf6
        return (reinterpret_tensor(buf7, (2, 8, 4, 16), (512, 64, 16, 1), 0), reinterpret_tensor(arg1_1, (2, 8, 4, 16), (512, 16, 128, 1), 0), reinterpret_tensor(arg2_1, (2, 8, 4, 16), (512, 16, 128, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float16)
    arg1_1 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float16)
    arg2_1 = rand_strided((2, 4, 8, 16), (512, 128, 16, 1), device='npu:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
