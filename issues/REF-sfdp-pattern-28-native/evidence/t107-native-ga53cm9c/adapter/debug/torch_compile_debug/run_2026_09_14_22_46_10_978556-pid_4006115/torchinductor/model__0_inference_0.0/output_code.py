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


# kernel path: /home/z50063656/tmp/t107-native-ga53cm9c/adapter/inductor-cache/tmpdb05chgl/ee/cee5g2qgo3wqkq2egvj5sdhzw3g766k7nncsgovaeheuslpkppch.py
# Topologically Sorted Source Nodes: [permute, unbind, ], Original ATen: [aten.permute, aten.unbind, aten.clone, npu.npu_fusion_attention_v3]
# Source node to ATen node mapping:
#    => clone_default, clone_default_1, clone_default_2, npu_fusion_attention_v3_default
#   permute => permute
#   unbind => unbind
# Graph fragment:
#   %arg0_1 : Tensor "f16[2, 3, 4, 16, 8][1536, 512, 128, 8, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %permute : Tensor "f16[3, 2, 4, 8, 16][512, 1536, 128, 1, 8]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg0_1, [1, 0, 2, 4, 3]), kwargs = {})
#   %unbind : [num_users=3] = call_function[target=torch.ops.aten.unbind.int](args = (%permute,), kwargs = {})
#   %clone_default : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_default_1 : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_1,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_default_2 : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_2,), kwargs = {memory_format: torch.contiguous_format})
#   %npu_fusion_attention_v3_default : [num_users=1] = call_function[target=torch.ops.npu.npu_fusion_attention_v3.default](args = (%clone_default, %clone_default_1, %clone_default_2, 4, BNSD, None, None, None, 0.2), kwargs = {})
#   return %buf0
triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_0 = async_compile.triton('triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_0', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /home/z50063656/tmp/t107-native-ga53cm9c/adapter/inductor-cache/tmpdb05chgl/np/cnpvu7fwjdd2jee4qpzzd5knw4o2egyuquzktsfb3ythxutpbkkf.py
# Topologically Sorted Source Nodes: [permute, unbind, ], Original ATen: [aten.permute, aten.unbind, aten.clone, npu.npu_fusion_attention_v3]
# Source node to ATen node mapping:
#    => clone_default, clone_default_1, clone_default_2, npu_fusion_attention_v3_default
#   permute => permute
#   unbind => unbind
# Graph fragment:
#   %arg0_1 : Tensor "f16[2, 3, 4, 16, 8][1536, 512, 128, 8, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %permute : Tensor "f16[3, 2, 4, 8, 16][512, 1536, 128, 1, 8]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg0_1, [1, 0, 2, 4, 3]), kwargs = {})
#   %unbind : [num_users=3] = call_function[target=torch.ops.aten.unbind.int](args = (%permute,), kwargs = {})
#   %clone_default : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_default_1 : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_1,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_default_2 : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_2,), kwargs = {memory_format: torch.contiguous_format})
#   %npu_fusion_attention_v3_default : [num_users=1] = call_function[target=torch.ops.npu.npu_fusion_attention_v3.default](args = (%clone_default, %clone_default_1, %clone_default_2, 4, BNSD, None, None, None, 0.2), kwargs = {})
#   return %buf1
triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_1 = async_compile.triton('triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_1', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (512 + y0 + 8*x3 + 128*y1 + 1536*y2), x3mask & y0mask & y1mask & y2mask).to(tl.float32)
        tl.store(out_ptr0 + (x3 + 16*y0 + 128*y1 + 512*y2), tmp0, x3mask & y0mask & y1mask & y2mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t107-native-ga53cm9c/adapter/inductor-cache/tmpdb05chgl/66/c66mkcm42enrn24szuixfdxjbqosh7bc3oen3d5koukbhvzgsrpb.py
# Topologically Sorted Source Nodes: [permute, unbind, ], Original ATen: [aten.permute, aten.unbind, aten.clone, npu.npu_fusion_attention_v3]
# Source node to ATen node mapping:
#    => clone_default, clone_default_1, clone_default_2, npu_fusion_attention_v3_default
#   permute => permute
#   unbind => unbind
# Graph fragment:
#   %arg0_1 : Tensor "f16[2, 3, 4, 16, 8][1536, 512, 128, 8, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %permute : Tensor "f16[3, 2, 4, 8, 16][512, 1536, 128, 1, 8]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg0_1, [1, 0, 2, 4, 3]), kwargs = {})
#   %unbind : [num_users=3] = call_function[target=torch.ops.aten.unbind.int](args = (%permute,), kwargs = {})
#   %clone_default : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_default_1 : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_1,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_default_2 : Tensor "f16[2, 4, 8, 16][512, 128, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_2,), kwargs = {memory_format: torch.contiguous_format})
#   %npu_fusion_attention_v3_default : [num_users=1] = call_function[target=torch.ops.npu.npu_fusion_attention_v3.default](args = (%clone_default, %clone_default_1, %clone_default_2, 4, BNSD, None, None, None, 0.2), kwargs = {})
#   return %buf2
triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_2 = async_compile.triton('triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_2', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 4, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
        arg0_1, = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_npu((2, 4, 8, 16), (512, 128, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [permute, unbind, ], Original ATen: [aten.permute, aten.unbind, aten.clone, npu.npu_fusion_attention_v3]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_0.run(arg0_1, buf0, 64, 16, stream=raw_stream0)
            buf1 = empty_strided_npu((2, 4, 8, 16), (512, 128, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [permute, unbind, ], Original ATen: [aten.permute, aten.unbind, aten.clone, npu.npu_fusion_attention_v3]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_1.run(arg0_1, buf1, 64, 16, stream=raw_stream0)
            buf2 = empty_strided_npu((2, 4, 8, 16), (512, 128, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [permute, unbind, ], Original ATen: [aten.permute, aten.unbind, aten.clone, npu.npu_fusion_attention_v3]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_clone_npu_fusion_attention_v3_permute_unbind_2.run(arg0_1, buf2, 64, 16, stream=raw_stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [permute, unbind, ], Original ATen: [aten.permute, aten.unbind, aten.clone, npu.npu_fusion_attention_v3]
            # [Provenance debug handles] torch.ops.npu.npu_fusion_attention_v3.default:1
            buf3 = torch.ops.npu.npu_fusion_attention_v3.default(buf0, buf1, buf2, 4, 'BNSD', None, None, None, 0.2, keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False, softmax_layout='', sink=None)
            del buf0
            del buf1
            del buf2
            buf4 = buf3[0]
            assert_alignment(buf4, 16, 'torch.ops.npu.npu_fusion_attention_v3.default')
            del buf3
        return (buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2, 3, 4, 16, 8), (1536, 512, 128, 8, 1), device='npu:0', dtype=torch.float16)
    return [arg0_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
