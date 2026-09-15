# AOT ID: ['11_forward']
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


# kernel path: /home/z50063656/tmp/t102-native-p5mtyp04/adapter/inductor-cache/tmpuertrzee/hu/chul34wo73dosgps7a7lruccvpm4wkwam3tm6citkatbjnfze32e.py
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


# kernel path: /home/z50063656/tmp/t102-native-p5mtyp04/adapter/inductor-cache/tmpuertrzee/vl/cvlf3izkjjpfy6vc7lrz4jzxcdypieyik3r2tklpqaizmdh46uzr.py
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


# kernel path: /home/z50063656/tmp/t102-native-p5mtyp04/adapter/inductor-cache/tmpuertrzee/5j/c5jftdq6opf24s3bmy57lq6tlwpxjgt5kudgvmdcm62csvafptth.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#    => mul_scalar
# Graph fragment:
#   %primals_2 : Tensor "f32[4, 2, 16, 32][1024, 512, 32, 1]npu:0" = PlaceHolder[target=primals_2]
#   %mul_scalar : Tensor "f32[4, 2, 16, 32][1024, 512, 32, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Scalar](args = (%primals_2, 0.42044820762685725), kwargs = {})
#   return %expand_default
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [4096, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 4096, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_mul_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_mul_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0), x0mask)
        tmp1 = tl.full([1], 0.42044820762685725, tl.float32)
        tmp2 = tmp0 * tmp1
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t102-native-p5mtyp04/adapter/inductor-cache/tmpuertrzee/rb/crb3wxrqilubvr7zbmbk6dkk3da3g4mor7wvqn72vjxybuvbivcv.py
# Topologically Sorted Source Nodes: [logical_not, masked_fill, ], Original ATen: [aten.logical_not, aten.masked_fill, aten._to_copy, aten.matmul, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#    => add_tensor, convert_element_type_default, eq_scalar, logical_not_default, view_default_2
#   logical_not => logical_not
#   masked_fill => full_default_1, where
# Graph fragment:
#   %bmm_default : Tensor "f32[8, 16, 16][256, 16, 1]npu:0" = PlaceHolder[target=bmm_default]
#   %logical_and : Tensor "b8[16, 16][16, 1]npu:0" = PlaceHolder[target=logical_and]
#   %logical_not : Tensor "b8[16, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and,), kwargs = {})
#   %full_default_1 : Tensor "b8[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "b8[16, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not, %full_default_1, %logical_and), kwargs = {})
#   %convert_element_type_default : Tensor "f32[16, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %view_default_2 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default, [4, 2, 16, 16]), kwargs = {})
#   %add_tensor : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %convert_element_type_default), kwargs = {})
#   %eq_scalar : Tensor "b8[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_tensor, -inf), kwargs = {})
#   %logical_not_default : Tensor "b8[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_scalar,), kwargs = {})
#   return %logical_not_default
triton_unk_fused__safe_softmax__to_copy_add_logical_not_masked_fill_matmul_3 = async_compile.triton('triton_unk_fused__safe_softmax__to_copy_add_logical_not_masked_fill_matmul_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [256, 8, {'divisors': [1, 256]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 256, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 8, 'divisor': 256, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax__to_copy_add_logical_not_masked_fill_matmul_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__safe_softmax__to_copy_add_logical_not_masked_fill_matmul_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 2048
    x_g_tile0 : tl.constexpr = (256) if (256) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (8) if (8) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 256
    x1numel : tl.constexpr = 8
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
        tmp0 = tl.load(in_ptr0 + (x0 + 256*x1), x0mask & x1mask)
        tmp1 = tl.load(in_ptr1 + x0, x0mask, eviction_policy='evict_last') != 0
        tmp2 = tmp1 == 0
        tmp3 = tl.full([1], True, tl.int1)
        tmp4 = tl.where(tmp2, tmp3, tmp1)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp0 + tmp5
        tmp7 = tl.full([1], float("-inf"), tl.float32)
        tmp8 = tmp6 == tmp7
        tmp9 = tmp8 == 0
        tl.store(out_ptr0 + (x0 + 256*x1), tmp9, x0mask & x1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t102-native-p5mtyp04/adapter/inductor-cache/tmpuertrzee/ng/cngeqjofpf5eavzn54ut3shzn4hgcrieceekzfyjz5kjjpu6t3oo.py
# Topologically Sorted Source Nodes: [logical_not, masked_fill, ], Original ATen: [aten.logical_not, aten.masked_fill, aten._to_copy, aten.matmul, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#    => add_tensor, amax_default, convert_element_type_default, div_tensor, exp_default, full_default_2, logical_not_default_1, sub_tensor, sum_dim_int_list, view_default_2, where_self
#   logical_not => logical_not
#   masked_fill => full_default_1, where
# Graph fragment:
#   %bmm_default : Tensor "f32[8, 16, 16][256, 16, 1]npu:0" = PlaceHolder[target=bmm_default]
#   %logical_and : Tensor "b8[16, 16][16, 1]npu:0" = PlaceHolder[target=logical_and]
#   %amax_default : Tensor "f32[4, 2, 16, 1][32, 16, 1, 128]npu:0" = PlaceHolder[target=amax_default]
#   %any_dim : Tensor "b8[4, 2, 16, 1][32, 16, 1, 1]npu:0" = PlaceHolder[target=any_dim]
#   %sum_dim_int_list : Tensor "f32[4, 2, 16, 1][32, 16, 1, 128]npu:0" = PlaceHolder[target=sum_dim_int_list]
#   %logical_not : Tensor "b8[16, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and,), kwargs = {})
#   %full_default_1 : Tensor "b8[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "b8[16, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not, %full_default_1, %logical_and), kwargs = {})
#   %convert_element_type_default : Tensor "f32[16, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %view_default_2 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_default, [4, 2, 16, 16]), kwargs = {})
#   %add_tensor : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %convert_element_type_default), kwargs = {})
#   %amax_default : Tensor "f32[4, 2, 16, 1][32, 16, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_tensor, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_tensor, %amax_default), kwargs = {})
#   %exp_default : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %sum_dim_int_list : Tensor "f32[4, 2, 16, 1][32, 16, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_default, [-1], True), kwargs = {})
#   %div_tensor : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %sum_dim_int_list), kwargs = {})
#   %logical_not_default_1 : Tensor "b8[4, 2, 16, 1][32, 16, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dim,), kwargs = {})
#   %full_default_2 : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 2, 16, 16], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_self : Tensor "f32[4, 2, 16, 16][512, 256, 16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_1, %full_default_2, %div_tensor), kwargs = {})
#   return %amax_default,%sum_dim_int_list,%expand_default_2
triton_unk_fused__safe_softmax__to_copy_add_logical_not_masked_fill_matmul_4 = async_compile.triton('triton_unk_fused__safe_softmax__to_copy_add_logical_not_masked_fill_matmul_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'r0_numel': 16}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [16, 8, {'divisors': [1, 16]}], 'R0_BLOCK_HINT': [16, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 16, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 8, 'divisor': 16, 'seed': 2}, {'name': 'r0_2', 'length': 16, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__safe_softmax__to_copy_add_logical_not_masked_fill_matmul_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused__safe_softmax__to_copy_add_logical_not_masked_fill_matmul_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        _tmp8 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], float("-inf"), tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (r0_2 + 16*x0 + 256*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=float("-inf"))
            tmp1 = tl.load(in_ptr0 + (r0_2 + 16 * x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0) != 0
            tmp2 = tmp1 == 0
            tmp3 = tl.full([1, 1], True, tl.int1)
            tmp4 = tl.where(tmp2, tmp3, tmp1)
            tmp5 = tmp4.to(tl.float32)
            tmp6 = tmp0 + tmp5
            tmp7 = tl.broadcast_to(tmp6, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp9 = tl.maximum(_tmp8, tmp7)
            _tmp8 = tmp9
        tmp8 = triton_helpers.max2(_tmp8, 2)[:, :, None]
        _tmp20 = tl.full([real_block_x1, real_block_x0, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp10 = tl.load(in_out_ptr0 + (r0_2 + 16*x0 + 256*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_last', other=float("-inf"))
            tmp11 = tl.load(in_ptr0 + (r0_2 + 16 * x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0) != 0
            tmp12 = tmp11 == 0
            tmp13 = tl.full([1, 1], True, tl.int1)
            tmp14 = tl.where(tmp12, tmp13, tmp11)
            tmp15 = tmp14.to(tl.float32)
            tmp16 = tmp10 + tmp15
            tmp17 = tmp16 - tmp8
            tmp18 = libdevice.exp(tmp17)
            tmp19 = tl.broadcast_to(tmp18, [real_block_x1, real_block_x0, R0_BLOCK])
            tmp21 = _tmp20 + tmp19
            _tmp20 = tmp21
        tmp20 = tl.sum(_tmp20, 2)[:, :, None]
        tmp22 = tl.load(in_ptr1 + (x0 + 16 * x1), x0mask & x1mask, eviction_policy='evict_last') != 0
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp24 = tl.load(in_out_ptr0 + (r0_2 + 16*x0 + 256*x1), r0_mask & x0mask & x1mask, eviction_policy='evict_first', other=0.0)
            tmp25 = tl.load(in_ptr0 + (r0_2 + 16 * x0), r0_mask & x0mask, eviction_policy='evict_last', other=0.0) != 0
            tmp23 = tmp22 == 0
            tmp26 = tmp25 == 0
            tmp27 = tl.full([1, 1], True, tl.int1)
            tmp28 = tl.where(tmp26, tmp27, tmp25)
            tmp29 = tmp28.to(tl.float32)
            tmp30 = tmp24 + tmp29
            tmp31 = tmp30 - tmp8
            tmp32 = libdevice.exp(tmp31)
            tmp33 = (tmp32 / tmp20)
            tmp34 = tl.full([1, 1], 0.0, tl.float32)
            tmp35 = tl.where(tmp23, tmp34, tmp33)
            tl.store(in_out_ptr0 + (r0_2 + 16*x0 + 256*x1), tmp35, r0_mask & x0mask & x1mask)
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
            buf0 = empty_strided_npu((16, 16), (16, 1), torch.bool)
            # Topologically Sorted Source Nodes: [tril], Original ATen: [aten.tril]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_tril_0.run(buf0, 256, stream=raw_stream0)
            buf1 = empty_strided_npu((16, 16), (16, 1), torch.bool)
            # Topologically Sorted Source Nodes: [ones], Original ATen: [aten.ones]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_ones_1.run(buf1, 256, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [ones, tril], Original ATen: [aten.ones, aten.tril]
            # [Provenance debug handles] torch.ops.aten.logical_and.default:1
            buf2 = torch.ops.aten.logical_and.default(buf0, buf1)
            assert_alignment(buf2, 16, 'torch.ops.aten.logical_and.default')
            del buf0
            del buf1
            primals_2 = copy_if_misaligned(primals_2)
            buf3 = empty_strided_npu((4, 2, 16, 32), (1024, 512, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_mul_2.run(primals_2, buf3, 4096, stream=raw_stream0)
            del primals_2
            primals_1 = copy_if_misaligned(primals_1)
            buf4 = empty_strided_npu((4, 2, 32, 16), (1024, 512, 1, 32), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_mul_2.run(primals_1, buf4, 4096, stream=raw_stream0)
            del primals_1
            buf5 = empty_strided_npu((8, 16, 16), (256, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:2
            extern_kernels.bmm(reinterpret_tensor(buf3, (8, 16, 32), (512, 32, 1), 0), reinterpret_tensor(buf4, (8, 32, 16), (512, 1, 32), 0), out=buf5)
            buf8 = empty_strided_npu((4, 2, 16, 16), (512, 256, 16, 1), torch.bool)
            # Topologically Sorted Source Nodes: [logical_not, masked_fill, ], Original ATen: [aten.logical_not, aten.masked_fill, aten._to_copy, aten.matmul, aten.add, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax__to_copy_add_logical_not_masked_fill_matmul_3.run(buf5, buf2, buf8, 2048, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [logical_not, masked_fill, ], Original ATen: [aten.logical_not, aten.masked_fill, aten._to_copy, aten.matmul, aten.add, aten._safe_softmax]
            # [Provenance debug handles] torch.ops.aten.any.dim:3
            buf9 = torch.ops.aten.any.dim(buf8, -1, True)
            assert_alignment(buf9, 16, 'torch.ops.aten.any.dim')
            del buf8
            buf10 = reinterpret_tensor(buf5, (4, 2, 16, 16), (512, 256, 16, 1), 0); del buf5  # reuse
            # Topologically Sorted Source Nodes: [logical_not, masked_fill, ], Original ATen: [aten.logical_not, aten.masked_fill, aten._to_copy, aten.matmul, aten.add, aten._safe_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__safe_softmax__to_copy_add_logical_not_masked_fill_matmul_4.run(buf10, buf2, buf9, 128, 16, stream=raw_stream0)
            del buf2
            del buf9
            primals_3 = copy_if_misaligned(primals_3)
            buf11 = empty_strided_npu((8, 16, 32), (512, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.matmul]
            # [Provenance debug handles] extern_kernels.bmm:4
            extern_kernels.bmm(reinterpret_tensor(buf10, (8, 16, 16), (256, 16, 1), 0), reinterpret_tensor(primals_3, (8, 16, 32), (512, 32, 1), 0), out=buf11)
        return (reinterpret_tensor(buf11, (4, 2, 16, 32), (1024, 512, 32, 1), 0), primals_3, buf3, buf4, buf10, )

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
