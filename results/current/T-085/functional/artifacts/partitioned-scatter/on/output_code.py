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


# kernel path: /home/z50063656/tmp/t085-performance-results/functional-partitioned-scatter-20260910T054442-vb4ys8_j/on/rank-0/inductor-cache/tmplchizw9z/s7/cs7z3cvsxv7zshfmzlwyqutjj77yflsfckxhldmiszu2dotyncvs.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
# Source node to ATen node mapping:
#    => bitwise_and_scalar_2, iota_default_2
# Graph fragment:
#   %iota_default_2 : Tensor "i64[1000000][1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1000000,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: npu:0, requires_grad: False})
#   %bitwise_and_scalar_2 : Tensor "i64[1000000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Scalar](args = (%iota_default_2, 63), kwargs = {})
#   return %bitwise_and_scalar_2
triton_unk_fused_index_put_0 = async_compile.triton('triton_unk_fused_index_put_0', '''
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
    size_hints={'x': 1000000}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32'}, 'downcast_args': {'out_ptr0': '*i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [1000000, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 1000000, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_index_put_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_index_put_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 1000000
    x_g_tile0 : tl.constexpr = (1000000) if (1000000) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 1000000
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
        tmp0 = (x0).to(tl.int32)
        tmp1 = tl.full([1], 63, tl.int32)
        tmp2 = tmp0 & tmp1
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t085-performance-results/functional-partitioned-scatter-20260910T054442-vb4ys8_j/on/rank-0/inductor-cache/tmplchizw9z/6s/c6swg3c4hsuclhracbicdftyrl4ltsz5io3dsjavaj66fl3aefbw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
# Source node to ATen node mapping:
#    => full_default_2
# Graph fragment:
#   %full_default_2 : Tensor "f32[32064, 100][100, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32064, 100], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   return %index_put_default_2
triton_unk_fused_index_put_1 = async_compile.triton('triton_unk_fused_index_put_1', '''
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
    size_hints={'x': 3206400}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [3206400, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 3206400, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_index_put_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_index_put_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 3206400
    x_g_tile0 : tl.constexpr = (3206400) if (3206400) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 3206400
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
        tmp0 = tl.full([], 0.0, tl.float32)
        tl.store(out_ptr0 + (x0), tmp0, x0mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t085-performance-results/functional-partitioned-scatter-20260910T054442-vb4ys8_j/on/rank-0/inductor-cache/tmplchizw9z/et/cetwyu5cfhxionagbawhqrgbwdxjjrpf7pprqsfyyotm76wewnkm.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
# Source node to ATen node mapping:
#    => add_tensor_5, sum_dim_int_list_2
# Graph fragment:
#   %view_default_2 : Tensor "f32[64, 501, 100][50100, 100, 1]npu:0" = PlaceHolder[target=view_default_2]
#   %arg0_1 : Tensor "f32[501, 100][100, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %sum_dim_int_list_2 : Tensor "f32[501, 100][100, 1]npu:0" = PlaceHolder[target=sum_dim_int_list_2]
#   %sum_dim_int_list_2 : Tensor "f32[501, 100][100, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_default_2, [0]), kwargs = {})
#   %add_tensor_5 : Tensor "f32[501, 100][100, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %sum_dim_int_list_2), kwargs = {})
#   return %sum_dim_int_list_2,%add_tensor_5
triton_unk_fused_index_put_2 = async_compile.triton('triton_unk_fused_index_put_2', '''
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
    size_hints={'x': 50100, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {'r0_numel': 64}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [50100, {'divisors': [1]}], 'R0_BLOCK_HINT': [64, {'divisors': [1]}]}, 'axis_hints': [{'name': 'x0', 'length': 50100, 'divisor': 1, 'seed': 1}, {'name': 'r0_1', 'length': 64, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_index_put_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused_index_put_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 50100
    r0_numel = 64
    x_g_tile0 : tl.constexpr = (50100) if (50100) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 50100
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
        rbase = tl.arange(0, R0_BLOCK)[:, None]
        r0_base = tl.arange(0, R0_BLOCK)[:, None]
        x0offset = (group_base + i) % x0_blocks * real_block_x0
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        xmask = x0mask
        rbase = r0_base
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        _tmp2 = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (x0 + 50100*r0_1), r0_mask & x0mask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK, real_block_x0])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tmp3
        tmp2 = tl.sum(_tmp2, 0)[None, :]
        tmp4 = tl.load(in_ptr1 + (x0), x0mask)
        tmp5 = tmp4 + tmp2
        tl.store(in_out_ptr0 + (x0), tmp5, x0mask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            buf0 = empty_strided_npu((1000000, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_index_put_0.run(buf0, 1000000, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            # [Provenance debug handles] torch.ops.aten.mul.Tensor:1
            buf1 = torch.ops.aten.mul.Tensor(buf0, 501)
            assert_alignment(buf1, 16, 'torch.ops.aten.mul.Tensor')
            del buf0
            arg2_1 = copy_if_misaligned(arg2_1)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            # [Provenance debug handles] torch.ops.aten.add.Tensor:2
            buf2 = torch.ops.aten.add.Tensor(arg2_1, buf1)
            assert_alignment(buf2, 16, 'torch.ops.aten.add.Tensor')
            del buf1
            buf3 = empty_strided_npu((32064, 100), (100, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_index_put_1.run(buf3, 3206400, stream=raw_stream0)
            arg1_1 = copy_if_misaligned(arg1_1)
            aten.index_put_(buf3, [buf2], arg1_1, True)
            del buf2
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            # [Provenance debug handles] torch.ops.aten.view.default:3
            buf5 = torch.ops.aten.view.default(buf3, [64, 501, 100])
            assert_alignment(buf5, 16, 'torch.ops.aten.view.default')
            arg0_1 = copy_if_misaligned(arg0_1)
            buf6 = empty_strided_npu((501, 100), (100, 1), torch.float32)
            buf7 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_index_put_2.run(buf7, buf5, arg0_1, 50100, 64, stream=raw_stream0)
            del arg0_1
            del buf3
            del buf5
            buf8 = empty_strided_npu((1000000, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_index_put_0.run(buf8, 1000000, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            # [Provenance debug handles] torch.ops.aten.mul.Tensor:4
            buf9 = torch.ops.aten.mul.Tensor(buf8, 501)
            assert_alignment(buf9, 16, 'torch.ops.aten.mul.Tensor')
            del buf8
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            # [Provenance debug handles] torch.ops.aten.add.Tensor:5
            buf10 = torch.ops.aten.add.Tensor(arg2_1, buf9)
            assert_alignment(buf10, 16, 'torch.ops.aten.add.Tensor')
            del buf9
            buf11 = empty_strided_npu((32064, 100), (100, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_index_put_1.run(buf11, 3206400, stream=raw_stream0)
            aten.index_put_(buf11, [buf10], arg1_1, True)
            del buf10
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            # [Provenance debug handles] torch.ops.aten.view.default:6
            buf13 = torch.ops.aten.view.default(buf11, [64, 501, 100])
            assert_alignment(buf13, 16, 'torch.ops.aten.view.default')
            arg3_1 = copy_if_misaligned(arg3_1)
            buf14 = empty_strided_npu((501, 100), (100, 1), torch.float32)
            buf15 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_index_put_2.run(buf15, buf13, arg3_1, 50100, 64, stream=raw_stream0)
            del arg3_1
            del buf11
            del buf13
            buf16 = empty_strided_npu((1000000, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_index_put_0.run(buf16, 1000000, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            # [Provenance debug handles] torch.ops.aten.mul.Tensor:7
            buf17 = torch.ops.aten.mul.Tensor(buf16, 501)
            assert_alignment(buf17, 16, 'torch.ops.aten.mul.Tensor')
            del buf16
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            # [Provenance debug handles] torch.ops.aten.add.Tensor:8
            buf18 = torch.ops.aten.add.Tensor(arg2_1, buf17)
            assert_alignment(buf18, 16, 'torch.ops.aten.add.Tensor')
            del arg2_1
            del buf17
            buf19 = empty_strided_npu((32064, 100), (100, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_index_put_1.run(buf19, 3206400, stream=raw_stream0)
            aten.index_put_(buf19, [buf18], arg1_1, True)
            del arg1_1
            del buf18
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            # [Provenance debug handles] torch.ops.aten.view.default:9
            buf21 = torch.ops.aten.view.default(buf19, [64, 501, 100])
            assert_alignment(buf21, 16, 'torch.ops.aten.view.default')
            arg4_1 = copy_if_misaligned(arg4_1)
            buf22 = empty_strided_npu((501, 100), (100, 1), torch.float32)
            buf23 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.index_put]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_index_put_2.run(buf23, buf21, arg4_1, 50100, 64, stream=raw_stream0)
            del arg4_1
            del buf19
            del buf21
        return (buf7, buf15, buf23, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((501, 100), (100, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((1000000, 100), (100, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((1000000, ), (1, ), device='npu:0', dtype=torch.int64)
    arg3_1 = rand_strided((501, 100), (100, 1), device='npu:0', dtype=torch.float32)
    arg4_1 = rand_strided((501, 100), (100, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
