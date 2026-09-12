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


# kernel path: /home/z50063656/tmp/t096-native-pxf3g41o/adapter/inductor-cache/tmps7x028zu/nf/cnfurrb4h6j5ssneiczlglm6f7xvueubksbllmt4vang2adol55z.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.bitwise_and, aten.ne, aten._to_copy, aten.add, aten.clamp, aten.gt, aten.eq, aten.where, aten.lt, aten.bitwise_or, aten.scalar_tensor]
# Source node to ATen node mapping:
#    => add_tensor, bitwise_and_scalar, bitwise_and_scalar_1, bitwise_and_tensor, bitwise_or_tensor, clamp_max_default, clamp_min_default, convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, eq_scalar, eq_scalar_1, gt_scalar, lt_scalar, ne_scalar, ne_scalar_1, scalar_tensor_default, where_self, where_self_1
# Graph fragment:
#   %view_dtype : Tensor "i32[7][1]npu:0" = PlaceHolder[target=view_dtype]
#   %__rshift___scalar : Tensor "i32[7][1]npu:0" = PlaceHolder[target=__rshift___scalar]
#   %bitwise_and_scalar : Tensor "i32[7][1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.bitwise_and.Scalar](args = (%__rshift___scalar, 255), kwargs = {})
#   %bitwise_and_scalar_1 : Tensor "i32[7][1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.bitwise_and.Scalar](args = (%view_dtype, 8388607), kwargs = {})
#   %ne_scalar : Tensor "b8[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%bitwise_and_scalar_1, 0), kwargs = {})
#   %convert_element_type_default : Tensor "i32[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ne_scalar, torch.int32), kwargs = {})
#   %add_tensor : Tensor "i32[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bitwise_and_scalar, %convert_element_type_default), kwargs = {})
#   %clamp_min_default : Tensor "i32[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_tensor, 0), kwargs = {})
#   %clamp_max_default : Tensor "i32[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_default, 254), kwargs = {})
#   %gt_scalar : Tensor "b8[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%bitwise_and_scalar_1, 4194304), kwargs = {})
#   %convert_element_type_default_1 : Tensor "i32[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_scalar, torch.int32), kwargs = {})
#   %eq_scalar : Tensor "b8[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%bitwise_and_scalar, 0), kwargs = {})
#   %where_self : Tensor "i32[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_scalar, %convert_element_type_default_1, %clamp_max_default), kwargs = {})
#   %lt_scalar : Tensor "b8[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%view_dtype, 0), kwargs = {})
#   %eq_scalar_1 : Tensor "b8[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%bitwise_and_scalar, 255), kwargs = {})
#   %ne_scalar_1 : Tensor "b8[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%bitwise_and_scalar_1, 0), kwargs = {})
#   %bitwise_and_tensor : Tensor "b8[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_scalar_1, %ne_scalar_1), kwargs = {})
#   %bitwise_or_tensor : Tensor "b8[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%lt_scalar, %bitwise_and_tensor), kwargs = {})
#   %scalar_tensor_default : Tensor "i32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (0,), kwargs = {dtype: torch.int32, layout: torch.strided, device: npu:0})
#   %where_self_1 : Tensor "i32[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_tensor, %scalar_tensor_default, %where_self), kwargs = {})
#   %convert_element_type_default_2 : Tensor "u8[7][1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_self_1, torch.uint8), kwargs = {})
#   return %convert_element_type_default_2
triton_unk_fused__to_copy_add_bitwise_and_bitwise_or_clamp_eq_gt_lt_ne_scalar_tensor_where_0 = async_compile.triton('triton_unk_fused__to_copy_add_bitwise_and_bitwise_or_clamp_eq_gt_lt_ne_scalar_tensor_where_0', '''
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
    size_hints={'x': 7}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*i32', 'out_ptr0': '*u8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [7, {'divisors': [1]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 7, 'divisor': 1, 'seed': 1}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__to_copy_add_bitwise_and_bitwise_or_clamp_eq_gt_lt_ne_scalar_tensor_where_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 1, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__to_copy_add_bitwise_and_bitwise_or_clamp_eq_gt_lt_ne_scalar_tensor_where_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 7
    x_g_tile0 : tl.constexpr = (7) if (7) < (XBLOCK) else (XBLOCK)
    x0numel : tl.constexpr = 7
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
        tmp0 = tl.load(in_ptr0 + (x0), x0mask)
        tmp3 = tl.load(in_ptr1 + (x0), x0mask)
        tmp1 = tl.full([1], 0, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp4 = tl.full([1], 255, tl.int32)
        tmp5 = tmp3 & tmp4
        tmp6 = tmp5 == tmp4
        tmp7 = tl.full([1], 8388607, tl.int32)
        tmp8 = tmp0 & tmp7
        tmp9 = tmp8 != tmp1
        tmp10 = tmp6 & tmp9
        tmp11 = tmp2 | tmp10
        tmp12 = tmp5 == tmp1
        tmp13 = tl.full([1], 4194304, tl.int32)
        tmp14 = tmp8 > tmp13
        tmp15 = tmp14.to(tl.int32)
        tmp16 = tmp9.to(tl.int32)
        tmp17 = tmp5 + tmp16
        tmp18 = tl.maximum(tmp17, tmp1)
        tmp19 = tl.full([1], 254, tl.int32)
        tmp20 = tl.minimum(tmp18, tmp19)
        tmp21 = tl.where(tmp12, tmp15, tmp20)
        tmp22 = tl.where(tmp11, tmp1, tmp21)
        tmp23 = tmp22.to(tl.uint8)
        tl.store(out_ptr0 + (x0), tmp23, x0mask)
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
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view]
            # [Provenance debug handles] torch.ops.aten.view.dtype:1
            buf0 = torch.ops.aten.view.dtype(arg0_1, torch.int32)
            assert_alignment(buf0, 16, 'torch.ops.aten.view.dtype')
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.__rshift__]
            # [Provenance debug handles] torch.ops.aten.__rshift__.Scalar:2
            buf1 = torch.ops.aten.__rshift__.Scalar(buf0, 23)
            assert_alignment(buf1, 16, 'torch.ops.aten.__rshift__.Scalar')
            buf2 = empty_strided_npu((7, ), (1, ), torch.uint8)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bitwise_and, aten.ne, aten._to_copy, aten.add, aten.clamp, aten.gt, aten.eq, aten.where, aten.lt, aten.bitwise_or, aten.scalar_tensor]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__to_copy_add_bitwise_and_bitwise_or_clamp_eq_gt_lt_ne_scalar_tensor_where_0.run(buf0, buf1, buf2, 7, stream=raw_stream0)
            del arg0_1
            del buf0
            del buf1
        return (buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((7, ), (1, ), device='npu:0', dtype=torch.float32)
    return [arg0_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
