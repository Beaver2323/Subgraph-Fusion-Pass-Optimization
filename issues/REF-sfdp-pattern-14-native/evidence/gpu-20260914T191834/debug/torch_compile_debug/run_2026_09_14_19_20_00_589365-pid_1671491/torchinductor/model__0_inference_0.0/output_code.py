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
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.runtime_utils import assert_tensor_metadata

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


# kernel path: /data/z50063656/tmp/t104-reference-results/reference-20260914T191834+0800-0x3nqs19/scratch_cache/REF-sfdp-pattern-14-native/tmpxhhk0h26/hm/chmj5fw5d3evjzxodokhh3let5uqdcypdbqi74dnf5ukvjful3pd.py
# Topologically Sorted Source Nodes: [, tril, ones, logical_not, masked_fill], Original ATen: [aten.transpose, aten.tril, aten.ones, aten.logical_not, aten.masked_fill, aten._to_copy, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#    => _scaled_dot_product_efficient_attention_default, constant_pad_nd_default, convert_element_type_default, expand_default, permute_default, permute_default_1, permute_default_2, slice_tensor
#   logical_not => logical_not
#   masked_fill => full_default_1, where
#   ones => full_default
#   tril => iota, iota_1, le, logical_and, sub, unsqueeze, unsqueeze_1
# Graph fragment:
#   %permute_default : Tensor "f32[4, 16, 2, 32][1024, 32, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg1_1, [0, 2, 1, 3]), kwargs = {})
#   %permute_default_1 : Tensor "f32[4, 16, 2, 32][1024, 32, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg0_1, [0, 2, 1, 3]), kwargs = {})
#   %permute_default_2 : Tensor "f32[4, 16, 2, 32][1024, 32, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg2_1, [0, 2, 1, 3]), kwargs = {})
#   %iota : Tensor "i64[2][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (2,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -2), kwargs = {})
#   %iota_1 : Tensor "i64[2][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (2,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_1 : Tensor "i64[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_1, -1), kwargs = {})
#   %sub : Tensor "i64[2, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %le : Tensor "b8[2, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub, 0), kwargs = {})
#   %full_default : Tensor "b8[2, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 2], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %logical_and : Tensor "b8[2, 2][2, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.logical_and.default](args = (%le, %full_default), kwargs = {})
#   %logical_not : Tensor "b8[2, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and,), kwargs = {})
#   %full_default_1 : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "b8[2, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not, %full_default_1, %logical_and), kwargs = {})
#   %convert_element_type_default : Tensor "f32[2, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %constant_pad_nd_default : Tensor "f32[2, 8][8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%convert_element_type_default, [0, 6], 0.0), kwargs = {})
#   %slice_tensor : Tensor "f32[2, 2][8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd_default, -1, 0, 2), kwargs = {})
#   %expand_default : Tensor "f32[4, 16, 2, 2][0, 0, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_tensor, [4, 16, 2, 2]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_default : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_default, %permute_default_1, %permute_default_2, %expand_default, False), kwargs = {scale: 0.3333333333333333})
#   return %buf0
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_constant_pad_nd_expand_logical_not_masked_fill_ones_slice_transpose_tril_0 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_constant_pad_nd_expand_logical_not_masked_fill_ones_slice_transpose_tril_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_constant_pad_nd_expand_logical_not_masked_fill_ones_slice_transpose_tril_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 16}, 'backend_hash': 'CF1D5F24995550CEF98AF7C0CB7BFF1113EADCE373F8996A3D8B53A7B338339C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_constant_pad_nd_expand_logical_not_masked_fill_ones_slice_transpose_tril_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    tmp0 = (x0).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x0).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 2, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = (((-1)*x1) + (x0)).to(tl.int64)
    tmp8 = (tmp7).to(tl.int64)
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = tmp8 <= tmp9
    tmp11 = tl.full([1], True, tl.int1)
    tmp12 = tmp10 & tmp11
    tmp13 = tmp12 == 0
    tmp14 = tl.where(tmp13, tmp11, tmp12)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp6, tmp15, tmp16)
    tmp18 = tmp0 >= tmp5
    tmp19 = tl.full([1], 8, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.full([1], 0.0, tl.float32)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp18, tmp21, tmp22)
    tmp24 = tl.where(tmp6, tmp17, tmp23)
    tl.store(out_ptr0 + (x0 + 8*x1), tmp24, xmask)
''', device_str='cuda')


# kernel path: /data/z50063656/tmp/t104-reference-results/reference-20260914T191834+0800-0x3nqs19/scratch_cache/REF-sfdp-pattern-14-native/tmpxhhk0h26/6g/c6gth3fzv4yihcvaicp3sijb6j3omfbikrq4t627ojt77tiy73cy.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view]
# Source node to ATen node mapping:
#    => getitem
# Graph fragment:
#   %buf2 : Tensor "f32[4, 16, 2, 32][1024, 32, 512, 1]cuda:0" = PlaceHolder[target=buf2]
#   %getitem : Tensor "f32[4, 16, 2, 32][1024, 32, 512, 1]cuda:0"[num_users=1] = call_function[target=operator.getitem](args = (%_scaled_dot_product_efficient_attention_default, 0), kwargs = {})
#   return %getitem
triton_poi_fused_view_1 = async_compile.triton('triton_poi_fused_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 49152}, 'backend_hash': 'CF1D5F24995550CEF98AF7C0CB7BFF1113EADCE373F8996A3D8B53A7B338339C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 2)
    x2 = ((xindex // 64) % 16)
    x3 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 512*x1 + 1024*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


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
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 1, 2, 2), (0, 0, 8, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, tril, ones, logical_not, masked_fill], Original ATen: [aten.transpose, aten.tril, aten.ones, aten.logical_not, aten.masked_fill, aten._to_copy, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_constant_pad_nd_expand_logical_not_masked_fill_ones_slice_transpose_tril_0:1
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_constant_pad_nd_expand_logical_not_masked_fill_ones_slice_transpose_tril_0.run(buf0, 4, stream=raw_stream0)
            assert_size_stride_grouped((arg1_1, arg0_1, arg2_1), ((4, 2, 16, 32), (4, 2, 16, 32), (4, 2, 16, 32)), ((1024, 512, 32, 1), (1024, 512, 32, 1), (1024, 512, 32, 1)), 'input')
            arg1_1 = copy_if_misaligned(arg1_1)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            # Topologically Sorted Source Nodes: [, tril, ones, logical_not, masked_fill], Original ATen: [aten.transpose, aten.tril, aten.ones, aten.logical_not, aten.masked_fill, aten._to_copy, aten.constant_pad_nd, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:2
            buf1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(arg1_1, (4, 16, 2, 32), (1024, 32, 512, 1), 0), reinterpret_tensor(arg0_1, (4, 16, 2, 32), (1024, 32, 512, 1), 0), reinterpret_tensor(arg2_1, (4, 16, 2, 32), (1024, 32, 512, 1), 0), reinterpret_tensor(buf0, (4, 16, 2, 2), (0, 0, 8, 1), 0), False, scale=0.3333333333333333)
            del arg0_1
            del arg1_1
            del arg2_1
            del buf0
            buf2 = buf1[0]
            assert_tensor_metadata(buf2, (4, 16, 2, 32), (1024, 32, 512, 1), torch.float32, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf2, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf1
            buf6 = empty_strided_cuda((4, 16, 2, 32), (1024, 64, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view]
            # [Provenance debug handles] triton_poi_fused_view_1:3
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_view_1.run(buf2, buf6, 4096, stream=raw_stream0)
            del buf2
        return (buf6, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((4, 2, 16, 32), (1024, 512, 32, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 2, 16, 32), (1024, 512, 32, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 2, 16, 32), (1024, 512, 32, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
