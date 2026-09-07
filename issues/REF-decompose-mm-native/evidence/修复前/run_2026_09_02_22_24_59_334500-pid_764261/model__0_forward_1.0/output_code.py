# AOT ID: ['0_forward']
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


# kernel path: /home/z50063656/tmp/t077-cache-mm-isolate0/tmp3vpxo1n2/ov/cov5nscckqr4sd2cpt37m77lbm7mmjr2o7afhygtbd75dofcr54u.py
# Topologically Sorted Source Nodes: [mm], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   mm => mm
# Graph fragment:
#   %primals_1 : Tensor "f32[2048, 2][2, 1]npu:0" = PlaceHolder[target=primals_1]
#   %mm : Tensor "f32[2048, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%primals_1, %primals_2), kwargs = {})
#   return %buf0
triton_unk_fused_mm_0 = async_compile.triton('triton_unk_fused_mm_0', '''
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
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [2, 4096, {'divisors': [1, 2]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 2, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 4096, 'divisor': 2, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_mm_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_mm_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 8192
    x_g_tile0 : tl.constexpr = (2) if (2) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (4096) if (4096) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 2
    x1numel : tl.constexpr = 4096
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
        tmp0 = tl.load(in_ptr0 + (x1), x1mask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (x0 + 2*x1), tmp0, x0mask & x1mask)
''', device_str='npu')


# kernel path: /home/z50063656/tmp/t077-cache-mm-isolate0/tmp3vpxo1n2/ol/colb5j43rzwhsnsqsz4b3edcqcmxu6blh4hogtsovckhb7jacaau.py
# Topologically Sorted Source Nodes: [mm], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   mm => mm
# Graph fragment:
#   %buf0 : Tensor "f32[2048, 2, 2][4, 2, 1]npu:0" = PlaceHolder[target=buf0]
#   %primals_2 : Tensor "f32[2, 2][2, 1]npu:0" = PlaceHolder[target=primals_2]
#   %mm : Tensor "f32[2048, 2][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%primals_1, %primals_2), kwargs = {})
#   return %mm
triton_unk_fused_mm_1 = async_compile.triton('triton_unk_fused_mm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [2, 2048, {'divisors': [1, 2]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 2, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 2048, 'divisor': 2, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_mm_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_mm_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 4096
    x_g_tile0 : tl.constexpr = (2) if (2) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (2048) if (2048) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 2
    x1numel : tl.constexpr = 2048
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
        tmp0 = tl.load(in_ptr0 + (x0 + 4*x1), x0mask & x1mask)
        tmp1 = tl.load(in_ptr1 + (x0), x0mask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr0 + (2 + x0 + 4*x1), x0mask & x1mask)
        tmp4 = tl.load(in_ptr1 + (2 + x0), x0mask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 * tmp4
        tmp6 = tmp2 + tmp5
        tl.store(out_ptr0 + (x0 + 2*x1), tmp6, x0mask & x1mask)
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
        primals_1, primals_2 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            primals_1 = copy_if_misaligned(primals_1)
            buf0 = empty_strided_npu((2048, 2, 2), (4, 2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mm], Original ATen: [aten.mm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_mm_0.run(primals_1, buf0, 8192, stream=raw_stream0)
            primals_2 = copy_if_misaligned(primals_2)
            buf1 = empty_strided_npu((2048, 2), (2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mm], Original ATen: [aten.mm]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_mm_1.run(buf0, primals_2, buf1, 4096, stream=raw_stream0)
            del buf0
        return (buf1, reinterpret_tensor(primals_1, (2, 2048), (1, 2), 0), reinterpret_tensor(primals_2, (2, 2), (1, 2), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((2048, 2), (2, 1), device='npu:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 2), (2, 1), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_2]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
