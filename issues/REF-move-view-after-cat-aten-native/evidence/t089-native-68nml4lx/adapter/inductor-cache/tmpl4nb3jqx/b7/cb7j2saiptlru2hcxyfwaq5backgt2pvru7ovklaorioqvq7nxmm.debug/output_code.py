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


# kernel path: /home/z50063656/tmp/t089-native-68nml4lx/adapter/inductor-cache/tmpl4nb3jqx/eh/cehmkkab2rg44gp4d4ofcqeijrsliqqau4e2xg5llfvzltgwgzxk.py
# Topologically Sorted Source Nodes: [split_with_sizes_default, view_default, , cat_default, cat_default_1], Original ATen: [aten.split_with_sizes, aten.view, aten.cat]
# Source node to ATen node mapping:
#    => permute_default
#   cat_default => reshape_default
#   cat_default_1 => cat_default_1
#   split_with_sizes_default => split_with_sizes_default
#   view_default => view
# Graph fragment:
#   %arg0_1 : Tensor "f32[7, 8, 96][768, 96, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %split_with_sizes_default : [num_users=7] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%arg0_1, [1, 1, 1, 1, 1, 1, 1]), kwargs = {dim: 0})
#   %view : Tensor "f32[8, 96][96, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [8, 96]), kwargs = {})
#   %permute_default : Tensor "f32[8, 7, 96][96, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg0_1, [1, 0, 2]), kwargs = {})
#   %reshape_default : Tensor "f32[8, 672][672, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_default, [8, 672]), kwargs = {})
#   %cat_default_1 : Tensor "f32[8, 768][768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view, %reshape_default],), kwargs = {dim: 1})
#   return %buf0
triton_unk_fused_cat_split_with_sizes_view_0 = async_compile.triton('triton_unk_fused_cat_split_with_sizes_view_0', '''
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
    size_hints={'x': 5376}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [8, 96, 7, {'divisors': [672, 1, 96]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x1', 'length': 8, 'divisor': 672, 'seed': 2}, {'name': 'x2', 'length': 96, 'divisor': 1, 'seed': 1}, {'name': 'x3', 'length': 7, 'divisor': 96, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_cat_split_with_sizes_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_cat_split_with_sizes_view_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 5376
    x_g_tile0 : tl.constexpr = (96) if (96) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (7) if (7) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (8) if (8) < (x_g_rem2) else (x_g_rem2)
    x1numel : tl.constexpr = 8
    x2numel : tl.constexpr = 96
    x3numel : tl.constexpr = 7
    real_block_x1 : tl.constexpr = x_g_tile2
    real_block_x2 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x3 : tl.constexpr = x_g_tile1
    x1_blocks : tl.constexpr = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks : tl.constexpr = (x2numel + real_block_x2 - 1) // real_block_x2
    x3_blocks : tl.constexpr = (x3numel + real_block_x3 - 1) // real_block_x3
    x_cumblk_1 = x1_blocks
    x_cumblk_2 = x1_blocks * x2_blocks
    total_blocks = x1_blocks * x2_blocks * x3_blocks
    group_size = total_blocks // total_thread
    group_tail = total_blocks % total_thread
    if group_id < group_tail:
        group_size = group_size + 1
        group_base = group_id * group_size
    else:
        group_base = group_id * group_size + group_tail
    for i in range(group_size):
        x1offset = (group_base + i) % x1_blocks * real_block_x1
        x2offset = (group_base + i) // x_cumblk_1 % x2_blocks * real_block_x2
        x3offset = (group_base + i) // x_cumblk_2 % x3_blocks * real_block_x3
        x1index = x1offset + tl.arange(0, real_block_x1)[:, None, None]
        x1 = x1index
        x1mask = x1index < x1numel
        x2index = x2offset + tl.arange(0, real_block_x2)[None, None, :]
        x2 = x2index
        x2mask = x2index < x2numel
        x3index = x3offset + tl.arange(0, real_block_x3)[None, :, None]
        x3 = x3index
        x3mask = x3index < x3numel
        xmask = x1mask & x2mask & x3mask
        x0 = x2 + 96*x3
        x0mask = x2mask & x3mask
        tmp0 = tl.load(in_ptr0 + (x2 + 96*x1 + 768*x3), x1mask & x2mask & x3mask)
        tl.store(out_ptr0 + (x2 + 96*x3 + 672*x1), tmp0, x1mask & x2mask & x3mask)
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
            buf0 = empty_strided_npu((8, 672), (672, 1), torch.float32)
            # Topologically Sorted Source Nodes: [split_with_sizes_default, view_default, , cat_default, cat_default_1], Original ATen: [aten.split_with_sizes, aten.view, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_cat_split_with_sizes_view_0.run(arg0_1, buf0, 5376, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [split_with_sizes_default, view_default, , cat_default, cat_default_1], Original ATen: [aten.split_with_sizes, aten.view, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:1
            buf1 = torch.ops.aten.cat.default([reinterpret_tensor(arg0_1, (8, 96), (96, 1), 0), buf0], 1)
            assert_alignment(buf1, 16, 'torch.ops.aten.cat.default')
            del buf0
            # Topologically Sorted Source Nodes: [split_with_sizes_default, view_default, cat], Original ATen: [aten.split_with_sizes, aten.view, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:2
            buf2 = torch.ops.aten.cat.default([reinterpret_tensor(arg0_1, (8, 96), (96, 1), 0), buf1], 1)
            assert_alignment(buf2, 16, 'torch.ops.aten.cat.default')
            del arg0_1
            del buf1
        return (buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((7, 8, 96), (768, 96, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
