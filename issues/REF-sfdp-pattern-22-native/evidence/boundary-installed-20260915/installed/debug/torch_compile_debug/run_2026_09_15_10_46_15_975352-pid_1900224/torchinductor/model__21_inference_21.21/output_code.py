# AOT ID: ['21_inference']
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


# kernel path: /home/z50063656/tmp/attention-slice-boundary-5k498lwi/installed/inductor-cache/tmpklpkqm3k/ra/cra4isiucplv5yp3wolmn5wub5rfd4lps3z3qefxqpnthf4h573u.py
# Topologically Sorted Source Nodes: [select, unsqueeze, add], Original ATen: [aten.select, aten.unsqueeze, aten.add]
# Source node to ATen node mapping:
#   add => add_5
#   select => select
#   unsqueeze => unsqueeze
# Graph fragment:
#   %arg2_1 : Tensor "f32[1, s21, s67][s21*s67, s67, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %arg6_1 : Tensor "f32[s3, s21, s32][s21*s32, s32, 1]npu:0" = PlaceHolder[target=arg6_1]
#   %select : Tensor "f32[1, s21][s21*s67, s67]npu:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg2_1, -1, %arg3_1), kwargs = {})
#   %unsqueeze : Tensor "f32[1, s21, 1][s21*s67, s67, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, -1), kwargs = {})
#   %add_5 : Tensor "f32[s3, s21, s32][s21*s32, s32, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, %arg6_1), kwargs = {})
#   return %add_5
triton_unk_fused_add_select_unsqueeze_0 = async_compile.triton('triton_unk_fused_add_select_unsqueeze_0', '''
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
    size_hints={'x': 105}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'ks3': 'i64', 'xnumel': 'i32', 'x0numel': 'i32', 'x1numel': 'i32', 'x2numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [-1, -1, -1, {'divisors': [1, 5, 35]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': -1, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': -1, 'divisor': 5, 'seed': 2}, {'name': 'x2', 'length': -1, 'divisor': 35, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_select_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'npu_num_x_nodes': 3, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_select_unsqueeze_0(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, xnumel, x0numel, x1numel, x2numel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    x_g_tile0 : tl.constexpr = (5) if (5) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (7) if (7) < (x_g_rem1) else (x_g_rem1)
    x_g_rem2 : tl.constexpr = ((x_g_rem1) // x_g_tile1) if ((x_g_rem1) // x_g_tile1) > 1 else 1
    x_g_tile2 : tl.constexpr = (3) if (3) < (x_g_rem2) else (x_g_rem2)
    real_block_x0 : tl.constexpr = (((x_g_tile0) + 7) // 8) * 8
    real_block_x1 : tl.constexpr = x_g_tile1
    real_block_x2 : tl.constexpr = x_g_tile2
    x0_blocks = (x0numel + real_block_x0 - 1) // real_block_x0
    x1_blocks = (x1numel + real_block_x1 - 1) // real_block_x1
    x2_blocks = (x2numel + real_block_x2 - 1) // real_block_x2
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
        tmp0 = tl.load(in_ptr0 + (ks0 + ks1*x1), x1mask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + ks3*x1 + ks2*ks3*x2), x0mask & x1mask & x2mask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + (x0 + ks3*x1 + ks2*ks3*x2), tmp2, x0mask & x1mask & x2mask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
        args.clear()
        s21 = arg0_1
        s67 = arg1_1
        s26 = arg3_1
        s3 = arg4_1
        s32 = arg5_1
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg2_1 = copy_if_misaligned(arg2_1)
            arg6_1 = copy_if_misaligned(arg6_1)
            buf0 = empty_strided_npu((s3, s21, s32), (s21*s32, s32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [select, unsqueeze, add], Original ATen: [aten.select, aten.unsqueeze, aten.add]
            triton_unk_fused_add_select_unsqueeze_0_xnumel = s21*s3*s32
            triton_unk_fused_add_select_unsqueeze_0_x0numel = s32
            triton_unk_fused_add_select_unsqueeze_0_x1numel = s21
            triton_unk_fused_add_select_unsqueeze_0_x2numel = s3
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_select_unsqueeze_0.run(arg2_1, arg6_1, buf0, s26, s67, s21, s32, triton_unk_fused_add_select_unsqueeze_0_xnumel, triton_unk_fused_add_select_unsqueeze_0_x0numel, triton_unk_fused_add_select_unsqueeze_0_x1numel, triton_unk_fused_add_select_unsqueeze_0_x2numel, stream=raw_stream0)
            del arg2_1
            del arg6_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 7
    arg1_1 = 4
    arg2_1 = rand_strided((1, 7, 4), (28, 4, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = 3
    arg4_1 = 3
    arg5_1 = 5
    arg6_1 = rand_strided((3, 7, 5), (35, 5, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
