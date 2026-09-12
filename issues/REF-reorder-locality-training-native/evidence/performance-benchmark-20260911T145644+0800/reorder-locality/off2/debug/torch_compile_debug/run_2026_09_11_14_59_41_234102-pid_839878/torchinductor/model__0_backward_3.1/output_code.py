# AOT ID: ['0_backward']
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


# kernel path: /home/z50063656/tmp/t087-npu-results/benchmark-20260911T145644+0800/reorder-locality/off2/inductor-cache/tmpnu1er1iq/oh/cohjayhbiv5pivvhjwbokdgo6xc2mdskd6ylfau3v7m5siuf2ic2.py
# Topologically Sorted Source Nodes: [unsqueeze, expand, a, sin, mul_2, b, cos, mul_3, sin_1, neg, mul_4, cos_1, mul_5, mul_6, mul_7, add_1, add_2, sub, mul_8, mul_9, matmul_backward, mul_10, sub_1, mul_11, matmul_backward_1], Original ATen: [aten.unsqueeze, aten.expand, aten.tanh, aten.sin, aten.mul, aten.sigmoid, aten.cos, aten.neg, aten.add, aten.sigmoid_backward, aten.matmul_backward, aten.tanh_backward]
# Source node to ATen node mapping:
#   a => tanh
#   add_1 => add_1
#   add_2 => add_2
#   b => sigmoid
#   cos => cos
#   cos_1 => cos_1
#   expand => expand
#   matmul_backward => matmul_backward
#   matmul_backward_1 => matmul_backward_1
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   neg => neg
#   sin => sin
#   sin_1 => sin_1
#   sub => sub
#   sub_1 => sub_1
#   unsqueeze => unsqueeze
# Graph fragment:
#   %tangents_1 : Tensor "f32[8][1]npu:0" = PlaceHolder[target=tangents_1]
#   %mm : Tensor "f32[8, 16][16, 1]npu:0" = PlaceHolder[target=mm]
#   %mm_1 : Tensor "f32[8, 16][16, 1]npu:0" = PlaceHolder[target=mm_1]
#   %unsqueeze : Tensor "f32[8, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%tangents_1, 1), kwargs = {})
#   %expand : Tensor "f32[8, 16][1, 0]npu:0"[num_users=4] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [8, 16]), kwargs = {})
#   %tanh : Tensor "f32[8, 16][16, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.tanh.default](args = (%mm,), kwargs = {})
#   %sin : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%tanh,), kwargs = {})
#   %mul_2 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %sin), kwargs = {})
#   %sigmoid : Tensor "f32[8, 16][16, 1]npu:0"[num_users=5] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mm_1,), kwargs = {})
#   %cos : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%sigmoid,), kwargs = {})
#   %mul_3 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %cos), kwargs = {})
#   %sin_1 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%sigmoid,), kwargs = {})
#   %neg : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sin_1,), kwargs = {})
#   %mul_4 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %neg), kwargs = {})
#   %cos_1 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%tanh,), kwargs = {})
#   %mul_5 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %cos_1), kwargs = {})
#   %mul_6 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %tanh), kwargs = {})
#   %mul_7 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %sigmoid), kwargs = {})
#   %add_1 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_7), kwargs = {})
#   %add_2 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_6), kwargs = {})
#   %sub : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %mul_8 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %sub), kwargs = {})
#   %mul_9 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %mul_8), kwargs = {})
#   %matmul_backward : [num_users=1] = call_function[target=torch.ops.aten.matmul_backward.default](args = (%mul_9, %primals_2, %primals_3, [False, True]), kwargs = {})
#   %mul_10 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %tanh), kwargs = {})
#   %sub_1 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_10), kwargs = {})
#   %mul_11 : Tensor "f32[8, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sub_1), kwargs = {})
#   %matmul_backward_1 : [num_users=1] = call_function[target=torch.ops.aten.matmul_backward.default](args = (%mul_11, %primals_2, %primals_1, [False, True]), kwargs = {})
#   return %buf0,%buf4
triton_unk_fused_add_cos_expand_matmul_backward_mul_neg_sigmoid_sigmoid_backward_sin_tanh_tanh_backward_unsqueeze_0 = async_compile.triton('triton_unk_fused_add_cos_expand_matmul_backward_mul_neg_sigmoid_sigmoid_backward_sin_tanh_tanh_backward_unsqueeze_0', '''
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
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'block_hints': {'XBLOCK_HINT': [16, 8, {'divisors': [1, 16]}], 'R0_BLOCK_HINT': [{'divisors': []}]}, 'axis_hints': [{'name': 'x0', 'length': 16, 'divisor': 1, 'seed': 1}, {'name': 'x1', 'length': 8, 'divisor': 16, 'seed': 2}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_cos_expand_matmul_backward_mul_neg_sigmoid_sigmoid_backward_sin_tanh_tanh_backward_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'npu_num_x_nodes': 2, 'npu_rsplit_partial': False, 'backend_hash': '573767764059784D1B071A77E46541176356CEAB012EDE3A29BDD4D6B9E2A472', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_cos_expand_matmul_backward_mul_neg_sigmoid_sigmoid_backward_sin_tanh_tanh_backward_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    total_thread = 48
    group_id = tl.program_id(0)
    xnumel = 128
    x_g_tile0 : tl.constexpr = (16) if (16) < (XBLOCK) else (XBLOCK)
    x_g_rem1 : tl.constexpr = ((XBLOCK) // x_g_tile0) if ((XBLOCK) // x_g_tile0) > 1 else 1
    x_g_tile1 : tl.constexpr = (8) if (8) < (x_g_rem1) else (x_g_rem1)
    x0numel : tl.constexpr = 16
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
        tmp0 = tl.load(in_ptr0 + (x1), x1mask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + 16*x1), x0mask & x1mask)
        tmp5 = tl.load(in_ptr2 + (x0 + 16*x1), x0mask & x1mask)
        tmp2 = libdevice.tanh(tmp1)
        tmp3 = tl_math.sin(tmp2)
        tmp4 = tmp0 * tmp3
        tmp6 = tl.sigmoid(tmp5)
        tmp7 = tl_math.sin(tmp6)
        tmp8 = -tmp7
        tmp9 = tmp4 * tmp8
        tmp10 = tmp0 * tmp2
        tmp11 = tmp9 + tmp10
        tmp12 = tl.full([1], 1.0, tl.float32)
        tmp13 = tmp12 - tmp6
        tmp14 = tmp6 * tmp13
        tmp15 = tmp11 * tmp14
        tmp16 = tl_math.cos(tmp6)
        tmp17 = tmp0 * tmp16
        tmp18 = tl_math.cos(tmp2)
        tmp19 = tmp17 * tmp18
        tmp20 = tmp0 * tmp6
        tmp21 = tmp19 + tmp20
        tmp22 = tmp2 * tmp2
        tmp23 = tmp12 - tmp22
        tmp24 = tmp21 * tmp23
        tl.store(out_ptr0 + (x0 + 16*x1), tmp15, x0mask & x1mask)
        tl.store(out_ptr1 + (x0 + 16*x1), tmp24, x0mask & x1mask)
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
        primals_1, primals_2, primals_3, mm, mm_1, tangents_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            tangents_1 = copy_if_misaligned(tangents_1)
            buf0 = empty_strided_npu((8, 16), (16, 1), torch.float32)
            buf4 = empty_strided_npu((8, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [unsqueeze, expand, a, sin, mul_2, b, cos, mul_3, sin_1, neg, mul_4, cos_1, mul_5, mul_6, mul_7, add_1, add_2, sub, mul_8, mul_9, matmul_backward, mul_10, sub_1, mul_11, matmul_backward_1], Original ATen: [aten.unsqueeze, aten.expand, aten.tanh, aten.sin, aten.mul, aten.sigmoid, aten.cos, aten.neg, aten.add, aten.sigmoid_backward, aten.matmul_backward, aten.tanh_backward]
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused_add_cos_expand_matmul_backward_mul_neg_sigmoid_sigmoid_backward_sin_tanh_tanh_backward_unsqueeze_0.run(tangents_1, mm, mm_1, buf0, buf4, 128, stream=raw_stream0)
            del mm
            del mm_1
            del tangents_1
            # Topologically Sorted Source Nodes: [unsqueeze, expand, a, sin, mul_2, b, sin_1, neg, mul_4, mul_6, add_2, sub, mul_8, mul_9, matmul_backward], Original ATen: [aten.unsqueeze, aten.expand, aten.tanh, aten.sin, aten.mul, aten.sigmoid, aten.neg, aten.add, aten.sigmoid_backward, aten.matmul_backward]
            # [Provenance debug handles] torch.ops.aten.matmul_backward.default:1
            buf1 = torch.ops.aten.matmul_backward.default(buf0, primals_2, primals_3, [False, True])
            del buf0
            del primals_3
            buf3 = buf1[1]
            assert_alignment(buf3, 16, 'torch.ops.aten.matmul_backward.default')
            del buf1
            # Topologically Sorted Source Nodes: [unsqueeze, expand, a, b, cos, mul_3, cos_1, mul_5, mul_7, add_1, mul_10, sub_1, mul_11, matmul_backward_1], Original ATen: [aten.unsqueeze, aten.expand, aten.tanh, aten.sigmoid, aten.cos, aten.mul, aten.add, aten.tanh_backward, aten.matmul_backward]
            # [Provenance debug handles] torch.ops.aten.matmul_backward.default:2
            buf5 = torch.ops.aten.matmul_backward.default(buf4, primals_2, primals_1, [False, True])
            del buf4
            del primals_1
            del primals_2
            buf7 = buf5[1]
            assert_alignment(buf7, 16, 'torch.ops.aten.matmul_backward.default')
            del buf5
        return (buf7, None, buf3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((16, 16), (16, 1), device='npu:0', dtype=torch.float32)
    primals_2 = rand_strided((8, 16), (16, 1), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((16, 16), (16, 1), device='npu:0', dtype=torch.float32)
    mm = rand_strided((8, 16), (16, 1), device='npu:0', dtype=torch.float32)
    mm_1 = rand_strided((8, 16), (16, 1), device='npu:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, ), (1, ), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, mm, mm_1, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
