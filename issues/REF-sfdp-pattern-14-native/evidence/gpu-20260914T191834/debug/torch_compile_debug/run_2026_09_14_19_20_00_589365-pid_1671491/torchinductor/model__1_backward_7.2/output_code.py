# AOT ID: ['1_backward']
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
        permute_default, permute_default_1, permute_default_2, slice_tensor, getitem, getitem_1, getitem_2, getitem_3, tangents_1 = args
        args.clear()
        assert_size_stride_grouped((tangents_1, permute_default, permute_default_1, permute_default_2, slice_tensor, getitem, getitem_1, getitem_2, getitem_3), ((4, 16, 2, 32), (4, 16, 2, 32), (4, 16, 2, 32), (4, 16, 2, 32), (2, 2), (4, 16, 2, 32), (4, 16, 32), (), ()), ((1024, 64, 32, 1), (1024, 32, 512, 1), (1024, 32, 512, 1), (1024, 32, 512, 1), (8, 1), (1024, 32, 512, 1), (512, 32, 1), (), ()), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            tangents_1 = copy_if_misaligned(tangents_1)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.expand, aten._scaled_dot_product_efficient_attention_backward]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention_backward.default:1
            buf0 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(tangents_1, permute_default, permute_default_1, permute_default_2, reinterpret_tensor(slice_tensor, (4, 16, 2, 2), (0, 0, 8, 1), 0), getitem, getitem_1, getitem_2, getitem_3, 0.0, [True, True, True, False], scale=0.3333333333333333)
            del getitem
            del getitem_1
            del getitem_2
            del getitem_3
            del permute_default
            del permute_default_1
            del permute_default_2
            del slice_tensor
            del tangents_1
            buf1 = buf0[0]
            assert_tensor_metadata(buf1, (4, 16, 2, 32), (1024, 32, 512, 1), torch.float32, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf2 = buf0[1]
            assert_tensor_metadata(buf2, (4, 16, 2, 32), (1024, 32, 512, 1), torch.float32, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf2, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf3 = buf0[2]
            assert_tensor_metadata(buf3, (4, 16, 2, 32), (1024, 32, 512, 1), torch.float32, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf3, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf0
        return (reinterpret_tensor(buf2, (4, 2, 16, 32), (1024, 512, 32, 1), 0), reinterpret_tensor(buf1, (4, 2, 16, 32), (1024, 512, 32, 1), 0), reinterpret_tensor(buf3, (4, 2, 16, 32), (1024, 512, 32, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    permute_default = rand_strided((4, 16, 2, 32), (1024, 32, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_default_1 = rand_strided((4, 16, 2, 32), (1024, 32, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_default_2 = rand_strided((4, 16, 2, 32), (1024, 32, 512, 1), device='cuda:0', dtype=torch.float32)
    slice_tensor = rand_strided((2, 2), (8, 1), device='cuda:0', dtype=torch.float32)
    getitem = rand_strided((4, 16, 2, 32), (1024, 32, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((4, 16, 32), (512, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_3 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_1 = rand_strided((4, 16, 2, 32), (1024, 64, 32, 1), device='cuda:0', dtype=torch.float32)
    return [permute_default, permute_default_1, permute_default_2, slice_tensor, getitem, getitem_1, getitem_2, getitem_3, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
