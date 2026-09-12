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
        arg0_1, arg1_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            # Topologically Sorted Source Nodes: [cat_default], Original ATen: [aten.cat]
            # [Provenance debug handles] torch.ops.aten.view.default:1
            buf0 = torch.ops.aten.view.default(arg0_1, (1024, 768))
            assert_alignment(buf0, 16, 'torch.ops.aten.view.default')
            # Topologically Sorted Source Nodes: [select_int, select_int_1, select_int_2, select_int_3, select_int_4, cat_default_1], Original ATen: [aten.select, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:2
            buf1 = torch.ops.aten.cat.default([reinterpret_tensor(arg0_1, (1024, 128), (768, 1), 0), reinterpret_tensor(arg0_1, (1024, 128), (768, 1), 128), reinterpret_tensor(arg0_1, (1024, 128), (768, 1), 256), reinterpret_tensor(arg0_1, (1024, 128), (768, 1), 384), reinterpret_tensor(arg0_1, (1024, 128), (768, 1), 512)], 1)
            assert_alignment(buf1, 16, 'torch.ops.aten.cat.default')
            # Topologically Sorted Source Nodes: [select_int, select_int_2, select_int_4, cat_default_2], Original ATen: [aten.select, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:3
            buf2 = torch.ops.aten.cat.default([reinterpret_tensor(arg0_1, (1024, 128), (768, 1), 0), reinterpret_tensor(arg0_1, (1024, 128), (768, 1), 256), reinterpret_tensor(arg0_1, (1024, 128), (768, 1), 512)], 1)
            assert_alignment(buf2, 16, 'torch.ops.aten.cat.default')
            del arg0_1
            arg1_1 = copy_if_misaligned(arg1_1)
            # Topologically Sorted Source Nodes: [select_int_6, select_int_7, select_int_8, select_int_9, select_int_10, cat_default_3], Original ATen: [aten.select, aten.cat]
            # [Provenance debug handles] torch.ops.aten.cat.default:4
            buf3 = torch.ops.aten.cat.default([reinterpret_tensor(arg1_1, (1024, 128), (768, 1), 0), reinterpret_tensor(arg1_1, (1024, 128), (768, 1), 128), reinterpret_tensor(arg1_1, (1024, 128), (768, 1), 256), reinterpret_tensor(arg1_1, (1024, 128), (768, 1), 384), reinterpret_tensor(arg1_1, (1024, 128), (768, 1), 512)], 1)
            assert_alignment(buf3, 16, 'torch.ops.aten.cat.default')
            del arg1_1
        return (buf0, buf1, buf2, buf3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1024, 6, 128), (768, 128, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((1024, 6, 128), (768, 128, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
