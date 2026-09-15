# AOT ID: ['99_inference']
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
_frozen_param4 = None  # device(type='npu', index=0) torch.float32 (3, 32) (32, 1) fffc6433c780
_frozen_param5 = None  # device(type='npu', index=0) torch.float32 (32,) (1,) fffc6433c820


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
        arg3_1, = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg3_1 = copy_if_misaligned(arg3_1)
            buf0 = empty_strided_npu((8, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:1
            extern_kernels.addmm(_frozen_param5, reinterpret_tensor(arg3_1, (8, 3), (3, 1), 0), _frozen_param4, alpha=1, beta=1, out=buf0)
            del arg3_1
        return (reinterpret_tensor(buf0, (2, 4, 32), (128, 32, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    global _frozen_param4
    _frozen_param4 = rand_strided((3, 32), (32, 1), device='npu:0', dtype=torch.float32)
    global _frozen_param5
    _frozen_param5 = rand_strided((32, ), (1, ), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((2, 4, 3), (12, 3, 1), device='npu:0', dtype=torch.float32)
    return [arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
