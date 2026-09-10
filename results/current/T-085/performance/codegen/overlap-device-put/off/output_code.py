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
            # Topologically Sorted Source Nodes: [exchanged], Original ATen: [_c10d_functional.all_to_all_single]
            # [Provenance debug handles] torch.ops._c10d_functional.all_to_all_single.default:1
            buf0 = torch.ops._c10d_functional.all_to_all_single.default(arg0_1, [4, 4], [4, 4], 't085-default')
            assert_alignment(buf0, 16, 'torch.ops._c10d_functional.all_to_all_single.default')
            # Topologically Sorted Source Nodes: [exchanged_1], Original ATen: [_c10d_functional.wait_tensor]
            # [Provenance debug handles] torch.ops._c10d_functional.wait_tensor.default:2
            buf1 = torch.ops._c10d_functional.wait_tensor.default(buf0)
            assert_alignment(buf1, 16, 'torch.ops._c10d_functional.wait_tensor.default')
            del buf0
            # Topologically Sorted Source Nodes: [view_1, output_splits], Original ATen: [aten.view, aten.sum]
            # [Provenance debug handles] torch.ops.aten.sum.dim_IntList:3
            buf2 = torch.ops.aten.sum.dim_IntList(reinterpret_tensor(buf1, (2, 4), (4, 1), 0), [1])
            assert_alignment(buf2, 16, 'torch.ops.aten.sum.dim_IntList')
            del buf1
        buf3 = empty_strided_cpu((2, ), (1, ), torch.int64)
        buf3.copy_(buf2, False)
        del buf2
        # Topologically Sorted Source Nodes: [getitem, item], Original ATen: [aten.select, aten._local_scalar_dense]
        # [Provenance debug handles] torch.ops.aten._local_scalar_dense.default:4
        buf4 = torch.ops.aten._local_scalar_dense.default(reinterpret_tensor(buf3, (), (), 0))
        u0 = buf4
        if not (u0 >= 0):
            raise RuntimeError('u0 >= 0')
        buf5 = None
        if not (0 <= u0):
            raise RuntimeError('0 <= u0')
        buf6 = None
        # Topologically Sorted Source Nodes: [getitem_1, item_1], Original ATen: [aten.select, aten._local_scalar_dense]
        # [Provenance debug handles] torch.ops.aten._local_scalar_dense.default:5
        buf7 = torch.ops.aten._local_scalar_dense.default(reinterpret_tensor(buf3, (), (), 1))
        u1 = buf7
        del buf3
        if not (u1 >= 0):
            raise RuntimeError('u1 >= 0')
        buf8 = None
        if not (0 <= u1):
            raise RuntimeError('0 <= u1')
        buf9 = None
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            # Topologically Sorted Source Nodes: [view, input_splits], Original ATen: [aten.view, aten.sum]
            # [Provenance debug handles] torch.ops.aten.sum.dim_IntList:6
            buf10 = torch.ops.aten.sum.dim_IntList(reinterpret_tensor(arg0_1, (2, 4), (4, 1), 0), [1])
            assert_alignment(buf10, 16, 'torch.ops.aten.sum.dim_IntList')
            del arg0_1
        buf11 = empty_strided_cpu_pinned((2, ), (1, ), torch.int64)
        buf11.copy_(buf10, True)
        _d2h_event_buf11 = torch.Event()
        _d2h_event_buf11.record()
        _d2h_event_buf11.synchronize()
        del buf10
        # Topologically Sorted Source Nodes: [getitem_2, item_2], Original ATen: [aten.select, aten._local_scalar_dense]
        # [Provenance debug handles] torch.ops.aten._local_scalar_dense.default:7
        buf12 = torch.ops.aten._local_scalar_dense.default(reinterpret_tensor(buf11, (), (), 0))
        u2 = buf12
        # Topologically Sorted Source Nodes: [getitem_3, item_3], Original ATen: [aten.select, aten._local_scalar_dense]
        # [Provenance debug handles] torch.ops.aten._local_scalar_dense.default:8
        buf13 = torch.ops.aten._local_scalar_dense.default(reinterpret_tensor(buf11, (), (), 1))
        u3 = buf13
        del buf11
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg1_1 = copy_if_misaligned(arg1_1)
            # Topologically Sorted Source Nodes: [routed], Original ATen: [_c10d_functional.all_to_all_single]
            # [Provenance debug handles] torch.ops._c10d_functional.all_to_all_single.default:9
            buf14 = torch.ops._c10d_functional.all_to_all_single.default(arg1_1, [u0, u1], [u2, u3], 't085-default')
            assert_alignment(buf14, 16, 'torch.ops._c10d_functional.all_to_all_single.default')
            del arg1_1
            # Topologically Sorted Source Nodes: [wait_tensor_1], Original ATen: [_c10d_functional.wait_tensor]
            # [Provenance debug handles] torch.ops._c10d_functional.wait_tensor.default:10
            buf15 = torch.ops._c10d_functional.wait_tensor.default(buf14)
            assert_alignment(buf15, 16, 'torch.ops._c10d_functional.wait_tensor.default')
            del buf14
        return (buf15, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8, ), (1, ), device='npu:0', dtype=torch.int64)
    arg1_1 = rand_strided((1024, 128), (128, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
