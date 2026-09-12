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


cpp_fused_log2_0 = async_compile.cpp_pybinding(['const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(7L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = std::log2(tmp0);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp1;
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__to_copy_add_clamp_1 = async_compile.cpp_pybinding(['const float*', 'uint8_t*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       uint8_t* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(7L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = static_cast<float>(-127.0);
                    auto tmp2 = max_propagate_nan(tmp0, tmp1);
                    auto tmp3 = static_cast<float>(127.0);
                    auto tmp4 = min_propagate_nan(tmp2, tmp3);
                    auto tmp5 = float(tmp4 + tmp3);
                    auto tmp6 = c10::convert<uint8_t>(tmp5);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp6;
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


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
        buf0 = empty_strided_cpu((7, ), (1, ), torch.float32)
        # [Provenance debug handles] cpp_fused_log2_0:1
        cpp_fused_log2_0(arg0_1, buf0)
        del arg0_1
        # Topologically Sorted Source Nodes: [log2, ceil], Original ATen: [aten.log2, aten.ceil]
        # [Provenance debug handles] torch.ops.aten.ceil.default:2
        buf1 = torch.ops.aten.ceil.default(buf0)
        assert_alignment(buf1, 16, 'torch.ops.aten.ceil.default')
        del buf0
        buf2 = empty_strided_cpu((7, ), (1, ), torch.uint8)
        # [Provenance debug handles] cpp_fused__to_copy_add_clamp_1:3
        cpp_fused__to_copy_add_clamp_1(buf1, buf2)
        return (buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((7, ), (1, ), device='cpu', dtype=torch.float32)
    return [arg0_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cpu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
