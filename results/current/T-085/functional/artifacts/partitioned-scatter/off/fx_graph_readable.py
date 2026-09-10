class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[501, 100]", arg1_1: "f32[1000000, 100]", arg2_1: "i64[1000000]", arg3_1: "f32[501, 100]", arg4_1: "f32[501, 100]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:408 in scatter_fn, code: out0 = out0.index_put([index], values, accumulate=True)
        index_put: "f32[501, 100]" = torch.ops.aten.index_put.default(arg0_1, [arg2_1], arg1_1, True);  arg0_1 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:409 in scatter_fn, code: out1 = out1.index_put([index], values, accumulate=True)
        index_put_1: "f32[501, 100]" = torch.ops.aten.index_put.default(arg3_1, [arg2_1], arg1_1, True);  arg3_1 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:410 in scatter_fn, code: out2 = out2.index_put([index], values, accumulate=True)
        index_put_2: "f32[501, 100]" = torch.ops.aten.index_put.default(arg4_1, [arg2_1], arg1_1, True);  arg4_1 = arg2_1 = arg1_1 = None
        return (index_put, index_put_1, index_put_2)
