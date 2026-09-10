class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[10]", arg1_1: "f32[1024]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t086_performance_worker.py:280 in fn, code: src = torch.ones(idx.size(0), device=x.device)
        full_default: "f32[10]" = torch.ops.aten.full.default([10], 1, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t086_performance_worker.py:281 in fn, code: x.index_put_((idx,), src)
        index_put: "f32[1024]" = torch.ops.aten.index_put.default(arg1_1, [arg0_1], full_default);  arg0_1 = full_default = None
        copy_: "f32[1024]" = torch.ops.aten.copy_.default(arg1_1, index_put);  arg1_1 = copy_ = None

        # No stacktrace found for following nodes
        expand_1: "f32[2, 1024]" = torch.ops.aten.expand.default(index_put, [2, 1024]);  index_put = None
        return (expand_1,)
