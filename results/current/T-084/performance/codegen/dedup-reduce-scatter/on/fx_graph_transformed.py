class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 128]", arg1_1: "f32[4, 128]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[4, 128]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        reduce_scatter_tensor_default: "f32[2, 128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_tensor, 'avg', 2, '0');  add_tensor = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t084_performance_worker.py:244 in fn, code: return wait_tensor(first_out) + wait_tensor(second_out)
        wait_tensor_default: "f32[2, 128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_default);  reduce_scatter_tensor_default = None
        return (wait_tensor_default,)
