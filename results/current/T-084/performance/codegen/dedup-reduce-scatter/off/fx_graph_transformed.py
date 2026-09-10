class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 128]", arg1_1: "f32[4, 128]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t084_performance_worker.py:242 in fn, code: first_out = reduce_scatter(first, "avg", world_size, group)
        reduce_scatter_tensor: "f32[2, 128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(arg0_1, 'avg', 2, '0');  arg0_1 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t084_performance_worker.py:243 in fn, code: second_out = reduce_scatter(second, "avg", world_size, group)
        reduce_scatter_tensor_1: "f32[2, 128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(arg1_1, 'avg', 2, '0');  arg1_1 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t084_performance_worker.py:244 in fn, code: return wait_tensor(first_out) + wait_tensor(second_out)
        wait_tensor: "f32[2, 128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        wait_tensor_1: "f32[2, 128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        add: "f32[2, 128]" = torch.ops.aten.add.Tensor(wait_tensor, wait_tensor_1);  wait_tensor = wait_tensor_1 = None
        return (add,)
