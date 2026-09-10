class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[8]", arg1_1: "f32[1024, 128]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:293 in function, code: exchanged = torch.ops._c10d_functional.all_to_all_single(
        all_to_all_single: "i64[8]" = torch.ops._c10d_functional.all_to_all_single.default(arg0_1, [4, 4], [4, 4], 't085-default')

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:299 in function, code: exchanged = torch.ops._c10d_functional.wait_tensor(exchanged)
        wait_tensor: "i64[8]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single);  all_to_all_single = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:301 in function, code: output_splits = exchanged.view(world, -1).sum(dim=1)
        view_1: "i64[2, 4]" = torch.ops.aten.reshape.default(wait_tensor, [2, -1]);  wait_tensor = None
        sum_2: "i64[2]" = torch.ops.aten.sum.dim_IntList(view_1, [1]);  view_1 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:303 in function, code: cpu_output = output_splits.to("cpu", non_blocking=False)
        device_put_1: "i64[2]" = torch.ops.prims.device_put.default(sum_2, device(type='cpu'));  sum_2 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:306 in function, code: cpu_output.tolist(),
        select: "i64[]" = torch.ops.aten.select.int(device_put_1, 0, 0)
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(select);  select = None
        ge: "Sym(u0 >= 0)" = _local_scalar_dense >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        select_1: "i64[]" = torch.ops.aten.select.int(device_put_1, 0, 1);  device_put_1 = None
        _local_scalar_dense_1: "Sym(u1)" = torch.ops.aten._local_scalar_dense.default(select_1);  select_1 = None
        ge_1: "Sym(u1 >= 0)" = _local_scalar_dense_1 >= 0
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_1 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:300 in function, code: input_splits = num_tokens_per_expert.view(world, -1).sum(dim=1)
        view: "i64[2, 4]" = torch.ops.aten.reshape.default(arg0_1, [2, -1]);  arg0_1 = None
        sum_1: "i64[2]" = torch.ops.aten.sum.dim_IntList(view, [1]);  view = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:302 in function, code: cpu_input = input_splits.to("cpu", non_blocking=True)
        device_put: "i64[2]" = torch.ops.prims.device_put.default(sum_1, device(type='cpu'), False);  sum_1 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:307 in function, code: cpu_input.tolist(),
        select_2: "i64[]" = torch.ops.aten.select.int(device_put, 0, 0)
        _local_scalar_dense_2: "Sym(u2)" = torch.ops.aten._local_scalar_dense.default(select_2);  select_2 = None
        select_3: "i64[]" = torch.ops.aten.select.int(device_put, 0, 1);  device_put = None
        _local_scalar_dense_3: "Sym(u3)" = torch.ops.aten._local_scalar_dense.default(select_3);  select_3 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:304 in function, code: routed = torch.ops._c10d_functional.all_to_all_single(
        all_to_all_single_1: "f32[u0 + u1, 128]" = torch.ops._c10d_functional.all_to_all_single.default(arg1_1, [_local_scalar_dense, _local_scalar_dense_1], [_local_scalar_dense_2, _local_scalar_dense_3], 't085-default');  arg1_1 = _local_scalar_dense = _local_scalar_dense_1 = _local_scalar_dense_2 = _local_scalar_dense_3 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:310 in function, code: return torch.ops._c10d_functional.wait_tensor(routed)
        wait_tensor_1: "f32[u0 + u1, 128]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_1);  all_to_all_single_1 = None
        return (wait_tensor_1,)
