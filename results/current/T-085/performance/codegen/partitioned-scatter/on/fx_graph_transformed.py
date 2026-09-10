class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[501, 100]", arg1_1: "f32[1000000, 100]", arg2_1: "i64[1000000]", arg3_1: "f32[501, 100]", arg4_1: "f32[501, 100]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:408 in scatter_fn, code: out0 = out0.index_put([index], values, accumulate=True)
        iota_default_2: "i64[1000000]" = torch.ops.prims.iota.default(1000000, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        bitwise_and_scalar_2: "i64[1000000]" = torch.ops.aten.bitwise_and.Scalar(iota_default_2, 63);  iota_default_2 = None
        full_default_2: "f32[32064, 100]" = torch.ops.aten.full.default([32064, 100], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        mul_tensor_2: "i64[1000000]" = torch.ops.aten.mul.Tensor(bitwise_and_scalar_2, 501);  bitwise_and_scalar_2 = None
        add_tensor_4: "i64[1000000]" = torch.ops.aten.add.Tensor(arg2_1, mul_tensor_2);  mul_tensor_2 = None
        index_put_default_2: "f32[32064, 100]" = torch.ops.aten.index_put_.default(full_default_2, [add_tensor_4], arg1_1, True);  full_default_2 = add_tensor_4 = None
        view_default_2: "f32[64, 501, 100]" = torch.ops.aten.view.default(index_put_default_2, [64, 501, 100]);  index_put_default_2 = None
        sum_dim_int_list_2: "f32[501, 100]" = torch.ops.aten.sum.dim_IntList(view_default_2, [0]);  view_default_2 = None
        add_tensor_5: "f32[501, 100]" = torch.ops.aten.add.Tensor(arg0_1, sum_dim_int_list_2);  arg0_1 = sum_dim_int_list_2 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:409 in scatter_fn, code: out1 = out1.index_put([index], values, accumulate=True)
        iota_default_1: "i64[1000000]" = torch.ops.prims.iota.default(1000000, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        bitwise_and_scalar_1: "i64[1000000]" = torch.ops.aten.bitwise_and.Scalar(iota_default_1, 63);  iota_default_1 = None
        full_default_1: "f32[32064, 100]" = torch.ops.aten.full.default([32064, 100], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        mul_tensor_1: "i64[1000000]" = torch.ops.aten.mul.Tensor(bitwise_and_scalar_1, 501);  bitwise_and_scalar_1 = None
        add_tensor_2: "i64[1000000]" = torch.ops.aten.add.Tensor(arg2_1, mul_tensor_1);  mul_tensor_1 = None
        index_put_default_1: "f32[32064, 100]" = torch.ops.aten.index_put_.default(full_default_1, [add_tensor_2], arg1_1, True);  full_default_1 = add_tensor_2 = None
        view_default_1: "f32[64, 501, 100]" = torch.ops.aten.view.default(index_put_default_1, [64, 501, 100]);  index_put_default_1 = None
        sum_dim_int_list_1: "f32[501, 100]" = torch.ops.aten.sum.dim_IntList(view_default_1, [0]);  view_default_1 = None
        add_tensor_3: "f32[501, 100]" = torch.ops.aten.add.Tensor(arg3_1, sum_dim_int_list_1);  arg3_1 = sum_dim_int_list_1 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:410 in scatter_fn, code: out2 = out2.index_put([index], values, accumulate=True)
        iota_default: "i64[1000000]" = torch.ops.prims.iota.default(1000000, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        bitwise_and_scalar: "i64[1000000]" = torch.ops.aten.bitwise_and.Scalar(iota_default, 63);  iota_default = None
        full_default: "f32[32064, 100]" = torch.ops.aten.full.default([32064, 100], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        mul_tensor: "i64[1000000]" = torch.ops.aten.mul.Tensor(bitwise_and_scalar, 501);  bitwise_and_scalar = None
        add_tensor: "i64[1000000]" = torch.ops.aten.add.Tensor(arg2_1, mul_tensor);  arg2_1 = mul_tensor = None
        index_put_default: "f32[32064, 100]" = torch.ops.aten.index_put_.default(full_default, [add_tensor], arg1_1, True);  full_default = add_tensor = arg1_1 = None
        view_default: "f32[64, 501, 100]" = torch.ops.aten.view.default(index_put_default, [64, 501, 100]);  index_put_default = None
        sum_dim_int_list: "f32[501, 100]" = torch.ops.aten.sum.dim_IntList(view_default, [0]);  view_default = None
        add_tensor_1: "f32[501, 100]" = torch.ops.aten.add.Tensor(arg4_1, sum_dim_int_list);  arg4_1 = sum_dim_int_list = None
        return (add_tensor_5, add_tensor_3, add_tensor_1)
