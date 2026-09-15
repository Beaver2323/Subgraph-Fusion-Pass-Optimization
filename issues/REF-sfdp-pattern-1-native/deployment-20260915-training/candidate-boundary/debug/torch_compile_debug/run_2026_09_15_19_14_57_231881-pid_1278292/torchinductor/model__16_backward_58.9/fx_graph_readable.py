class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "f32[2, 2, 8, 16]", mul_scalar: "f32[2, 2, 8, 16]", mul_scalar_1: "f32[2, 2, 16, 8]", where_self: "f32[2, 2, 8, 8]", tangents_1: "f32[2, 2, 8, 16]"):
        # No stacktrace found for following nodes
        matmul_backward_default = torch.ops.aten.matmul_backward.default(tangents_1, where_self, primals_4, [True, True]);  tangents_1 = primals_4 = None
        getitem_4: "f32[2, 2, 8, 8]" = matmul_backward_default[0]

        # Backward of forward node: File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/attention_training_boundary.py:57 in mask, code: return ((q @ k.transpose(-2,-1)).div(4.0) + extra).softmax(-1) @ v
        getitem_5: "f32[2, 2, 8, 16]" = matmul_backward_default[1];  matmul_backward_default = None

        # No stacktrace found for following nodes
        mul_tensor: "f32[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_4, where_self);  getitem_4 = None
        sum_dim_int_list_1: "f32[2, 2, 8, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        mul_tensor_1: "f32[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(where_self, sum_dim_int_list_1);  where_self = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[2, 2, 8, 8]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        matmul_backward_default_1 = torch.ops.aten.matmul_backward.default(sub_tensor_1, mul_scalar, mul_scalar_1, [True, True]);  sub_tensor_1 = mul_scalar = mul_scalar_1 = None
        getitem_6: "f32[2, 2, 8, 16]" = matmul_backward_default_1[0]
        getitem_7: "f32[2, 2, 16, 8]" = matmul_backward_default_1[1];  matmul_backward_default_1 = None
        mul_scalar_2: "f32[2, 2, 16, 8]" = torch.ops.aten.mul.Scalar(getitem_7, 0.5);  getitem_7 = None

        # Backward of forward node: File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/attention_training_boundary.py:57 in mask, code: return ((q @ k.transpose(-2,-1)).div(4.0) + extra).softmax(-1) @ v
        permute_default_1: "f32[2, 2, 8, 16]" = torch.ops.aten.permute.default(mul_scalar_2, [0, 1, 3, 2]);  mul_scalar_2 = None
        mul_scalar_3: "f32[2, 2, 8, 16]" = torch.ops.aten.mul.Scalar(getitem_6, 0.5);  getitem_6 = None
        return (permute_default_1, mul_scalar_3, None, getitem_5)
