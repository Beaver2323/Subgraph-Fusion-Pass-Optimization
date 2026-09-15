class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "f32[4, 2, 16, 32]", mul_scalar: "f32[4, 2, 16, 32]", mul_scalar_1: "f32[4, 2, 32, 16]", where_self: "f32[4, 2, 16, 16]", tangents_1: "f32[4, 2, 16, 32]"):
        # No stacktrace found for following nodes
        matmul_backward_default = torch.ops.aten.matmul_backward.default(tangents_1, where_self, primals_3, [True, True]);  tangents_1 = primals_3 = None
        getitem_4: "f32[4, 2, 16, 16]" = matmul_backward_default[0]

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:372 in sfdp_pattern_5_v1, code: return attn_weight @ value
        getitem_5: "f32[4, 2, 16, 32]" = matmul_backward_default[1];  matmul_backward_default = None

        # No stacktrace found for following nodes
        mul_tensor: "f32[4, 2, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_4, where_self);  getitem_4 = None
        sum_dim_int_list_1: "f32[4, 2, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        mul_tensor_1: "f32[4, 2, 16, 16]" = torch.ops.aten.mul.Tensor(where_self, sum_dim_int_list_1);  where_self = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[4, 2, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        matmul_backward_default_1 = torch.ops.aten.matmul_backward.default(sub_tensor_1, mul_scalar, mul_scalar_1, [True, True]);  sub_tensor_1 = mul_scalar = mul_scalar_1 = None
        getitem_6: "f32[4, 2, 16, 32]" = matmul_backward_default_1[0]
        getitem_7: "f32[4, 2, 32, 16]" = matmul_backward_default_1[1];  matmul_backward_default_1 = None
        mul_scalar_2: "f32[4, 2, 32, 16]" = torch.ops.aten.mul.Scalar(getitem_7, 0.42044820762685725);  getitem_7 = None

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:369 in sfdp_pattern_5_v1, code: (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
        permute_default_1: "f32[4, 2, 16, 32]" = torch.ops.aten.permute.default(mul_scalar_2, [0, 1, 3, 2]);  mul_scalar_2 = None
        mul_scalar_3: "f32[4, 2, 16, 32]" = torch.ops.aten.mul.Scalar(getitem_6, 0.42044820762685725);  getitem_6 = None
        return (permute_default_1, mul_scalar_3, getitem_5)
