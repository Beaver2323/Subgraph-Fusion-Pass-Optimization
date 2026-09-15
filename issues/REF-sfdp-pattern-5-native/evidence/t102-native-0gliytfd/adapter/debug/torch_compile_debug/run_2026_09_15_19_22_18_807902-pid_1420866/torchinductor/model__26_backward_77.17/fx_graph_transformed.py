class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 2, 16, 32]", primals_2: "f32[4, 2, 16, 32]", primals_3: "f32[4, 2, 16, 32]", tangents_1: "f32[4, 2, 16, 32]"):
        # ac_graph_id: 12 - PREFER_RECOMPUTE Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:387 in sfdp_pattern_5_v3, code: attn_mask = torch.ones(
        full_default: "b8[16, 16]" = torch.ops.aten.full.default([16, 16], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # ac_graph_id: 12 - PREFER_RECOMPUTE Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:389 in sfdp_pattern_5_v3, code: ).tril(diagonal=0)
        iota: "i64[16]" = torch.ops.prims.iota.default(16, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze: "i64[1, 16]" = torch.ops.aten.unsqueeze.default(iota, -2)
        unsqueeze_1: "i64[16, 1]" = torch.ops.aten.unsqueeze.default(iota, -1);  iota = None
        sub: "i64[16, 16]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        le: "b8[16, 16]" = torch.ops.aten.le.Scalar(sub, 0);  sub = None
        logical_and: "b8[16, 16]" = torch.ops.aten.logical_and.default(le, full_default);  le = full_default = None

        # ac_graph_id: 12 - PREFER_RECOMPUTE Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:391 in sfdp_pattern_5_v3, code: torch.logical_not(attn_mask), -float("inf")
        logical_not: "b8[16, 16]" = torch.ops.aten.logical_not.default(logical_and)

        # ac_graph_id: 12 - PREFER_RECOMPUTE Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:390 in sfdp_pattern_5_v3, code: attn_mask = attn_mask.masked_fill(
        full_default_1: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "b8[16, 16]" = torch.ops.aten.where.self(logical_not, full_default_1, logical_and);  logical_not = full_default_1 = logical_and = None

        # ac_graph_id: 12 - PREFER_RECOMPUTE No stacktrace found for following nodes
        convert_element_type_default: "f32[16, 16]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        mul_scalar: "f32[4, 2, 16, 32]" = torch.ops.aten.mul.Scalar(primals_1, 0.4167804918837871);  primals_1 = None
        permute_default: "f32[4, 2, 32, 16]" = torch.ops.aten.permute.default(primals_2, [0, 1, 3, 2]);  primals_2 = None
        mul_scalar_1: "f32[4, 2, 32, 16]" = torch.ops.aten.mul.Scalar(permute_default, 0.4167804918837871);  permute_default = None
        expand_default: "f32[4, 2, 16, 32]" = torch.ops.aten.expand.default(mul_scalar, [4, 2, 16, 32])
        view_default: "f32[8, 16, 32]" = torch.ops.aten.reshape.default(expand_default, [8, 16, 32]);  expand_default = None
        expand_default_1: "f32[4, 2, 32, 16]" = torch.ops.aten.expand.default(mul_scalar_1, [4, 2, 32, 16])
        view_default_1: "f32[8, 32, 16]" = torch.ops.aten.reshape.default(expand_default_1, [8, 32, 16]);  expand_default_1 = None
        bmm_default: "f32[8, 16, 16]" = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[4, 2, 16, 16]" = torch.ops.aten.reshape.default(bmm_default, [4, 2, 16, 16]);  bmm_default = None
        add_tensor: "f32[4, 2, 16, 16]" = torch.ops.aten.add.Tensor(view_default_2, convert_element_type_default);  view_default_2 = convert_element_type_default = None
        amax_default: "f32[4, 2, 16, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[4, 2, 16, 16]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  amax_default = None
        exp_default: "f32[4, 2, 16, 16]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[4, 2, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[4, 2, 16, 16]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        eq_scalar: "b8[4, 2, 16, 16]" = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
        logical_not_default: "b8[4, 2, 16, 16]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[4, 2, 16, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[4, 2, 16, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_2: "f32[4, 2, 16, 16]" = torch.ops.aten.full.default([4, 2, 16, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_self: "f32[4, 2, 16, 16]" = torch.ops.aten.where.self(logical_not_default_1, full_default_2, div_tensor);  logical_not_default_1 = full_default_2 = div_tensor = None

        # No stacktrace found for following nodes
        matmul_backward_default = torch.ops.aten.matmul_backward.default(tangents_1, where_self, primals_3, [True, True]);  tangents_1 = primals_3 = None
        getitem_4: "f32[4, 2, 16, 16]" = matmul_backward_default[0]

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:398 in sfdp_pattern_5_v3, code: return attn_weight @ value
        getitem_5: "f32[4, 2, 16, 32]" = matmul_backward_default[1];  matmul_backward_default = None

        # No stacktrace found for following nodes
        mul_tensor: "f32[4, 2, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_4, where_self);  getitem_4 = None
        sum_dim_int_list_1: "f32[4, 2, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        mul_tensor_1: "f32[4, 2, 16, 16]" = torch.ops.aten.mul.Tensor(where_self, sum_dim_int_list_1);  where_self = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[4, 2, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        matmul_backward_default_1 = torch.ops.aten.matmul_backward.default(sub_tensor_1, mul_scalar, mul_scalar_1, [True, True]);  sub_tensor_1 = mul_scalar = mul_scalar_1 = None
        getitem_6: "f32[4, 2, 16, 32]" = matmul_backward_default_1[0]
        getitem_7: "f32[4, 2, 32, 16]" = matmul_backward_default_1[1];  matmul_backward_default_1 = None
        mul_scalar_2: "f32[4, 2, 32, 16]" = torch.ops.aten.mul.Scalar(getitem_7, 0.4167804918837871);  getitem_7 = None

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:394 in sfdp_pattern_5_v3, code: (query @ key.transpose(-2, -1) / (math.sqrt(query.size(-1)) + 0.1))
        permute_default_1: "f32[4, 2, 16, 32]" = torch.ops.aten.permute.default(mul_scalar_2, [0, 1, 3, 2]);  mul_scalar_2 = None
        mul_scalar_3: "f32[4, 2, 16, 32]" = torch.ops.aten.mul.Scalar(getitem_6, 0.4167804918837871);  getitem_6 = None
        return (mul_scalar_3, permute_default_1, getitem_5)
