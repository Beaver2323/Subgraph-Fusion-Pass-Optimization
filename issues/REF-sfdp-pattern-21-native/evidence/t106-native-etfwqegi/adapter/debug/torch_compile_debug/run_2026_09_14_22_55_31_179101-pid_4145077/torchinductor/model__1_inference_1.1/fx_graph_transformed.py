class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 1, 1, 2]", arg1_1: "f32[1, 2, 16, 32]", arg2_1: "f32[1, 2, 16, 32]", arg3_1: "f32[1, 2, 16, 32]"):
        # No stacktrace found for following nodes
        permute_default: "f32[1, 16, 2, 32]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
        mul_scalar: "f32[1, 16, 2, 32]" = torch.ops.aten.mul.Scalar(permute_default, 1.0);  permute_default = None
        expand_default: "f32[1, 16, 2, 32]" = torch.ops.aten.expand.default(mul_scalar, [1, 16, 2, 32]);  mul_scalar = None
        view_default: "f32[16, 2, 32]" = torch.ops.aten.reshape.default(expand_default, [16, 2, 32]);  expand_default = None
        permute_default_1: "f32[1, 16, 2, 32]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
        permute_default_3: "f32[1, 16, 32, 2]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        mul_scalar_1: "f32[1, 16, 32, 2]" = torch.ops.aten.mul.Scalar(permute_default_3, 1.0);  permute_default_3 = None
        expand_default_1: "f32[1, 16, 32, 2]" = torch.ops.aten.expand.default(mul_scalar_1, [1, 16, 32, 2]);  mul_scalar_1 = None
        view_default_1: "f32[16, 32, 2]" = torch.ops.aten.reshape.default(expand_default_1, [16, 32, 2]);  expand_default_1 = None
        bmm_default: "f32[16, 2, 2]" = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[1, 16, 2, 2]" = torch.ops.aten.reshape.default(bmm_default, [1, 16, 2, 2]);  bmm_default = None
        add_tensor: "f32[1, 16, 2, 2]" = torch.ops.aten.add.Tensor(view_default_2, arg0_1);  view_default_2 = arg0_1 = None
        eq_scalar: "b8[1, 16, 2, 2]" = torch.ops.aten.eq.Scalar(add_tensor, -inf)
        logical_not_default: "b8[1, 16, 2, 2]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[1, 16, 2, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[1, 16, 2, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default: "f32[1, 16, 2, 2]" = torch.ops.aten.full.default([1, 16, 2, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        amax_default: "f32[1, 16, 2, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[1, 16, 2, 2]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[1, 16, 2, 2]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[1, 16, 2, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[1, 16, 2, 2]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self: "f32[1, 16, 2, 2]" = torch.ops.aten.where.self(logical_not_default_1, full_default, div_tensor);  logical_not_default_1 = full_default = div_tensor = None
        expand_default_2: "f32[1, 16, 2, 2]" = torch.ops.aten.expand.default(where_self, [1, 16, 2, 2]);  where_self = None
        view_default_3: "f32[16, 2, 2]" = torch.ops.aten.reshape.default(expand_default_2, [16, 2, 2]);  expand_default_2 = None
        permute_default_2: "f32[1, 16, 2, 32]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1, 3]);  arg3_1 = None
        expand_default_3: "f32[1, 16, 2, 32]" = torch.ops.aten.expand.default(permute_default_2, [1, 16, 2, 32]);  permute_default_2 = None
        view_default_4: "f32[16, 2, 32]" = torch.ops.aten.reshape.default(expand_default_3, [16, 2, 32]);  expand_default_3 = None
        bmm_default_1: "f32[16, 2, 32]" = torch.ops.aten.bmm.default(view_default_3, view_default_4);  view_default_3 = view_default_4 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1361 in dot_prod_attention, code: return attn_weights.matmul(value)
        view_default_5: "f32[1, 16, 2, 32]" = torch.ops.aten.reshape.default(bmm_default_1, [1, 16, 2, 32]);  bmm_default_1 = None
        return (view_default_5,)
