class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 4, 8, 16]", arg1_1: "f32[2, 4, 8, 16]", arg2_1: "b8[1, 1, 8, 8]", arg3_1: "f32[1, 1, 8, 8]", arg4_1: "f32[2, 4, 8, 16]"):
        # No stacktrace found for following nodes
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, device = device(type='npu', index=0), pin_memory = False)
        where_self: "f32[1, 1, 8, 8]" = torch.ops.aten.where.self(arg2_1, arg3_1, full_default_2);  arg2_1 = arg3_1 = full_default_2 = None
        mul_scalar: "f32[2, 4, 8, 16]" = torch.ops.aten.mul.Scalar(arg1_1, 1.2247509951618742);  arg1_1 = None
        permute_default: "f32[2, 4, 16, 8]" = torch.ops.aten.permute.default(arg0_1, [0, 1, 3, 2]);  arg0_1 = None
        mul_scalar_1: "f32[2, 4, 16, 8]" = torch.ops.aten.mul.Scalar(permute_default, 1.2247509951618742);  permute_default = None
        expand_default: "f32[2, 4, 8, 16]" = torch.ops.aten.expand.default(mul_scalar, [2, 4, 8, 16]);  mul_scalar = None
        view_default: "f32[8, 8, 16]" = torch.ops.aten.view.default(expand_default, [8, 8, 16]);  expand_default = None
        expand_default_1: "f32[2, 4, 16, 8]" = torch.ops.aten.expand.default(mul_scalar_1, [2, 4, 16, 8]);  mul_scalar_1 = None
        view_default_1: "f32[8, 16, 8]" = torch.ops.aten.view.default(expand_default_1, [8, 16, 8]);  expand_default_1 = None
        bmm_default: "f32[8, 8, 8]" = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[2, 4, 8, 8]" = torch.ops.aten.view.default(bmm_default, [2, 4, 8, 8]);  bmm_default = None
        add_tensor: "f32[2, 4, 8, 8]" = torch.ops.aten.add.Tensor(view_default_2, where_self);  view_default_2 = where_self = None
        amax_default: "f32[2, 4, 8, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[2, 4, 8, 8]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  amax_default = None
        exp_default: "f32[2, 4, 8, 8]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[2, 4, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[2, 4, 8, 8]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        eq_scalar: "b8[2, 4, 8, 8]" = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
        logical_not_default: "b8[2, 4, 8, 8]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[2, 4, 8, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[2, 4, 8, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_3: "f32[2, 4, 8, 8]" = torch.ops.aten.full.default([2, 4, 8, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_self_1: "f32[2, 4, 8, 8]" = torch.ops.aten.where.self(logical_not_default_1, full_default_3, div_tensor);  logical_not_default_1 = full_default_3 = div_tensor = None
        expand_default_2: "f32[2, 4, 8, 8]" = torch.ops.aten.expand.default(where_self_1, [2, 4, 8, 8]);  where_self_1 = None
        view_default_3: "f32[8, 8, 8]" = torch.ops.aten.view.default(expand_default_2, [8, 8, 8]);  expand_default_2 = None
        expand_default_3: "f32[2, 4, 8, 16]" = torch.ops.aten.expand.default(arg4_1, [2, 4, 8, 16]);  arg4_1 = None
        view_default_4: "f32[8, 8, 16]" = torch.ops.aten.view.default(expand_default_3, [8, 8, 16]);  expand_default_3 = None
        bmm_default_1: "f32[8, 8, 16]" = torch.ops.aten.bmm.default(view_default_3, view_default_4);  view_default_3 = view_default_4 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:568 in _sfdp_pattern_19, code: return torch.nn.functional.dropout(attn_weights, dropout_p).matmul(value)
        view_default_5: "f32[2, 4, 8, 16]" = torch.ops.aten.view.default(bmm_default_1, [2, 4, 8, 16]);  bmm_default_1 = None
        return (view_default_5,)
