class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[2, 4, 8, 16]", arg1_1: "f16[2, 4, 8, 16]", arg2_1: "f16[2, 4, 8, 16]", arg3_1: "f32[1, 1, 8, 8]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f16[1, 1, 8, 8]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float16);  arg3_1 = None
        convert_element_type_default_1: "f32[2, 4, 8, 16]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_default_2: "f32[2, 4, 8, 16]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        convert_element_type_default_3: "f32[2, 4, 8, 16]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul_scalar: "f32[2, 4, 8, 16]" = torch.ops.aten.mul.Scalar(convert_element_type_default_1, 1.0);  convert_element_type_default_1 = None
        permute_default: "f32[2, 4, 16, 8]" = torch.ops.aten.permute.default(convert_element_type_default_2, [0, 1, 3, 2]);  convert_element_type_default_2 = None
        mul_scalar_1: "f32[2, 4, 16, 8]" = torch.ops.aten.mul.Scalar(permute_default, 1.0);  permute_default = None
        expand_default: "f32[2, 4, 8, 16]" = torch.ops.aten.expand.default(mul_scalar, [2, 4, 8, 16]);  mul_scalar = None
        view_default: "f32[8, 8, 16]" = torch.ops.aten.view.default(expand_default, [8, 8, 16]);  expand_default = None
        expand_default_1: "f32[2, 4, 16, 8]" = torch.ops.aten.expand.default(mul_scalar_1, [2, 4, 16, 8]);  mul_scalar_1 = None
        view_default_1: "f32[8, 16, 8]" = torch.ops.aten.view.default(expand_default_1, [8, 16, 8]);  expand_default_1 = None
        bmm_default: "f32[8, 8, 8]" = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[2, 4, 8, 8]" = torch.ops.aten.view.default(bmm_default, [2, 4, 8, 8]);  bmm_default = None
        add_tensor: "f32[2, 4, 8, 8]" = torch.ops.aten.add.Tensor(view_default_2, convert_element_type_default);  view_default_2 = convert_element_type_default = None
        amax_default: "f32[2, 4, 8, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[2, 4, 8, 8]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  amax_default = None
        exp_default: "f32[2, 4, 8, 8]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[2, 4, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[2, 4, 8, 8]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        eq_scalar: "b8[2, 4, 8, 8]" = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
        logical_not_default: "b8[2, 4, 8, 8]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[2, 4, 8, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[2, 4, 8, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default: "f32[2, 4, 8, 8]" = torch.ops.aten.full.default([2, 4, 8, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_self: "f32[2, 4, 8, 8]" = torch.ops.aten.where.self(logical_not_default_1, full_default, div_tensor);  logical_not_default_1 = full_default = div_tensor = None
        expand_default_2: "f32[2, 4, 8, 8]" = torch.ops.aten.expand.default(where_self, [2, 4, 8, 8]);  where_self = None
        view_default_3: "f32[8, 8, 8]" = torch.ops.aten.view.default(expand_default_2, [8, 8, 8]);  expand_default_2 = None
        expand_default_3: "f32[2, 4, 8, 16]" = torch.ops.aten.expand.default(convert_element_type_default_3, [2, 4, 8, 16]);  convert_element_type_default_3 = None
        view_default_4: "f32[8, 8, 16]" = torch.ops.aten.view.default(expand_default_3, [8, 8, 16]);  expand_default_3 = None
        bmm_default_1: "f32[8, 8, 16]" = torch.ops.aten.bmm.default(view_default_3, view_default_4);  view_default_3 = view_default_4 = None
        view_default_5: "f32[2, 4, 8, 16]" = torch.ops.aten.view.default(bmm_default_1, [2, 4, 8, 16]);  bmm_default_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:736 in _sfdp_pattern_24, code: attn_output = attn_output.view(bs, n_head, seq_len, head_size)
        convert_element_type_default_4: "f16[2, 4, 8, 16]" = torch.ops.prims.convert_element_type.default(view_default_5, torch.float16);  view_default_5 = None
        return (convert_element_type_default_4,)
