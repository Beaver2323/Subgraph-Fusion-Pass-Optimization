class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[2, 4, 8, 16]", arg1_1: "f16[2, 4, 8, 16]", arg2_1: "f16[2, 4, 8, 16]", arg3_1: "f32[2, 1, 1, 4]"):
        # No stacktrace found for following nodes
        permute_default: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        convert_element_type_default_1: "f32[2, 8, 4, 16]" = torch.ops.prims.convert_element_type.default(permute_default, torch.float32);  permute_default = None
        mul_scalar: "f32[2, 8, 4, 16]" = torch.ops.aten.mul.Scalar(convert_element_type_default_1, 1.0);  convert_element_type_default_1 = None
        expand_default: "f32[2, 8, 4, 16]" = torch.ops.aten.expand.default(mul_scalar, [2, 8, 4, 16]);  mul_scalar = None
        clone_default: "f32[2, 8, 4, 16]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default: "f32[16, 4, 16]" = torch.ops.aten.reshape.default(clone_default, [16, 4, 16]);  clone_default = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:631 in _sfdp_pattern_21, code: key = key.permute([0, 2, 1, 3])
        permute_default_1: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None

        # No stacktrace found for following nodes
        convert_element_type_default_2: "f32[2, 8, 4, 16]" = torch.ops.prims.convert_element_type.default(permute_default_1, torch.float32);  permute_default_1 = None
        permute_default_3: "f32[2, 8, 16, 4]" = torch.ops.aten.permute.default(convert_element_type_default_2, [0, 1, 3, 2]);  convert_element_type_default_2 = None
        mul_scalar_1: "f32[2, 8, 16, 4]" = torch.ops.aten.mul.Scalar(permute_default_3, 1.0);  permute_default_3 = None
        expand_default_1: "f32[2, 8, 16, 4]" = torch.ops.aten.expand.default(mul_scalar_1, [2, 8, 16, 4]);  mul_scalar_1 = None
        clone_default_1: "f32[2, 8, 16, 4]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_1: "f32[16, 16, 4]" = torch.ops.aten.reshape.default(clone_default_1, [16, 16, 4]);  clone_default_1 = None
        bmm_default: "f32[16, 4, 4]" = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[2, 8, 4, 4]" = torch.ops.aten.reshape.default(bmm_default, [2, 8, 4, 4]);  bmm_default = None
        convert_element_type_default: "f16[2, 1, 1, 4]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float16);  arg3_1 = None
        add_tensor: "f32[2, 8, 4, 4]" = torch.ops.aten.add.Tensor(view_default_2, convert_element_type_default);  view_default_2 = convert_element_type_default = None
        eq_scalar: "b8[2, 8, 4, 4]" = torch.ops.aten.eq.Scalar(add_tensor, -inf)
        logical_not_default: "b8[2, 8, 4, 4]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[2, 8, 4, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[2, 8, 4, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default: "f32[2, 8, 4, 4]" = torch.ops.aten.full.default([2, 8, 4, 4], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        amax_default: "f32[2, 8, 4, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[2, 8, 4, 4]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[2, 8, 4, 4]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[2, 8, 4, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[2, 8, 4, 4]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self: "f32[2, 8, 4, 4]" = torch.ops.aten.where.self(logical_not_default_1, full_default, div_tensor);  logical_not_default_1 = full_default = div_tensor = None
        expand_default_2: "f32[2, 8, 4, 4]" = torch.ops.aten.expand.default(where_self, [2, 8, 4, 4]);  where_self = None
        view_default_3: "f32[16, 4, 4]" = torch.ops.aten.reshape.default(expand_default_2, [16, 4, 4]);  expand_default_2 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:632 in _sfdp_pattern_21, code: value = value.permute([0, 2, 1, 3])
        permute_default_2: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None

        # No stacktrace found for following nodes
        convert_element_type_default_3: "f32[2, 8, 4, 16]" = torch.ops.prims.convert_element_type.default(permute_default_2, torch.float32);  permute_default_2 = None
        expand_default_3: "f32[2, 8, 4, 16]" = torch.ops.aten.expand.default(convert_element_type_default_3, [2, 8, 4, 16]);  convert_element_type_default_3 = None
        clone_default_2: "f32[2, 8, 4, 16]" = torch.ops.aten.clone.default(expand_default_3, memory_format = torch.contiguous_format);  expand_default_3 = None
        view_default_4: "f32[16, 4, 16]" = torch.ops.aten.reshape.default(clone_default_2, [16, 4, 16]);  clone_default_2 = None
        bmm_default_1: "f32[16, 4, 16]" = torch.ops.aten.bmm.default(view_default_3, view_default_4);  view_default_3 = view_default_4 = None
        view_default_5: "f32[2, 8, 4, 16]" = torch.ops.aten.reshape.default(bmm_default_1, [2, 8, 4, 16]);  bmm_default_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:636 in _sfdp_pattern_21, code: return score.float().softmax(dim=-1).type_as(query).matmul(value)
        convert_element_type_default_4: "f16[2, 8, 4, 16]" = torch.ops.prims.convert_element_type.default(view_default_5, torch.float16);  view_default_5 = None
        return (convert_element_type_default_4,)
