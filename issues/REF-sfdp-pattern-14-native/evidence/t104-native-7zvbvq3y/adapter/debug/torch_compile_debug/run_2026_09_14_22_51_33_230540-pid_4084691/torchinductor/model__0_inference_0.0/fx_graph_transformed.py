class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 2, 16, 32]", arg1_1: "f32[4, 2, 16, 32]", arg2_1: "f32[4, 2, 16, 32]"):
        # No stacktrace found for following nodes
        permute_default: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
        mul_scalar: "f32[4, 16, 2, 32]" = torch.ops.aten.mul.Scalar(permute_default, 0.5773502691896257);  permute_default = None
        expand_default: "f32[4, 16, 2, 32]" = torch.ops.aten.expand.default(mul_scalar, [4, 16, 2, 32]);  mul_scalar = None
        clone_default: "f32[4, 16, 2, 32]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default: "f32[64, 2, 32]" = torch.ops.aten.reshape.default(clone_default, [64, 2, 32]);  clone_default = None
        permute_default_1: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        permute_default_3: "f32[4, 16, 32, 2]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        mul_scalar_1: "f32[4, 16, 32, 2]" = torch.ops.aten.mul.Scalar(permute_default_3, 0.5773502691896257);  permute_default_3 = None
        expand_default_1: "f32[4, 16, 32, 2]" = torch.ops.aten.expand.default(mul_scalar_1, [4, 16, 32, 2]);  mul_scalar_1 = None
        clone_default_1: "f32[4, 16, 32, 2]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_1: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(clone_default_1, [64, 32, 2]);  clone_default_1 = None
        bmm_default: "f32[64, 2, 2]" = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[4, 16, 2, 2]" = torch.ops.aten.reshape.default(bmm_default, [4, 16, 2, 2]);  bmm_default = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:952 in dot_prod_attention, code: ).tril(diagonal=0)
        iota: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze: "i64[1, 2]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze_1: "i64[2, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub: "i64[2, 2]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        le: "b8[2, 2]" = torch.ops.aten.le.Scalar(sub, 0);  sub = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:950 in dot_prod_attention, code: attn_mask = torch.ones(
        full_default: "b8[2, 2]" = torch.ops.aten.full.default([2, 2], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:952 in dot_prod_attention, code: ).tril(diagonal=0)
        logical_and: "b8[2, 2]" = torch.ops.aten.logical_and.default(le, full_default);  le = full_default = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:954 in dot_prod_attention, code: torch.logical_not(attn_mask), -float("inf")
        logical_not: "b8[2, 2]" = torch.ops.aten.logical_not.default(logical_and)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:953 in dot_prod_attention, code: attn_mask = attn_mask.masked_fill(
        full_default_1: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "b8[2, 2]" = torch.ops.aten.where.self(logical_not, full_default_1, logical_and);  logical_not = full_default_1 = logical_and = None

        # No stacktrace found for following nodes
        convert_element_type_default: "f32[2, 2]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        add_tensor: "f32[4, 16, 2, 2]" = torch.ops.aten.add.Tensor(view_default_2, convert_element_type_default);  view_default_2 = convert_element_type_default = None
        eq_scalar: "b8[4, 16, 2, 2]" = torch.ops.aten.eq.Scalar(add_tensor, -inf)
        logical_not_default: "b8[4, 16, 2, 2]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[4, 16, 2, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[4, 16, 2, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_2: "f32[4, 16, 2, 2]" = torch.ops.aten.full.default([4, 16, 2, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        amax_default: "f32[4, 16, 2, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[4, 16, 2, 2]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[4, 16, 2, 2]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[4, 16, 2, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[4, 16, 2, 2]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self: "f32[4, 16, 2, 2]" = torch.ops.aten.where.self(logical_not_default_1, full_default_2, div_tensor);  logical_not_default_1 = full_default_2 = div_tensor = None
        expand_default_2: "f32[4, 16, 2, 2]" = torch.ops.aten.expand.default(where_self, [4, 16, 2, 2]);  where_self = None
        view_default_3: "f32[64, 2, 2]" = torch.ops.aten.reshape.default(expand_default_2, [64, 2, 2]);  expand_default_2 = None
        permute_default_2: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
        expand_default_3: "f32[4, 16, 2, 32]" = torch.ops.aten.expand.default(permute_default_2, [4, 16, 2, 32]);  permute_default_2 = None
        clone_default_2: "f32[4, 16, 2, 32]" = torch.ops.aten.clone.default(expand_default_3, memory_format = torch.contiguous_format);  expand_default_3 = None
        view_default_4: "f32[64, 2, 32]" = torch.ops.aten.reshape.default(clone_default_2, [64, 2, 32]);  clone_default_2 = None
        bmm_default_1: "f32[64, 2, 32]" = torch.ops.aten.bmm.default(view_default_3, view_default_4);  view_default_3 = view_default_4 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:962 in dot_prod_attention, code: .matmul(v)
        view_default_5: "f32[4, 16, 2, 32]" = torch.ops.aten.reshape.default(bmm_default_1, [4, 16, 2, 32]);  bmm_default_1 = None
        return (view_default_5,)
