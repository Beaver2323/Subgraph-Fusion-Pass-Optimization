class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 2, 16, 32]", arg1_1: "f32[4, 2, 16, 32]", arg2_1: "f32[4, 2, 16, 32]"):
        # No stacktrace found for following nodes
        permute_default: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
        permute_default_1: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        permute_default_2: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None

        # File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:952 in dot_prod_attention, code: ).tril(diagonal=0)
        iota: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 2]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[2, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub: "i64[2, 2]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        le: "b8[2, 2]" = torch.ops.aten.le.Scalar(sub, 0);  sub = None

        # File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:950 in dot_prod_attention, code: attn_mask = torch.ones(
        full_default: "b8[2, 2]" = torch.ops.aten.full.default([2, 2], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:952 in dot_prod_attention, code: ).tril(diagonal=0)
        logical_and: "b8[2, 2]" = torch.ops.aten.logical_and.default(le, full_default);  le = full_default = None

        # File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:954 in dot_prod_attention, code: torch.logical_not(attn_mask), -float("inf")
        logical_not: "b8[2, 2]" = torch.ops.aten.logical_not.default(logical_and)

        # File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:953 in dot_prod_attention, code: attn_mask = attn_mask.masked_fill(
        full_default_1: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "b8[2, 2]" = torch.ops.aten.where.self(logical_not, full_default_1, logical_and);  logical_not = full_default_1 = logical_and = None

        # No stacktrace found for following nodes
        convert_element_type_default: "f32[2, 2]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        constant_pad_nd_default: "f32[2, 8]" = torch.ops.aten.constant_pad_nd.default(convert_element_type_default, [0, 6], 0.0);  convert_element_type_default = None
        slice_tensor: "f32[2, 2]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default, -1, 0, 2);  constant_pad_nd_default = None
        expand_default: "f32[4, 16, 2, 2]" = torch.ops.aten.expand.default(slice_tensor, [4, 16, 2, 2]);  slice_tensor = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default, permute_default_1, permute_default_2, expand_default, False, scale = 0.3333333333333333);  permute_default = permute_default_1 = permute_default_2 = expand_default = None

        # File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:962 in dot_prod_attention, code: .matmul(v)
        getitem: "f32[4, 16, 2, 32]" = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
        return (getitem,)
