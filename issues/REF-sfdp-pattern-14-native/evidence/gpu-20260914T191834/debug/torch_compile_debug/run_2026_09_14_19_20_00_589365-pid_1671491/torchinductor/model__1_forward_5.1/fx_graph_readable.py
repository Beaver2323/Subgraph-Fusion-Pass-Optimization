class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 2, 16, 32]", primals_2: "f32[4, 2, 16, 32]", primals_3: "f32[4, 2, 16, 32]"):
        # File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:950 in dot_prod_attention, code: attn_mask = torch.ones(
        full_default: "b8[2, 2]" = torch.ops.aten.full.default([2, 2], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:952 in dot_prod_attention, code: ).tril(diagonal=0)
        iota: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 2]" = torch.ops.aten.unsqueeze.default(iota, -2)
        unsqueeze_1: "i64[2, 1]" = torch.ops.aten.unsqueeze.default(iota, -1);  iota = None
        sub: "i64[2, 2]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        le: "b8[2, 2]" = torch.ops.aten.le.Scalar(sub, 0);  sub = None
        logical_and: "b8[2, 2]" = torch.ops.aten.logical_and.default(le, full_default);  le = full_default = None

        # File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:954 in dot_prod_attention, code: torch.logical_not(attn_mask), -float("inf")
        logical_not: "b8[2, 2]" = torch.ops.aten.logical_not.default(logical_and)

        # File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:953 in dot_prod_attention, code: attn_mask = attn_mask.masked_fill(
        full_default_1: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "b8[2, 2]" = torch.ops.aten.where.self(logical_not, full_default_1, logical_and);  logical_not = full_default_1 = logical_and = None

        # No stacktrace found for following nodes
        permute_default: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(primals_2, [0, 2, 1, 3]);  primals_2 = None
        permute_default_1: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(primals_1, [0, 2, 1, 3]);  primals_1 = None
        permute_default_2: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(primals_3, [0, 2, 1, 3]);  primals_3 = None
        convert_element_type_default: "f32[2, 2]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        constant_pad_nd_default: "f32[2, 8]" = torch.ops.aten.constant_pad_nd.default(convert_element_type_default, [0, 6], 0.0);  convert_element_type_default = None
        slice_tensor: "f32[2, 2]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default, -1, 0, 2);  constant_pad_nd_default = None
        expand_default: "f32[4, 16, 2, 2]" = torch.ops.aten.expand.default(slice_tensor, [4, 16, 2, 2])
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default, permute_default_1, permute_default_2, expand_default, True, scale = 0.3333333333333333);  expand_default = None

        # File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:962 in dot_prod_attention, code: .matmul(v)
        getitem: "f32[4, 16, 2, 32]" = _scaled_dot_product_efficient_attention_default[0]

        # No stacktrace found for following nodes
        getitem_1: "f32[4, 16, 32]" = _scaled_dot_product_efficient_attention_default[1]
        getitem_2: "i64[]" = _scaled_dot_product_efficient_attention_default[2]
        getitem_3: "i64[]" = _scaled_dot_product_efficient_attention_default[3];  _scaled_dot_product_efficient_attention_default = None
        return (getitem, permute_default, permute_default_1, permute_default_2, slice_tensor, getitem, getitem_1, getitem_2, getitem_3)
