class GraphModule(torch.nn.Module):
    def forward(self, permute_default: "f32[4, 16, 2, 32]", permute_default_1: "f32[4, 16, 2, 32]", permute_default_2: "f32[4, 16, 2, 32]", slice_tensor: "f32[2, 2]", getitem: "f32[4, 16, 2, 32]", getitem_1: "f32[4, 16, 32]", getitem_2: "i64[]", getitem_3: "i64[]", tangents_1: "f32[4, 16, 2, 32]"):
        # No stacktrace found for following nodes
        expand_default: "f32[4, 16, 2, 2]" = torch.ops.aten.expand.default(slice_tensor, [4, 16, 2, 2]);  slice_tensor = None
        _scaled_dot_product_efficient_attention_backward_default = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(tangents_1, permute_default, permute_default_1, permute_default_2, expand_default, getitem, getitem_1, getitem_2, getitem_3, 0.0, [True, True, True, False], scale = 0.3333333333333333);  tangents_1 = permute_default = permute_default_1 = permute_default_2 = expand_default = getitem = getitem_1 = getitem_2 = getitem_3 = None
        getitem_4: "f32[4, 16, 2, 32]" = _scaled_dot_product_efficient_attention_backward_default[0]
        getitem_5: "f32[4, 16, 2, 32]" = _scaled_dot_product_efficient_attention_backward_default[1]
        getitem_6: "f32[4, 16, 2, 32]" = _scaled_dot_product_efficient_attention_backward_default[2];  _scaled_dot_product_efficient_attention_backward_default = None

        # Backward of forward node: File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:958 in dot_prod_attention, code: v = value.permute(0, 2, 1, 3)
        permute_default_3: "f32[4, 2, 16, 32]" = torch.ops.aten.permute.default(getitem_6, [0, 2, 1, 3]);  getitem_6 = None

        # Backward of forward node: File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:957 in dot_prod_attention, code: k = key.permute(0, 2, 1, 3)
        permute_default_4: "f32[4, 2, 16, 32]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None

        # Backward of forward node: File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:956 in dot_prod_attention, code: q = query.permute(0, 2, 1, 3)
        permute_default_5: "f32[4, 2, 16, 32]" = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3]);  getitem_4 = None
        return (permute_default_4, permute_default_5, permute_default_3)
