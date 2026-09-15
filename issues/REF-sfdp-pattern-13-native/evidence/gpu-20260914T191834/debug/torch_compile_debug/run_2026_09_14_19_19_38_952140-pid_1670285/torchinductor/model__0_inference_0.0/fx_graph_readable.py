class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[4, 8, 16]", arg1_1: "f16[4, 8, 16]", arg2_1: "f16[4, 8, 16]"):
        # No stacktrace found for following nodes
        unsqueeze_default: "f16[1, 4, 8, 16]" = torch.ops.aten.unsqueeze.default(arg1_1, 0);  arg1_1 = None
        unsqueeze_default_1: "f16[1, 4, 8, 16]" = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
        unsqueeze_default_2: "f16[1, 4, 8, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        _scaled_dot_product_flash_attention_default = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, scale = 1.0);  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = None
        getitem: "f16[1, 4, 8, 16]" = _scaled_dot_product_flash_attention_default[0];  _scaled_dot_product_flash_attention_default = None

        # File: /data/z50063656/src/pytorch/test/inductor/test_fused_attention.py:897 in dot_prod_attention, code: return torch.bmm(attn_weight, value)
        squeeze_dim: "f16[4, 8, 16]" = torch.ops.aten.squeeze.dim(getitem, 0);  getitem = None
        return (squeeze_dim,)
