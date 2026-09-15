


def forward(self, arg0_1, arg1_1, arg2_1):
    unsqueeze_default = torch.ops.aten.unsqueeze.default(arg1_1, 0);  arg1_1 = None
    unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
    unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
    _scaled_dot_product_flash_attention_default = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, scale = 1.0);  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = None
    getitem = _scaled_dot_product_flash_attention_default[0];  _scaled_dot_product_flash_attention_default = None
    squeeze_dim = torch.ops.aten.squeeze.dim(getitem, 0);  getitem = None
    return (squeeze_dim,)
    