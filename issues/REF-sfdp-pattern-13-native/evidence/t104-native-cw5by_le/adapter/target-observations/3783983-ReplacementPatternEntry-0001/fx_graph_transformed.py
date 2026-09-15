


def forward(self, arg0_1, arg1_1, arg2_1):
    unsqueeze_default = torch.ops.aten.unsqueeze.default(arg1_1, 0);  arg1_1 = None
    unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
    unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
    npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, 4, 'BNSD');  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = None
    getitem = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
    squeeze_dim = torch.ops.aten.squeeze.dim(getitem, 0);  getitem = None
    return (squeeze_dim,)
    