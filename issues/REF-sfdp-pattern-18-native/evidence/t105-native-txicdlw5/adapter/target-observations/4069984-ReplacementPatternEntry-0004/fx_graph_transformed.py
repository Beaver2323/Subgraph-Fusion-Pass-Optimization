


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    permute_default = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
    permute_default_1 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1, 3]);  arg3_1 = None
    permute_default_2 = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
    logical_not_default = torch.ops.aten.logical_not.default(arg0_1);  arg0_1 = None
    npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(permute_default_2, permute_default, permute_default_1, 16, 'BNSD', None, None, logical_not_default, 0.17370597841489152, 1.0, 2147483647, 2147483647, 2);  permute_default_2 = logical_not_default = None
    getitem = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
    permute_4 = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
    permute_5 = torch.ops.aten.permute.default(permute_default_1, [0, 2, 1, 3]);  permute_default_1 = None
    return (getitem, permute_4, permute_5)
    