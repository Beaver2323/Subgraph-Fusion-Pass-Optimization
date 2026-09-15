


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    permute_default = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
    permute_default_1 = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
    permute_default_2 = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
    expand_default = torch.ops.aten.expand.default(arg3_1, [2, 8, 4, 4]);  arg3_1 = None
    logical_not_default = torch.ops.aten.logical_not.default(expand_default);  expand_default = None
    npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(permute_default_2, permute_default, permute_default_1, 8, 'BNSD', None, None, logical_not_default, 1.5000150001500014, 1.0, 2147483647, 2147483647, 2);  permute_default_2 = logical_not_default = None
    getitem = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
    return (getitem, permute_default, permute_default_1)
    