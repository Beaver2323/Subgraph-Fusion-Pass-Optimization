


def forward(self, arg0_1, arg1_1, arg2_1):
    permute_default = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
    permute_default_1 = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
    permute_default_2 = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
    npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(permute_default, permute_default_1, permute_default_2, 16, 'BNSD', None, None, None, 0.17677669529663687);  permute_default = permute_default_1 = permute_default_2 = None
    getitem = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
    return (getitem,)
    