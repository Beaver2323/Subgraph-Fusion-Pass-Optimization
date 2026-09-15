


def forward(self, arg0_1, arg1_1, arg2_1):
    npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(arg1_1, arg0_1, arg2_1, 2, 'BNSD', None, None, None, 0.4);  arg1_1 = arg0_1 = arg2_1 = None
    getitem = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
    return (getitem,)
    