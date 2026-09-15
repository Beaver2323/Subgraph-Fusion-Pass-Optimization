


def forward(self, arg0_1, arg1_1, arg2_1):
    full_default = torch.ops.aten.full.default([16, 16], False, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  full_default = None
    npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(arg0_1, arg1_1, arg2_1, 2, 'BNSD', None, None, None, 0.17677669529663687);  arg0_1 = arg1_1 = arg2_1 = None
    getitem = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
    return (getitem,)
    