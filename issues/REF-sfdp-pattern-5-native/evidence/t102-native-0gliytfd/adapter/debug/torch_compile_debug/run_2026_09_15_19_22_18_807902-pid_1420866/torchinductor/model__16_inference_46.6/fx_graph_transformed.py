class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 2, 16, 32]", arg1_1: "f32[4, 2, 16, 32]", arg2_1: "f32[4, 2, 16, 32]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:376 in sfdp_pattern_5_v2, code: attn_mask = torch.zeros(
        full_default: "b8[16, 16]" = torch.ops.aten.full.default([16, 16], False, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  full_default = None

        # No stacktrace found for following nodes
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(arg1_1, arg0_1, arg2_1, 2, 'BNSD', None, None, None, 0.17677669529663687);  arg1_1 = arg0_1 = arg2_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:383 in sfdp_pattern_5_v2, code: return attn_weight @ value
        getitem: "f32[4, 2, 16, 32]" = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
        return (getitem,)
