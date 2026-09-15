class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 2, 16, 32]", arg1_1: "f32[4, 2, 16, 32]", arg2_1: "f32[4, 2, 16, 32]"):
        # No stacktrace found for following nodes
        permute_default: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
        permute_default_1: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        permute_default_2: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(permute_default, permute_default_1, permute_default_2, 16, 'BNSD', None, None, None, 0.17677669529663687);  permute_default = permute_default_1 = permute_default_2 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:827 in dot_prod_attention, code: return attn_weight.matmul(v)
        getitem: "f32[4, 16, 2, 32]" = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
        return (getitem,)
