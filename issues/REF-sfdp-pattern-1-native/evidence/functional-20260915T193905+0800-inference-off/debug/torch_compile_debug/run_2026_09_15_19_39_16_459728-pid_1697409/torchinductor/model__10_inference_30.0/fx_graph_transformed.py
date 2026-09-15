class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[2, 4, 8, 16]", arg1_1: "f16[2, 4, 8, 16]", arg2_1: "f16[2, 4, 8, 16]"):
        # No stacktrace found for following nodes
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(arg1_1, arg0_1, arg2_1, 4, 'BNSD', None, None, None, 0.5);  arg1_1 = arg0_1 = arg2_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:48 in _sfdp_pattern_1, code: .matmul(value)
        getitem: "f16[2, 4, 8, 16]" = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
        return (getitem,)
