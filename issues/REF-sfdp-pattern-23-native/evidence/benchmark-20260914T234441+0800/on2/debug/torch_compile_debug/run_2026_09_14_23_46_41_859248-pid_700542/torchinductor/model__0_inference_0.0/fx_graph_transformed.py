class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[2, 4, 8, 16]", arg1_1: "f16[2, 4, 8, 16]", arg2_1: "f16[2, 4, 8, 16]"):
        # No stacktrace found for following nodes
        permute_default: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:689 in _sfdp_pattern_23, code: key = key.permute([0, 2, 1, 3])
        permute_default_1: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:690 in _sfdp_pattern_23, code: value = value.permute([0, 2, 1, 3])
        permute_default_2: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None

        # No stacktrace found for following nodes
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(permute_default, permute_default_1, permute_default_2, 8, 'BNSD');  permute_default = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:694 in _sfdp_pattern_23, code: return score.float().softmax(dim=-1).type_as(query).matmul(value), key, value
        getitem: "f16[2, 8, 4, 16]" = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
        return (getitem, permute_default_1, permute_default_2)
