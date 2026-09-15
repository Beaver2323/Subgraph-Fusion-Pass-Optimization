class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[2, 4, 8, 16]", arg1_1: "f16[2, 4, 8, 16]", arg2_1: "f16[2, 4, 8, 16]", arg3_1: "b8[2, 1, 1, 4]"):
        # No stacktrace found for following nodes
        permute_default_2: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:508 in _sfdp_pattern_18, code: key = key.permute([0, 2, 1, 3])
        permute_default: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:509 in _sfdp_pattern_18, code: value = value.permute([0, 2, 1, 3])
        permute_default_1: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None

        # No stacktrace found for following nodes
        expand_default: "b8[2, 8, 4, 4]" = torch.ops.aten.expand.default(arg3_1, [2, 8, 4, 4]);  arg3_1 = None
        logical_not_default: "b8[2, 8, 4, 4]" = torch.ops.aten.logical_not.default(expand_default);  expand_default = None
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(permute_default_2, permute_default, permute_default_1, 8, 'BNSD', None, None, logical_not_default, 1.5000150001500014, 1.0, 2147483647, 2147483647, 2);  permute_default_2 = logical_not_default = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:524 in _sfdp_pattern_18, code: torch.nn.functional.dropout(attn_weights.softmax(dim=-1), dropout_p).matmul(
        getitem: "f16[2, 8, 4, 16]" = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
        return (getitem, permute_default, permute_default_1)
