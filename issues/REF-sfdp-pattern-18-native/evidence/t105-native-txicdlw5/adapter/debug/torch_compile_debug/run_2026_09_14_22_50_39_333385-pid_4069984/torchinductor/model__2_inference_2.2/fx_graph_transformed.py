class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "b8[2, 2]", arg1_1: "f32[1, 2, 16, 32]", arg2_1: "f32[1, 2, 16, 32]", arg3_1: "f32[1, 2, 16, 32]"):
        # No stacktrace found for following nodes
        permute_default_2: "f32[1, 16, 2, 32]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1131 in dot_prod_attention, code: key = key.permute([0, 2, 1, 3])
        permute_default: "f32[1, 16, 2, 32]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1132 in dot_prod_attention, code: value = value.permute([0, 2, 1, 3])
        permute_default_1: "f32[1, 16, 2, 32]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1, 3]);  arg3_1 = None

        # No stacktrace found for following nodes
        logical_not_default: "b8[2, 2]" = torch.ops.aten.logical_not.default(arg0_1);  arg0_1 = None
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(permute_default_2, permute_default, permute_default_1, 16, 'BNSD', None, None, logical_not_default, 0.17677669529663687, 1.0, 2147483647, 2147483647, 2);  permute_default_2 = logical_not_default = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1146 in dot_prod_attention, code: ).matmul(value)
        getitem: "f32[1, 16, 2, 32]" = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1148 in dot_prod_attention, code: key.permute([0, 2, 1, 3]),
        permute_4: "f32[1, 2, 16, 32]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1149 in dot_prod_attention, code: value.permute([0, 2, 1, 3]),
        permute_5: "f32[1, 2, 16, 32]" = torch.ops.aten.permute.default(permute_default_1, [0, 2, 1, 3]);  permute_default_1 = None
        return (getitem, permute_4, permute_5)
