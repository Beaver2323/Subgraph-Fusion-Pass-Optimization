class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 2, 16, 32]", arg1_1: "f32[1, 2, 16, 32]", arg2_1: "f32[1, 2, 16, 32]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1434 in dot_prod_attention, code: attn_mask = torch.full((1, 1, 1, 2), 0.0, device=query.device)
        full_default: "f32[1, 1, 1, 2]" = torch.ops.aten.full.default([1, 1, 1, 2], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  full_default = None

        # No stacktrace found for following nodes
        permute_default: "f32[1, 16, 2, 32]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1436 in dot_prod_attention, code: key = key.transpose(1, 2)
        permute_default_1: "f32[1, 16, 2, 32]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1437 in dot_prod_attention, code: value = value.transpose(1, 2)
        permute_default_2: "f32[1, 16, 2, 32]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None

        # No stacktrace found for following nodes
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(permute_default, permute_default_1, permute_default_2, 16, 'BNSD');  permute_default = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1441 in dot_prod_attention, code: return attn_weights.matmul(value), key, value
        getitem: "f32[1, 16, 2, 32]" = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
        return (getitem, permute_default_1, permute_default_2)
