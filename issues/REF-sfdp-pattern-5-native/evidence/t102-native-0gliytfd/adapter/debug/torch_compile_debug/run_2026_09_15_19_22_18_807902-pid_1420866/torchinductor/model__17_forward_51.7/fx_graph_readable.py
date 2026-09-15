class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 2, 16, 32]", primals_2: "f32[4, 2, 16, 32]", primals_3: "f32[4, 2, 16, 32]"):
        # No stacktrace found for following nodes
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(primals_2, primals_1, primals_3, 2, 'BNSD', None, None, None, 0.17677669529663687)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:383 in sfdp_pattern_5_v2, code: return attn_weight @ value
        getitem_4: "f32[4, 2, 16, 32]" = npu_fusion_attention_v3_default[0]

        # No stacktrace found for following nodes
        getitem_5: "f32[4, 2, 16, 8]" = npu_fusion_attention_v3_default[1]
        getitem_6: "f32[4, 2, 16, 8]" = npu_fusion_attention_v3_default[2]
        getitem_7: "f32[0]" = npu_fusion_attention_v3_default[3]
        getitem_8: "i64[1]" = npu_fusion_attention_v3_default[4]
        getitem_9: "i64[1]" = npu_fusion_attention_v3_default[5];  npu_fusion_attention_v3_default = None
        return (getitem_4, primals_1, primals_2, primals_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9)
