class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 2, 16, 32]", primals_2: "f32[4, 2, 16, 32]", primals_3: "f32[4, 2, 16, 32]", getitem_6: "f32[4, 2, 16, 32]", getitem_7: "f32[4, 2, 16, 8]", getitem_8: "f32[4, 2, 16, 8]", getitem_9: "f32[0]", getitem_10: "i64[1]", getitem_11: "i64[1]", tangents_1: "f32[4, 2, 16, 32]"):
        # No stacktrace found for following nodes
        npu_fusion_attention_grad_v3_default = torch.ops.npu.npu_fusion_attention_grad_v3.default(primals_2, primals_1, primals_3, tangents_1, 2, 'BNSD', softmax_max = getitem_7, softmax_sum = getitem_8, softmax_in = getitem_9, attention_in = getitem_6, scale_value = 0.3333333333333333, keep_prob = 0.6, seed = getitem_10, offset = getitem_11);  primals_2 = primals_1 = primals_3 = tangents_1 = getitem_7 = getitem_8 = getitem_9 = getitem_6 = getitem_10 = getitem_11 = None

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:330 in dot_prod_attention, code: torch.matmul(query, key.transpose(-2, -1)).div(3.0).softmax(dim=-1),
        getitem_12: "f32[4, 2, 16, 32]" = npu_fusion_attention_grad_v3_default[0]
        getitem_13: "f32[4, 2, 16, 32]" = npu_fusion_attention_grad_v3_default[1]

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:334 in dot_prod_attention, code: ).matmul(value)
        getitem_14: "f32[4, 2, 16, 32]" = npu_fusion_attention_grad_v3_default[2];  npu_fusion_attention_grad_v3_default = None
        return (getitem_13, getitem_12, getitem_14)
