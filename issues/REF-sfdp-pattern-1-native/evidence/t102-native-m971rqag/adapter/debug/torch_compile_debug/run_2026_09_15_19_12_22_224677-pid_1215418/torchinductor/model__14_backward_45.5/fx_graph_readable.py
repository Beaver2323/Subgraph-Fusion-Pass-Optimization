class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 2, 16, 32]", primals_2: "f32[4, 2, 16, 32]", primals_3: "f32[4, 2, 16, 32]", getitem_7: "f32[0]", tangents_1: "f32[4, 2, 16, 32]"):
        # ac_graph_id: 4 - PREFER_RECOMPUTE No stacktrace found for following nodes
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(primals_1, primals_2, primals_3, 2, 'BNSD', None, None, None, 0.17677669529663687)

        # ac_graph_id: 4 - PREFER_RECOMPUTE Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:160 in dot_prod_attention, code: .matmul(value)
        getitem_4: "f32[4, 2, 16, 32]" = npu_fusion_attention_v3_default[0]

        # No stacktrace found for following nodes
        getitem_5: "f32[4, 2, 16, 8]" = npu_fusion_attention_v3_default[1]
        getitem_6: "f32[4, 2, 16, 8]" = npu_fusion_attention_v3_default[2]
        getitem_8: "i64[1]" = npu_fusion_attention_v3_default[4]
        getitem_9: "i64[1]" = npu_fusion_attention_v3_default[5];  npu_fusion_attention_v3_default = None
        npu_fusion_attention_grad_v3_default = torch.ops.npu.npu_fusion_attention_grad_v3.default(primals_1, primals_2, primals_3, tangents_1, 2, 'BNSD', softmax_max = getitem_5, softmax_sum = getitem_6, softmax_in = getitem_7, attention_in = getitem_4, scale_value = 0.17677669529663687, seed = getitem_8, offset = getitem_9);  primals_1 = primals_2 = primals_3 = tangents_1 = getitem_5 = getitem_6 = getitem_7 = getitem_4 = getitem_8 = getitem_9 = None

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:157 in dot_prod_attention, code: torch.matmul(query, key.transpose(-2, -1))
        getitem_10: "f32[4, 2, 16, 32]" = npu_fusion_attention_grad_v3_default[0]
        getitem_11: "f32[4, 2, 16, 32]" = npu_fusion_attention_grad_v3_default[1]

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:160 in dot_prod_attention, code: .matmul(value)
        getitem_12: "f32[4, 2, 16, 32]" = npu_fusion_attention_grad_v3_default[2];  npu_fusion_attention_grad_v3_default = None
        return (getitem_10, getitem_11, getitem_12)
