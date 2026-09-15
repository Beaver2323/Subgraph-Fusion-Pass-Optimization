class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f16[2, 2, 8, 16]", primals_2: "f16[2, 2, 8, 16]", primals_3: "f16[2, 2, 8, 16]", getitem_4: "f16[2, 2, 8, 16]", getitem_5: "f32[2, 2, 8, 8]", getitem_6: "f32[2, 2, 8, 8]", getitem_7: "f16[0]", getitem_8: "i64[1]", getitem_9: "i64[1]", tangents_1: "f16[2, 2, 8, 16]"):
        # No stacktrace found for following nodes
        npu_fusion_attention_grad_v3_default = torch.ops.npu.npu_fusion_attention_grad_v3.default(primals_2, primals_1, primals_3, tangents_1, 2, 'BNSD', softmax_max = getitem_5, softmax_sum = getitem_6, softmax_in = getitem_7, attention_in = getitem_4, scale_value = 0.25, seed = getitem_8, offset = getitem_9);  primals_2 = primals_1 = primals_3 = tangents_1 = getitem_5 = getitem_6 = getitem_7 = getitem_4 = getitem_8 = getitem_9 = None

        # Backward of forward node: File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/attention_training_boundary.py:47 in mul, code: return (q @ k.transpose(-2,-1)).mul(0.25).softmax(-1) @ v
        getitem_10: "f16[2, 2, 8, 16]" = npu_fusion_attention_grad_v3_default[0]
        getitem_11: "f16[2, 2, 8, 16]" = npu_fusion_attention_grad_v3_default[1]
        getitem_12: "f16[2, 2, 8, 16]" = npu_fusion_attention_grad_v3_default[2];  npu_fusion_attention_grad_v3_default = None
        return (getitem_11, getitem_10, getitem_12)
