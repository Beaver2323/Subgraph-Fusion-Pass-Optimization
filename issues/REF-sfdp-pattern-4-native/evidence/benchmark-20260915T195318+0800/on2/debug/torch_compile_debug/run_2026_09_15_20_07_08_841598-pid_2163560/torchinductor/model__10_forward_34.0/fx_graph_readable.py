class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f16[2, 4, 8, 16]", primals_2: "f16[2, 4, 8, 16]", primals_3: "f16[2, 4, 8, 16]"):
        # No stacktrace found for following nodes
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(primals_2, primals_1, primals_3, 4, 'BNSD', None, None, None, 2.0, 0.99999999999)

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:113 in _sfdp_pattern_4, code: ).matmul(value)
        getitem_6: "f16[2, 4, 8, 16]" = npu_fusion_attention_v3_default[0]

        # No stacktrace found for following nodes
        getitem_7: "f32[2, 4, 8, 8]" = npu_fusion_attention_v3_default[1]
        getitem_8: "f32[2, 4, 8, 8]" = npu_fusion_attention_v3_default[2]
        getitem_9: "f16[0]" = npu_fusion_attention_v3_default[3]
        getitem_10: "i64[1]" = npu_fusion_attention_v3_default[4]
        getitem_11: "i64[1]" = npu_fusion_attention_v3_default[5];  npu_fusion_attention_v3_default = None
        return (getitem_6, primals_1, primals_2, primals_3, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11)
