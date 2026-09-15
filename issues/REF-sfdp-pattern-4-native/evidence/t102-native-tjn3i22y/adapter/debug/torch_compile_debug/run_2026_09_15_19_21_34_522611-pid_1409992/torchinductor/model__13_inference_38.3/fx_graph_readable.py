class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 2, 16, 32]", arg1_1: "f32[4, 2, 16, 32]", arg2_1: "f32[4, 2, 16, 32]"):
        # ac_graph_id: 2 - PREFER_RECOMPUTE No stacktrace found for following nodes
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(arg0_1, arg1_1, arg2_1, 2, 'BNSD', None, None, None, 0.4);  arg0_1 = arg1_1 = arg2_1 = None

        # ac_graph_id: 2 - PREFER_RECOMPUTE File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:353 in dot_prod_attention, code: ).matmul(value)
        getitem: "f32[4, 2, 16, 32]" = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
        return (getitem,)
