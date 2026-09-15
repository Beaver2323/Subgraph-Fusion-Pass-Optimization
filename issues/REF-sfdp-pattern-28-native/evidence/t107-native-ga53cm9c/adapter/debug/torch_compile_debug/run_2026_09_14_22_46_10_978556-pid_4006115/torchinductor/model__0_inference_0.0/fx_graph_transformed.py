class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[2, 3, 4, 16, 8]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1747 in dot_prod_attention, code: q, k, v = qkv.permute(1, 0, 2, 4, 3).unbind(0)
        permute: "f16[3, 2, 4, 8, 16]" = torch.ops.aten.permute.default(arg0_1, [1, 0, 2, 4, 3]);  arg0_1 = None
        unbind = torch.ops.aten.unbind.int(permute);  permute = None
        getitem: "f16[2, 4, 8, 16]" = unbind[0]
        getitem_1: "f16[2, 4, 8, 16]" = unbind[1]
        getitem_2: "f16[2, 4, 8, 16]" = unbind[2];  unbind = None

        # No stacktrace found for following nodes
        clone_default: "f16[2, 4, 8, 16]" = torch.ops.aten.clone.default(getitem, memory_format = torch.contiguous_format);  getitem = None
        clone_default_1: "f16[2, 4, 8, 16]" = torch.ops.aten.clone.default(getitem_1, memory_format = torch.contiguous_format);  getitem_1 = None
        clone_default_2: "f16[2, 4, 8, 16]" = torch.ops.aten.clone.default(getitem_2, memory_format = torch.contiguous_format);  getitem_2 = None
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(clone_default, clone_default_1, clone_default_2, 4, 'BNSD', None, None, None, 0.2);  clone_default = clone_default_1 = clone_default_2 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1754 in dot_prod_attention, code: return attn_weights.matmul(v)
        getitem_3: "f16[2, 4, 8, 16]" = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
        return (getitem_3,)
