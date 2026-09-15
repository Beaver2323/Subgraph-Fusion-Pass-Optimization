class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[1024, 128, 128]", arg1_1: "f16[1024, 128, 128]", arg2_1: "f16[1024, 128, 128]"):
        # No stacktrace found for following nodes
        unsqueeze_default: "f16[1, 1024, 128, 128]" = torch.ops.aten.unsqueeze.default(arg1_1, 0);  arg1_1 = None
        unsqueeze_default_1: "f16[1, 1024, 128, 128]" = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
        unsqueeze_default_2: "f16[1, 1024, 128, 128]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, 1024, 'BNSD');  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = None
        getitem: "f16[1, 1024, 128, 128]" = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:344 in _sfdp_pattern_13, code: return torch.bmm(attn_weight, value)
        squeeze_dim: "f16[1024, 128, 128]" = torch.ops.aten.squeeze.dim(getitem, 0);  getitem = None
        return (squeeze_dim,)
