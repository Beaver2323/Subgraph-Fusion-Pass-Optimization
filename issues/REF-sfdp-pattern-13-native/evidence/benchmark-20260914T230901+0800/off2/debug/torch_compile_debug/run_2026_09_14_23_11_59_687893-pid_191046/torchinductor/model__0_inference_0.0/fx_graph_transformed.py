class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[1024, 128, 128]", arg1_1: "f16[1024, 128, 128]", arg2_1: "f16[1024, 128, 128]"):
        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:342 in _sfdp_pattern_13, code: attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
        permute: "f16[1024, 128, 128]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1]);  arg0_1 = None
        bmm: "f16[1024, 128, 128]" = torch.ops.aten.bmm.default(arg1_1, permute);  arg1_1 = permute = None
        convert_element_type_2: "f32[1024, 128, 128]" = torch.ops.prims.convert_element_type.default(bmm, torch.float32);  bmm = None
        amax: "f32[1024, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub: "f32[1024, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[1024, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[1024, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1024, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "f16[1024, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.float16);  div = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:344 in _sfdp_pattern_13, code: return torch.bmm(attn_weight, value)
        bmm_1: "f16[1024, 128, 128]" = torch.ops.aten.bmm.default(convert_element_type_3, arg2_1);  convert_element_type_3 = arg2_1 = None
        return (bmm_1,)
