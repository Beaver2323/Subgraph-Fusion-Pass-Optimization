class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f16[2, 4, 8, 16]", primals_2: "f16[2, 4, 8, 16]", primals_3: "f16[1, 1, 8, 8]", primals_4: "f16[2, 4, 8, 16]"):
        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:131 in _sfdp_pattern_5, code: (query @ key.transpose(-2, -1) / (inv_scale)) + attn_mask, dim=-1
        permute: "f16[2, 4, 16, 8]" = torch.ops.aten.permute.default(primals_1, [0, 1, 3, 2]);  primals_1 = None
        expand: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(primals_2, [2, 4, 8, 16])
        view: "f16[8, 8, 16]" = torch.ops.aten.reshape.default(expand, [8, 8, 16]);  expand = None
        expand_1: "f16[2, 4, 16, 8]" = torch.ops.aten.expand.default(permute, [2, 4, 16, 8])
        view_1: "f16[8, 16, 8]" = torch.ops.aten.reshape.default(expand_1, [8, 16, 8]);  expand_1 = None
        bmm: "f16[8, 8, 8]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f16[2, 4, 8, 8]" = torch.ops.aten.reshape.default(bmm, [2, 4, 8, 8]);  bmm = None
        div: "f16[2, 4, 8, 8]" = torch.ops.aten.div.Tensor(view_2, 0.66666);  view_2 = None
        add: "f16[2, 4, 8, 8]" = torch.ops.aten.add.Tensor(div, primals_3);  div = primals_3 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:130 in _sfdp_pattern_5, code: attn_weight = torch.softmax(
        convert_element_type_2: "f32[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        amax: "f32[2, 4, 8, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub: "f32[2, 4, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[2, 4, 8, 8]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[2, 4, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[2, 4, 8, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "f16[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:134 in _sfdp_pattern_5, code: return attn_weight @ value
        expand_2: "f16[2, 4, 8, 8]" = torch.ops.aten.expand.default(convert_element_type_3, [2, 4, 8, 8])
        view_3: "f16[8, 8, 8]" = torch.ops.aten.reshape.default(expand_2, [8, 8, 8]);  expand_2 = None
        expand_3: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(primals_4, [2, 4, 8, 16])
        view_4: "f16[8, 8, 16]" = torch.ops.aten.reshape.default(expand_3, [8, 8, 16]);  expand_3 = None
        bmm_1: "f16[8, 8, 16]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f16[2, 4, 8, 16]" = torch.ops.aten.reshape.default(bmm_1, [2, 4, 8, 16]);  bmm_1 = None
        return (view_5, primals_2, primals_4, permute, convert_element_type_3)
