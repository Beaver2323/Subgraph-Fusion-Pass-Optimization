class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[2, 4, 8, 16]", arg1_1: "f16[2, 4, 8, 16]", arg2_1: "f16[2, 4, 8, 16]", arg3_1: "f32[2, 1, 1, 4]"):
        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:656 in _sfdp_pattern_22, code: query = query.permute([0, 2, 1, 3])
        permute: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:657 in _sfdp_pattern_22, code: key = key.permute([0, 2, 1, 3])
        permute_1: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:658 in _sfdp_pattern_22, code: value = value.permute([0, 2, 1, 3])
        permute_2: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:659 in _sfdp_pattern_22, code: score = torch.matmul(query, key.permute(0, 1, 3, 2))
        permute_3: "f16[2, 8, 16, 4]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2])
        expand: "f16[2, 8, 4, 16]" = torch.ops.aten.expand.default(permute, [2, 8, 4, 16]);  permute = None
        clone: "f16[2, 8, 4, 16]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f16[16, 4, 16]" = torch.ops.aten.view.default(clone, [16, 4, 16]);  clone = None
        expand_1: "f16[2, 8, 16, 4]" = torch.ops.aten.expand.default(permute_3, [2, 8, 16, 4]);  permute_3 = None
        clone_1: "f16[2, 8, 16, 4]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1: "f16[16, 16, 4]" = torch.ops.aten.view.default(clone_1, [16, 16, 4]);  clone_1 = None
        bmm: "f16[16, 4, 4]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f16[2, 8, 4, 4]" = torch.ops.aten.view.default(bmm, [2, 8, 4, 4]);  bmm = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:660 in _sfdp_pattern_22, code: masked_score = score + attn_mask
        add: "f32[2, 8, 4, 4]" = torch.ops.aten.add.Tensor(view_2, arg3_1);  view_2 = arg3_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:661 in _sfdp_pattern_22, code: score = masked_score.type_as(query)
        convert_element_type_2: "f16[2, 8, 4, 4]" = torch.ops.prims.convert_element_type.default(add, torch.float16);  add = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:662 in _sfdp_pattern_22, code: return score.float().softmax(dim=-1).type_as(query).matmul(value), key, value
        convert_element_type_3: "f32[2, 8, 4, 4]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        amax: "f32[2, 8, 4, 1]" = torch.ops.aten.amax.default(convert_element_type_3, [-1], True)
        sub: "f32[2, 8, 4, 4]" = torch.ops.aten.sub.Tensor(convert_element_type_3, amax);  convert_element_type_3 = amax = None
        exp: "f32[2, 8, 4, 4]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[2, 8, 4, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[2, 8, 4, 4]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_4: "f16[2, 8, 4, 4]" = torch.ops.prims.convert_element_type.default(div, torch.float16);  div = None
        expand_2: "f16[2, 8, 4, 4]" = torch.ops.aten.expand.default(convert_element_type_4, [2, 8, 4, 4]);  convert_element_type_4 = None
        view_3: "f16[16, 4, 4]" = torch.ops.aten.view.default(expand_2, [16, 4, 4]);  expand_2 = None
        expand_3: "f16[2, 8, 4, 16]" = torch.ops.aten.expand.default(permute_2, [2, 8, 4, 16])
        clone_2: "f16[2, 8, 4, 16]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_4: "f16[16, 4, 16]" = torch.ops.aten.view.default(clone_2, [16, 4, 16]);  clone_2 = None
        bmm_1: "f16[16, 4, 16]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f16[2, 8, 4, 16]" = torch.ops.aten.view.default(bmm_1, [2, 8, 4, 16]);  bmm_1 = None
        return (view_5, permute_1, permute_2)
