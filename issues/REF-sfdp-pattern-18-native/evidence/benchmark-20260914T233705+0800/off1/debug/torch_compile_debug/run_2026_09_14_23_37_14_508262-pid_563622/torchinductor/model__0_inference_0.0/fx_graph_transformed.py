class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[2, 4, 8, 16]", arg1_1: "f16[2, 4, 8, 16]", arg2_1: "f16[2, 4, 8, 16]", arg3_1: "b8[2, 1, 1, 4]"):
        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:507 in _sfdp_pattern_18, code: query = query.permute([0, 2, 1, 3])
        permute: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:510 in _sfdp_pattern_18, code: attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
        expand: "f16[2, 8, 4, 16]" = torch.ops.aten.expand.default(permute, [2, 8, 4, 16]);  permute = None
        clone: "f16[2, 8, 4, 16]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f16[16, 4, 16]" = torch.ops.aten.reshape.default(clone, [16, 4, 16]);  clone = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:508 in _sfdp_pattern_18, code: key = key.permute([0, 2, 1, 3])
        permute_1: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:510 in _sfdp_pattern_18, code: attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
        permute_3: "f16[2, 8, 16, 4]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2])
        expand_1: "f16[2, 8, 16, 4]" = torch.ops.aten.expand.default(permute_3, [2, 8, 16, 4]);  permute_3 = None
        clone_1: "f16[2, 8, 16, 4]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1: "f16[16, 16, 4]" = torch.ops.aten.reshape.default(clone_1, [16, 16, 4]);  clone_1 = None
        bmm: "f16[16, 4, 4]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f16[2, 8, 4, 4]" = torch.ops.aten.reshape.default(bmm, [2, 8, 4, 4]);  bmm = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:511 in _sfdp_pattern_18, code: inv_scale = torch.full(
        full_default: "f16[]" = torch.ops.aten.full.default([], 0.66666, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:517 in _sfdp_pattern_18, code: attn_weights = attn_weights.div(inv_scale)
        div: "f16[2, 8, 4, 4]" = torch.ops.aten.div.Tensor(view_2, full_default);  view_2 = full_default = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:518 in _sfdp_pattern_18, code: causal_mask_value = torch.full(
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:521 in _sfdp_pattern_18, code: attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
        where: "f16[2, 8, 4, 4]" = torch.ops.aten.where.self(arg3_1, div, full_default_1);  arg3_1 = div = full_default_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:524 in _sfdp_pattern_18, code: torch.nn.functional.dropout(attn_weights.softmax(dim=-1), dropout_p).matmul(
        convert_element_type_2: "f32[2, 8, 4, 4]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        amax: "f32[2, 8, 4, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub: "f32[2, 8, 4, 4]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[2, 8, 4, 4]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[2, 8, 4, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[2, 8, 4, 4]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "f16[2, 8, 4, 4]" = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None
        expand_2: "f16[2, 8, 4, 4]" = torch.ops.aten.expand.default(convert_element_type_3, [2, 8, 4, 4]);  convert_element_type_3 = None
        view_3: "f16[16, 4, 4]" = torch.ops.aten.reshape.default(expand_2, [16, 4, 4]);  expand_2 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:509 in _sfdp_pattern_18, code: value = value.permute([0, 2, 1, 3])
        permute_2: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:524 in _sfdp_pattern_18, code: torch.nn.functional.dropout(attn_weights.softmax(dim=-1), dropout_p).matmul(
        expand_3: "f16[2, 8, 4, 16]" = torch.ops.aten.expand.default(permute_2, [2, 8, 4, 16])
        clone_2: "f16[2, 8, 4, 16]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_4: "f16[16, 4, 16]" = torch.ops.aten.reshape.default(clone_2, [16, 4, 16]);  clone_2 = None
        bmm_1: "f16[16, 4, 16]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f16[2, 8, 4, 16]" = torch.ops.aten.reshape.default(bmm_1, [2, 8, 4, 16]);  bmm_1 = None
        return (view_5, permute_1, permute_2)
