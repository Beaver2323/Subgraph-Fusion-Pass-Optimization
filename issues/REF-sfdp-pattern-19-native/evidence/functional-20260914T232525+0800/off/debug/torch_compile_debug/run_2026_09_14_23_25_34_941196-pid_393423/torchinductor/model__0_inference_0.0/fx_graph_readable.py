class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[2, 4, 8, 16]", arg1_1: "f16[2, 4, 8, 16]", arg2_1: "b8[1, 1, 8, 8]", arg3_1: "f32[1, 1, 8, 8]", arg4_1: "f16[2, 4, 8, 16]"):
        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:554 in _sfdp_pattern_19, code: attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
        permute: "f16[2, 4, 16, 8]" = torch.ops.aten.permute.default(arg0_1, [0, 1, 3, 2]);  arg0_1 = None
        expand: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(arg1_1, [2, 4, 8, 16]);  arg1_1 = None
        view: "f16[8, 8, 16]" = torch.ops.aten.view.default(expand, [8, 8, 16]);  expand = None
        expand_1: "f16[2, 4, 16, 8]" = torch.ops.aten.expand.default(permute, [2, 4, 16, 8]);  permute = None
        view_1: "f16[8, 16, 8]" = torch.ops.aten.view.default(expand_1, [8, 16, 8]);  expand_1 = None
        bmm: "f16[8, 8, 8]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f16[2, 4, 8, 8]" = torch.ops.aten.view.default(bmm, [2, 4, 8, 8]);  bmm = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:555 in _sfdp_pattern_19, code: inv_scale = torch.full(
        full_default: "f16[]" = torch.ops.aten.full.default([], 0.66666, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:561 in _sfdp_pattern_19, code: attn_weights = attn_weights.div(inv_scale)
        div: "f16[2, 4, 8, 8]" = torch.ops.aten.div.Tensor(view_2, full_default);  view_2 = full_default = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:562 in _sfdp_pattern_19, code: causal_mask_value = torch.full(
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:565 in _sfdp_pattern_19, code: attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
        where: "f16[2, 4, 8, 8]" = torch.ops.aten.where.self(arg2_1, div, full_default_1);  arg2_1 = div = full_default_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:566 in _sfdp_pattern_19, code: attn_weights = attn_weights + attn_mask
        add: "f32[2, 4, 8, 8]" = torch.ops.aten.add.Tensor(where, arg3_1);  where = arg3_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:567 in _sfdp_pattern_19, code: attn_weights = attn_weights.softmax(dim=-1).type(value.dtype)
        amax: "f32[2, 4, 8, 1]" = torch.ops.aten.amax.default(add, [-1], True)
        sub: "f32[2, 4, 8, 8]" = torch.ops.aten.sub.Tensor(add, amax);  add = amax = None
        exp: "f32[2, 4, 8, 8]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[2, 4, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[2, 4, 8, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_2: "f16[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:568 in _sfdp_pattern_19, code: return torch.nn.functional.dropout(attn_weights, dropout_p).matmul(value)
        expand_2: "f16[2, 4, 8, 8]" = torch.ops.aten.expand.default(convert_element_type_2, [2, 4, 8, 8]);  convert_element_type_2 = None
        view_3: "f16[8, 8, 8]" = torch.ops.aten.view.default(expand_2, [8, 8, 8]);  expand_2 = None
        expand_3: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(arg4_1, [2, 4, 8, 16]);  arg4_1 = None
        view_4: "f16[8, 8, 16]" = torch.ops.aten.view.default(expand_3, [8, 8, 16]);  expand_3 = None
        bmm_1: "f16[8, 8, 16]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f16[2, 4, 8, 16]" = torch.ops.aten.view.default(bmm_1, [2, 4, 8, 16]);  bmm_1 = None
        return (view_5,)
