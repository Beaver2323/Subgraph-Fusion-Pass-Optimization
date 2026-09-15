class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[2, 4, 8, 16]", arg1_1: "f16[2, 4, 8, 16]", arg2_1: "f16[2, 4, 8, 16]", arg3_1: "f16[2, 4]"):
        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:396 in _sfdp_pattern_15, code: attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
        eq: "b8[2, 4]" = torch.ops.aten.eq.Scalar(arg3_1, 0);  arg3_1 = None
        view_3: "b8[2, 1, 1, 4]" = torch.ops.aten.reshape.default(eq, [2, 1, 1, 4]);  eq = None
        expand_2: "b8[2, 8, 4, 4]" = torch.ops.aten.expand.default(view_3, [2, 8, 4, 4]);  view_3 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:395 in _sfdp_pattern_15, code: fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
        full_default: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:388 in _sfdp_pattern_15, code: q = query.permute([0, 2, 1, 3])
        permute: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:393 in _sfdp_pattern_15, code: scores = q @ k.transpose(-2, -1)
        expand: "f16[2, 8, 4, 16]" = torch.ops.aten.expand.default(permute, [2, 8, 4, 16]);  permute = None
        clone: "f16[2, 8, 4, 16]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f16[16, 4, 16]" = torch.ops.aten.reshape.default(clone, [16, 4, 16]);  clone = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:389 in _sfdp_pattern_15, code: k = key.permute([0, 2, 1, 3])
        permute_1: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:393 in _sfdp_pattern_15, code: scores = q @ k.transpose(-2, -1)
        permute_3: "f16[2, 8, 16, 4]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2]);  permute_1 = None
        expand_1: "f16[2, 8, 16, 4]" = torch.ops.aten.expand.default(permute_3, [2, 8, 16, 4]);  permute_3 = None
        clone_1: "f16[2, 8, 16, 4]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1: "f16[16, 16, 4]" = torch.ops.aten.reshape.default(clone_1, [16, 16, 4]);  clone_1 = None
        bmm: "f16[16, 4, 4]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f16[2, 8, 4, 4]" = torch.ops.aten.reshape.default(bmm, [2, 8, 4, 4]);  bmm = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:394 in _sfdp_pattern_15, code: scores = scores.div(inv_scale)
        div: "f16[2, 8, 4, 4]" = torch.ops.aten.div.Tensor(view_2, 2.0);  view_2 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:397 in _sfdp_pattern_15, code: return torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1) @ v
        where: "f16[2, 8, 4, 4]" = torch.ops.aten.where.self(expand_2, full_default, div);  expand_2 = full_default = div = None
        convert_element_type_2: "f32[2, 8, 4, 4]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        amax: "f32[2, 8, 4, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub: "f32[2, 8, 4, 4]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[2, 8, 4, 4]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[2, 8, 4, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[2, 8, 4, 4]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "f16[2, 8, 4, 4]" = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None
        expand_3: "f16[2, 8, 4, 4]" = torch.ops.aten.expand.default(convert_element_type_3, [2, 8, 4, 4]);  convert_element_type_3 = None
        view_4: "f16[16, 4, 4]" = torch.ops.aten.reshape.default(expand_3, [16, 4, 4]);  expand_3 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:390 in _sfdp_pattern_15, code: v = value.permute([0, 2, 1, 3])
        permute_2: "f16[2, 8, 4, 16]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:397 in _sfdp_pattern_15, code: return torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1) @ v
        expand_4: "f16[2, 8, 4, 16]" = torch.ops.aten.expand.default(permute_2, [2, 8, 4, 16]);  permute_2 = None
        clone_2: "f16[2, 8, 4, 16]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_5: "f16[16, 4, 16]" = torch.ops.aten.reshape.default(clone_2, [16, 4, 16]);  clone_2 = None
        bmm_1: "f16[16, 4, 16]" = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = view_5 = None
        view_6: "f16[2, 8, 4, 16]" = torch.ops.aten.reshape.default(bmm_1, [2, 8, 4, 16]);  bmm_1 = None
        return (view_6,)
