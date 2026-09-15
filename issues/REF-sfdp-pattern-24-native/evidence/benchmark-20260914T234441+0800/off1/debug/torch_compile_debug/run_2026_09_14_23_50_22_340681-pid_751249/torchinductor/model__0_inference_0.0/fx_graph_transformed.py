class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[2, 4, 8, 16]", arg1_1: "f16[2, 4, 8, 16]", arg2_1: "f16[2, 4, 8, 16]", arg3_1: "f32[1, 1, 8, 8]"):
        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:726 in _sfdp_pattern_24, code: q = query.view(bs * n_head, -1, head_size)
        view: "f16[8, 8, 16]" = torch.ops.aten.reshape.default(arg0_1, [8, -1, 16]);  arg0_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:727 in _sfdp_pattern_24, code: k = key.reshape(bs * n_head, -1, head_size)
        view_1: "f16[8, 8, 16]" = torch.ops.aten.reshape.default(arg1_1, [8, -1, 16]);  arg1_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:729 in _sfdp_pattern_24, code: attn_weights = torch.bmm(q, k.transpose(1, 2))
        permute: "f16[8, 16, 8]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        bmm: "f16[8, 8, 8]" = torch.ops.aten.bmm.default(view, permute);  view = permute = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:730 in _sfdp_pattern_24, code: attn_weights = attn_weights.view(bs, n_head, seq_len, -1) + attention_mask
        view_3: "f16[2, 4, 8, 8]" = torch.ops.aten.reshape.default(bmm, [2, 4, 8, -1]);  bmm = None
        add: "f32[2, 4, 8, 8]" = torch.ops.aten.add.Tensor(view_3, arg3_1);  view_3 = arg3_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:731 in _sfdp_pattern_24, code: attn_weights = attn_weights.view(bs * n_head, seq_len, -1)
        view_4: "f32[8, 8, 8]" = torch.ops.aten.reshape.default(add, [8, 8, -1]);  add = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:732 in _sfdp_pattern_24, code: attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        amax: "f32[8, 8, 1]" = torch.ops.aten.amax.default(view_4, [-1], True)
        sub: "f32[8, 8, 8]" = torch.ops.aten.sub.Tensor(view_4, amax);  view_4 = amax = None
        exp: "f32[8, 8, 8]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[8, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8, 8, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:734 in _sfdp_pattern_24, code: attn_weights = attn_weights.to(torch.half)
        convert_element_type_2: "f16[8, 8, 8]" = torch.ops.prims.convert_element_type.default(div, torch.float16);  div = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:728 in _sfdp_pattern_24, code: v = value.reshape(bs * n_head, -1, head_size)
        view_2: "f16[8, 8, 16]" = torch.ops.aten.reshape.default(arg2_1, [8, -1, 16]);  arg2_1 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:735 in _sfdp_pattern_24, code: attn_output = torch.bmm(attn_weights, v)
        bmm_1: "f16[8, 8, 16]" = torch.ops.aten.bmm.default(convert_element_type_2, view_2);  convert_element_type_2 = view_2 = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:736 in _sfdp_pattern_24, code: attn_output = attn_output.view(bs, n_head, seq_len, head_size)
        view_5: "f16[2, 4, 8, 16]" = torch.ops.aten.reshape.default(bmm_1, [2, 4, 8, 16]);  bmm_1 = None
        return (view_5,)
