class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f16[2, 8, 4, 16]", primals_2: "f16[2, 8, 4, 16]", primals_3: "f16[2, 8, 4, 16]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:665 in sfdp_pattern_10, code: k = key.permute(0, 2, 1, 3)
        permute: "f16[2, 4, 8, 16]" = torch.ops.aten.permute.default(primals_1, [0, 2, 1, 3]);  primals_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:664 in sfdp_pattern_10, code: q = query.permute(0, 2, 1, 3)
        permute_1: "f16[2, 4, 8, 16]" = torch.ops.aten.permute.default(primals_2, [0, 2, 1, 3]);  primals_2 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:666 in sfdp_pattern_10, code: v = value.permute(0, 2, 1, 3)
        permute_2: "f16[2, 4, 8, 16]" = torch.ops.aten.permute.default(primals_3, [0, 2, 1, 3]);  primals_3 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:668 in sfdp_pattern_10, code: div = q @ k.transpose(-2, -1)
        permute_3: "f16[2, 4, 16, 8]" = torch.ops.aten.permute.default(permute, [0, 1, 3, 2]);  permute = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:667 in sfdp_pattern_10, code: q = q / math.sqrt(q.size(-1))
        div: "f16[2, 4, 8, 16]" = torch.ops.aten.div.Tensor(permute_1, 4.0);  permute_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:668 in sfdp_pattern_10, code: div = q @ k.transpose(-2, -1)
        expand: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(div, [2, 4, 8, 16])
        clone: "f16[2, 4, 8, 16]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f16[8, 8, 16]" = torch.ops.aten.view.default(clone, [8, 8, 16]);  clone = None
        expand_1: "f16[2, 4, 16, 8]" = torch.ops.aten.expand.default(permute_3, [2, 4, 16, 8])
        clone_1: "f16[2, 4, 16, 8]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1: "f16[8, 16, 8]" = torch.ops.aten.view.default(clone_1, [8, 16, 8]);  clone_1 = None
        bmm: "f16[8, 8, 8]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f16[2, 4, 8, 8]" = torch.ops.aten.view.default(bmm, [2, 4, 8, 8])

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:669 in sfdp_pattern_10, code: div = div.to(torch.float32)
        convert_element_type_2: "f32[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:670 in sfdp_pattern_10, code: attn_weight = torch.softmax(div, dim=-1)
        amax: "f32[2, 4, 8, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub: "f32[2, 4, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = None
        exp: "f32[2, 4, 8, 8]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[2, 4, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[2, 4, 8, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:671 in sfdp_pattern_10, code: attn_weight = attn_weight.to(torch.float16)
        convert_element_type_3: "f16[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:672 in sfdp_pattern_10, code: return attn_weight @ v
        expand_2: "f16[2, 4, 8, 8]" = torch.ops.aten.expand.default(convert_element_type_3, [2, 4, 8, 8])
        view_3: "f16[8, 8, 8]" = torch.ops.aten.view.default(expand_2, [8, 8, 8]);  expand_2 = None
        expand_3: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(permute_2, [2, 4, 8, 16])
        clone_2: "f16[2, 4, 8, 16]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_4: "f16[8, 8, 16]" = torch.ops.aten.view.default(clone_2, [8, 8, 16]);  clone_2 = None
        bmm_1: "f16[8, 8, 16]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f16[2, 4, 8, 16]" = torch.ops.aten.view.default(bmm_1, [2, 4, 8, 16]);  bmm_1 = None
        return (view_5, permute_2, permute_3, div, bmm, amax, sum_1, convert_element_type_3)
