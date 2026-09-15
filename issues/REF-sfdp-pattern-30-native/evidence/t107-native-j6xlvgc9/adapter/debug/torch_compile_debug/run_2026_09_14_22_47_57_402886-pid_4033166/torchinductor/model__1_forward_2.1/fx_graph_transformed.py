class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f16[2, 8, 4, 16]", primals_2: "f16[2, 8, 4, 16]", primals_3: "f16[2, 8, 4, 16]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1710 in dot_prod_attention, code: k = key.permute(0, 2, 1, 3)
        permute: "f16[2, 4, 8, 16]" = torch.ops.aten.permute.default(primals_1, [0, 2, 1, 3]);  primals_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1709 in dot_prod_attention, code: q = query.permute(0, 2, 1, 3)
        permute_1: "f16[2, 4, 8, 16]" = torch.ops.aten.permute.default(primals_2, [0, 2, 1, 3]);  primals_2 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1713 in dot_prod_attention, code: q = torch.ops.aten.mul.Scalar(q, scale)
        mul: "f16[2, 4, 8, 16]" = torch.ops.aten.mul.Scalar(permute_1, 0.25);  permute_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1711 in dot_prod_attention, code: v = value.permute(0, 2, 1, 3)
        permute_2: "f16[2, 4, 8, 16]" = torch.ops.aten.permute.default(primals_3, [0, 2, 1, 3]);  primals_3 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1714 in dot_prod_attention, code: k_t = torch.ops.aten.mul.Scalar(k.transpose(-2, -1), scale)
        permute_3: "f16[2, 4, 16, 8]" = torch.ops.aten.permute.default(permute, [0, 1, 3, 2]);  permute = None
        mul_1: "f16[2, 4, 16, 8]" = torch.ops.aten.mul.Scalar(permute_3, 0.25);  permute_3 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1715 in dot_prod_attention, code: attn_weight = q @ k_t
        expand: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(mul, [2, 4, 8, 16])
        clone: "f16[2, 4, 8, 16]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f16[8, 8, 16]" = torch.ops.aten.reshape.default(clone, [8, 8, 16]);  clone = None
        expand_1: "f16[2, 4, 16, 8]" = torch.ops.aten.expand.default(mul_1, [2, 4, 16, 8])
        clone_1: "f16[2, 4, 16, 8]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1: "f16[8, 16, 8]" = torch.ops.aten.reshape.default(clone_1, [8, 16, 8]);  clone_1 = None
        bmm: "f16[8, 8, 8]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f16[2, 4, 8, 8]" = torch.ops.aten.reshape.default(bmm, [2, 4, 8, 8]);  bmm = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1716 in dot_prod_attention, code: attn_weight = torch.ops.aten._safe_softmax(attn_weight, -1)
        convert_element_type_2: "f32[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32)
        amax: "f32[2, 4, 8, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub: "f32[2, 4, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[2, 4, 8, 8]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[2, 4, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[2, 4, 8, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "f16[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(div, torch.float16);  div = None
        eq: "b8[2, 4, 8, 8]" = torch.ops.aten.eq.Scalar(view_2, -inf);  view_2 = None
        logical_not: "b8[2, 4, 8, 8]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[2, 4, 8, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[2, 4, 8, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default: "f16[2, 4, 8, 8]" = torch.ops.aten.full.default([2, 4, 8, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "f16[2, 4, 8, 8]" = torch.ops.aten.where.self(logical_not_1, full_default, convert_element_type_3);  logical_not_1 = full_default = convert_element_type_3 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1717 in dot_prod_attention, code: return attn_weight @ v
        expand_2: "f16[2, 4, 8, 8]" = torch.ops.aten.expand.default(where, [2, 4, 8, 8])
        view_3: "f16[8, 8, 8]" = torch.ops.aten.reshape.default(expand_2, [8, 8, 8]);  expand_2 = None
        expand_3: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(permute_2, [2, 4, 8, 16])
        clone_2: "f16[2, 4, 8, 16]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_4: "f16[8, 8, 16]" = torch.ops.aten.reshape.default(clone_2, [8, 8, 16]);  clone_2 = None
        bmm_1: "f16[8, 8, 16]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f16[2, 4, 8, 16]" = torch.ops.aten.reshape.default(bmm_1, [2, 4, 8, 16]);  bmm_1 = None
        return (view_5, mul, permute_2, mul_1, where)
