class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f16[2, 8, 4, 16]", primals_2: "f16[2, 8, 4, 16]", primals_3: "f16[2, 8, 4, 16]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:533 in sfdp_pattern_8, code: k = key.permute(0, 2, 1, 3)
        permute: "f16[2, 4, 8, 16]" = torch.ops.aten.permute.default(primals_1, [0, 2, 1, 3]);  primals_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:532 in sfdp_pattern_8, code: q = query.permute(0, 2, 1, 3)
        permute_1: "f16[2, 4, 8, 16]" = torch.ops.aten.permute.default(primals_2, [0, 2, 1, 3]);  primals_2 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:534 in sfdp_pattern_8, code: v = value.permute(0, 2, 1, 3)
        permute_2: "f16[2, 4, 8, 16]" = torch.ops.aten.permute.default(primals_3, [0, 2, 1, 3]);  primals_3 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:535 in sfdp_pattern_8, code: div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        permute_3: "f16[2, 4, 16, 8]" = torch.ops.aten.permute.default(permute, [0, 1, 3, 2]);  permute = None
        expand: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(permute_1, [2, 4, 8, 16])
        clone: "f16[2, 4, 8, 16]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f16[8, 8, 16]" = torch.ops.aten.view.default(clone, [8, 8, 16]);  clone = None
        expand_1: "f16[2, 4, 16, 8]" = torch.ops.aten.expand.default(permute_3, [2, 4, 16, 8])
        clone_1: "f16[2, 4, 16, 8]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1: "f16[8, 16, 8]" = torch.ops.aten.view.default(clone_1, [8, 16, 8]);  clone_1 = None
        bmm: "f16[8, 8, 8]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f16[2, 4, 8, 8]" = torch.ops.aten.view.default(bmm, [2, 4, 8, 8]);  bmm = None

        # No stacktrace found for following nodes
        div_tensor: "f16[2, 4, 8, 8]" = torch.ops.aten.div.Tensor(view_2, 4.0)
        convert_element_type_default: "f32[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float32);  div_tensor = None
        convert_element_type_default_1: "f32[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        mul_tensor: "f32[2, 4, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        amax_default: "f32[2, 4, 8, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[2, 4, 8, 8]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        div_tensor_1: "f32[2, 4, 8, 8]" = torch.ops.aten.div.Tensor(sub_tensor, 4.0);  sub_tensor = None
        amax_default_1: "f32[2, 4, 8, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor_1: "f32[2, 4, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default_1);  amax_default_1 = None
        eq_tensor: "b8[2, 4, 8, 8]" = torch.ops.aten.eq.Tensor(convert_element_type_default, convert_element_type_default)
        abs_default: "f32[2, 4, 8, 8]" = torch.ops.aten.abs.default(convert_element_type_default);  convert_element_type_default = None
        ne_scalar: "b8[2, 4, 8, 8]" = torch.ops.aten.ne.Scalar(abs_default, inf);  abs_default = None
        mul_tensor_1: "b8[2, 4, 8, 8]" = torch.ops.aten.mul.Tensor(eq_tensor, ne_scalar);  eq_tensor = ne_scalar = None
        logical_not_default: "b8[2, 4, 8, 8]" = torch.ops.aten.logical_not.default(mul_tensor_1);  mul_tensor_1 = None
        any_dims: "b8[2, 4, 8, 1]" = torch.ops.aten.any.dims(logical_not_default, [-1], True);  logical_not_default = None
        logical_not_default_1: "b8[2, 4, 8, 1]" = torch.ops.aten.logical_not.default(any_dims);  any_dims = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:537 in sfdp_pattern_8, code: attn_weight = torch.softmax(div, dim=-1)
        where_self: "f32[2, 4, 8, 8]" = torch.ops.aten.where.self(logical_not_default_1, div_tensor_1, sub_tensor_1);  logical_not_default_1 = div_tensor_1 = sub_tensor_1 = None
        exp: "f32[2, 4, 8, 8]" = torch.ops.aten.exp.default(where_self);  where_self = None
        sum_1: "f32[2, 4, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[2, 4, 8, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:538 in sfdp_pattern_8, code: attn_weight = attn_weight.to(torch.float16)
        convert_element_type_3: "f16[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(div_1, torch.float16)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:539 in sfdp_pattern_8, code: return attn_weight @ v
        expand_2: "f16[2, 4, 8, 8]" = torch.ops.aten.expand.default(convert_element_type_3, [2, 4, 8, 8])
        view_3: "f16[8, 8, 8]" = torch.ops.aten.view.default(expand_2, [8, 8, 8]);  expand_2 = None
        expand_3: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(permute_2, [2, 4, 8, 16])
        clone_2: "f16[2, 4, 8, 16]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_4: "f16[8, 8, 16]" = torch.ops.aten.view.default(clone_2, [8, 8, 16]);  clone_2 = None
        bmm_1: "f16[8, 8, 16]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f16[2, 4, 8, 16]" = torch.ops.aten.view.default(bmm_1, [2, 4, 8, 16]);  bmm_1 = None
        return (view_5, permute_1, permute_2, permute_3, div_1, convert_element_type_3)
