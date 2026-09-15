class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f16[2, 3, 4, 16, 8]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1747 in dot_prod_attention, code: q, k, v = qkv.permute(1, 0, 2, 4, 3).unbind(0)
        permute: "f16[3, 2, 4, 8, 16]" = torch.ops.aten.permute.default(primals_1, [1, 0, 2, 4, 3]);  primals_1 = None
        unbind = torch.ops.aten.unbind.int(permute);  permute = None
        getitem: "f16[2, 4, 8, 16]" = unbind[0]
        getitem_1: "f16[2, 4, 8, 16]" = unbind[1]
        getitem_2: "f16[2, 4, 8, 16]" = unbind[2];  unbind = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1748 in dot_prod_attention, code: scores = torch.matmul(q, k.transpose(-2, -1))
        permute_1: "f16[2, 4, 16, 8]" = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(getitem, [2, 4, 8, 16])
        clone: "f16[2, 4, 8, 16]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f16[8, 8, 16]" = torch.ops.aten.reshape.default(clone, [8, 8, 16]);  clone = None
        expand_1: "f16[2, 4, 16, 8]" = torch.ops.aten.expand.default(permute_1, [2, 4, 16, 8])
        clone_1: "f16[2, 4, 16, 8]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1: "f16[8, 16, 8]" = torch.ops.aten.reshape.default(clone_1, [8, 16, 8]);  clone_1 = None
        bmm: "f16[8, 8, 8]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f16[2, 4, 8, 8]" = torch.ops.aten.reshape.default(bmm, [2, 4, 8, 8]);  bmm = None

        # No stacktrace found for following nodes
        mul_tensor: "f16[2, 4, 8, 8]" = torch.ops.aten.mul.Tensor(view_2, 0.2)
        convert_element_type_default: "f32[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        convert_element_type_default_1: "f32[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        mul_tensor_1: "f32[2, 4, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        amax_default: "f32[2, 4, 8, 1]" = torch.ops.aten.amax.default(mul_tensor_1, [-1], True)
        sub_tensor: "f32[2, 4, 8, 8]" = torch.ops.aten.sub.Tensor(mul_tensor_1, amax_default);  mul_tensor_1 = amax_default = None
        mul_tensor_2: "f32[2, 4, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, 0.2);  sub_tensor = None
        amax_default_1: "f32[2, 4, 8, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor_1: "f32[2, 4, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default_1);  amax_default_1 = None
        eq_tensor: "b8[2, 4, 8, 8]" = torch.ops.aten.eq.Tensor(convert_element_type_default, convert_element_type_default)
        abs_default: "f32[2, 4, 8, 8]" = torch.ops.aten.abs.default(convert_element_type_default);  convert_element_type_default = None
        ne_scalar: "b8[2, 4, 8, 8]" = torch.ops.aten.ne.Scalar(abs_default, inf);  abs_default = None
        mul_tensor_3: "b8[2, 4, 8, 8]" = torch.ops.aten.mul.Tensor(eq_tensor, ne_scalar);  eq_tensor = ne_scalar = None
        logical_not_default: "b8[2, 4, 8, 8]" = torch.ops.aten.logical_not.default(mul_tensor_3);  mul_tensor_3 = None
        any_dims: "b8[2, 4, 8, 1]" = torch.ops.aten.any.dims(logical_not_default, [-1], True);  logical_not_default = None
        logical_not_default_1: "b8[2, 4, 8, 1]" = torch.ops.aten.logical_not.default(any_dims);  any_dims = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1750 in dot_prod_attention, code: attn_weights = scores.softmax(dim=-1)
        where_self: "f32[2, 4, 8, 8]" = torch.ops.aten.where.self(logical_not_default_1, mul_tensor_2, sub_tensor_1);  logical_not_default_1 = mul_tensor_2 = sub_tensor_1 = None
        exp: "f32[2, 4, 8, 8]" = torch.ops.aten.exp.default(where_self);  where_self = None
        sum_1: "f32[2, 4, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[2, 4, 8, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "f16[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(div, torch.float16);  div = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1751 in dot_prod_attention, code: attn_weights = torch.nn.functional.dropout(
        _npu_dropout = torch.ops.npu._npu_dropout.default(convert_element_type_3, 0.1)
        getitem_3: "f16[2, 4, 8, 8]" = _npu_dropout[0]
        getitem_4: "u8[64]" = _npu_dropout[1];  _npu_dropout = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1754 in dot_prod_attention, code: return attn_weights.matmul(v)
        expand_2: "f16[2, 4, 8, 8]" = torch.ops.aten.expand.default(getitem_3, [2, 4, 8, 8])
        view_3: "f16[8, 8, 8]" = torch.ops.aten.reshape.default(expand_2, [8, 8, 8]);  expand_2 = None
        expand_3: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(getitem_2, [2, 4, 8, 16])
        clone_2: "f16[2, 4, 8, 16]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_4: "f16[8, 8, 16]" = torch.ops.aten.reshape.default(clone_2, [8, 8, 16]);  clone_2 = None
        bmm_1: "f16[8, 8, 16]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f16[2, 4, 8, 16]" = torch.ops.aten.reshape.default(bmm_1, [2, 4, 8, 16]);  bmm_1 = None
        return (view_5, getitem, getitem_2, permute_1, convert_element_type_3, getitem_3, getitem_4)
