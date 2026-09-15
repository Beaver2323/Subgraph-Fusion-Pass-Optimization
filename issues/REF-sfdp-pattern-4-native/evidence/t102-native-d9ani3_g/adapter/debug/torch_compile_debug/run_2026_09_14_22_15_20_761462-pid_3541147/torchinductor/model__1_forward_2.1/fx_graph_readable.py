class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 2, 16, 32]", primals_2: "f32[4, 2, 16, 32]", primals_3: "f32[4, 2, 16, 32]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:349 in dot_prod_attention, code: torch.matmul(query, key.transpose(-2, -1)).mul(0.4).softmax(dim=-1),
        permute: "f32[4, 2, 32, 16]" = torch.ops.aten.permute.default(primals_1, [0, 1, 3, 2]);  primals_1 = None
        expand: "f32[4, 2, 16, 32]" = torch.ops.aten.expand.default(primals_2, [4, 2, 16, 32])
        view: "f32[8, 16, 32]" = torch.ops.aten.view.default(expand, [8, 16, 32]);  expand = None
        expand_1: "f32[4, 2, 32, 16]" = torch.ops.aten.expand.default(permute, [4, 2, 32, 16])
        view_1: "f32[8, 32, 16]" = torch.ops.aten.view.default(expand_1, [8, 32, 16]);  expand_1 = None
        bmm: "f32[8, 16, 16]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f32[4, 2, 16, 16]" = torch.ops.aten.view.default(bmm, [4, 2, 16, 16]);  bmm = None

        # No stacktrace found for following nodes
        mul_tensor: "f32[4, 2, 16, 16]" = torch.ops.aten.mul.Tensor(view_2, 0.4)
        mul_tensor_1: "f32[4, 2, 16, 16]" = torch.ops.aten.mul.Tensor(view_2, 1);  view_2 = None
        amax_default: "f32[4, 2, 16, 1]" = torch.ops.aten.amax.default(mul_tensor_1, [-1], True)
        sub_tensor: "f32[4, 2, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor_1, amax_default);  mul_tensor_1 = amax_default = None
        mul_tensor_2: "f32[4, 2, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor, 0.4);  sub_tensor = None
        amax_default_1: "f32[4, 2, 16, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor_1: "f32[4, 2, 16, 16]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default_1);  amax_default_1 = None
        eq_tensor: "b8[4, 2, 16, 16]" = torch.ops.aten.eq.Tensor(mul_tensor, mul_tensor)
        abs_default: "f32[4, 2, 16, 16]" = torch.ops.aten.abs.default(mul_tensor);  mul_tensor = None
        ne_scalar: "b8[4, 2, 16, 16]" = torch.ops.aten.ne.Scalar(abs_default, inf);  abs_default = None
        mul_tensor_3: "b8[4, 2, 16, 16]" = torch.ops.aten.mul.Tensor(eq_tensor, ne_scalar);  eq_tensor = ne_scalar = None
        logical_not_default: "b8[4, 2, 16, 16]" = torch.ops.aten.logical_not.default(mul_tensor_3);  mul_tensor_3 = None
        any_dims: "b8[4, 2, 16, 1]" = torch.ops.aten.any.dims(logical_not_default, [-1], True);  logical_not_default = None
        logical_not_default_1: "b8[4, 2, 16, 1]" = torch.ops.aten.logical_not.default(any_dims);  any_dims = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:349 in dot_prod_attention, code: torch.matmul(query, key.transpose(-2, -1)).mul(0.4).softmax(dim=-1),
        where_self: "f32[4, 2, 16, 16]" = torch.ops.aten.where.self(logical_not_default_1, mul_tensor_2, sub_tensor_1);  logical_not_default_1 = mul_tensor_2 = sub_tensor_1 = None
        exp: "f32[4, 2, 16, 16]" = torch.ops.aten.exp.default(where_self);  where_self = None
        sum_1: "f32[4, 2, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[4, 2, 16, 16]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:348 in dot_prod_attention, code: return torch.nn.functional.dropout(
        _npu_dropout = torch.ops.npu._npu_dropout.default(div, 0.2)
        getitem: "f32[4, 2, 16, 16]" = _npu_dropout[0]
        getitem_1: "u8[256]" = _npu_dropout[1];  _npu_dropout = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:353 in dot_prod_attention, code: ).matmul(value)
        expand_2: "f32[4, 2, 16, 16]" = torch.ops.aten.expand.default(getitem, [4, 2, 16, 16])
        view_3: "f32[8, 16, 16]" = torch.ops.aten.view.default(expand_2, [8, 16, 16]);  expand_2 = None
        expand_3: "f32[4, 2, 16, 32]" = torch.ops.aten.expand.default(primals_3, [4, 2, 16, 32])
        view_4: "f32[8, 16, 32]" = torch.ops.aten.view.default(expand_3, [8, 16, 32]);  expand_3 = None
        bmm_1: "f32[8, 16, 32]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f32[4, 2, 16, 32]" = torch.ops.aten.view.default(bmm_1, [4, 2, 16, 32]);  bmm_1 = None
        return (view_5, primals_2, primals_3, permute, div, getitem, getitem_1)
