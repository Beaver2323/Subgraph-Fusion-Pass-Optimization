


def forward(self, primals, tangents):
    primals_1, primals_2, primals_3, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    permute = torch.ops.aten.permute.default(primals_1, [0, 1, 3, 2]);  primals_1 = None
    expand = torch.ops.aten.expand.default(primals_2, [2, 4, 8, 16])
    view = torch.ops.aten.view.default(expand, [8, 8, 16]);  expand = None
    expand_1 = torch.ops.aten.expand.default(permute, [2, 4, 16, 8])
    view_1 = torch.ops.aten.view.default(expand_1, [8, 16, 8]);  expand_1 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2 = torch.ops.aten.view.default(bmm, [2, 4, 8, 8]);  bmm = None
    div = torch.ops.aten.div.Tensor(view_2, 2.0);  view_2 = None
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(div, torch.float32);  div = None
    amax = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
    sub = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
    exp = torch.ops.aten.exp.default(sub);  sub = None
    sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    convert_element_type_3 = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None
    _npu_dropout = torch.ops.npu._npu_dropout.default(convert_element_type_3, 1e-11)
    getitem = _npu_dropout[0]
    getitem_1 = _npu_dropout[1];  _npu_dropout = None
    expand_2 = torch.ops.aten.expand.default(getitem, [2, 4, 8, 8])
    view_3 = torch.ops.aten.view.default(expand_2, [8, 8, 8]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(primals_3, [2, 4, 8, 16])
    view_4 = torch.ops.aten.view.default(expand_3, [8, 8, 16]);  expand_3 = None
    bmm_1 = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
    view_5 = torch.ops.aten.view.default(bmm_1, [2, 4, 8, 16]);  bmm_1 = None
    matmul_backward = torch.ops.aten.matmul_backward.default(tangents_1, getitem, primals_3, [True, True]);  tangents_1 = getitem = primals_3 = None
    getitem_2 = matmul_backward[0]
    getitem_3 = matmul_backward[1];  matmul_backward = None
    npu_dropout_backward = torch.ops.npu.npu_dropout_backward.default(getitem_2, getitem_1, 1.000000082740371e-11);  getitem_2 = getitem_1 = None
    convert_element_type_6 = torch.ops.prims.convert_element_type.default(npu_dropout_backward, torch.float32);  npu_dropout_backward = None
    convert_element_type_7 = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
    mul = torch.ops.aten.mul.Tensor(convert_element_type_6, convert_element_type_7);  convert_element_type_6 = None
    sum_2 = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
    mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_7, sum_2);  convert_element_type_7 = sum_2 = None
    sub_1 = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
    convert_element_type_8 = torch.ops.prims.convert_element_type.default(sub_1, torch.float16);  sub_1 = None
    div_2 = torch.ops.aten.div.Tensor(convert_element_type_8, 2.0);  convert_element_type_8 = None
    matmul_backward_1 = torch.ops.aten.matmul_backward.default(div_2, primals_2, permute, [True, True]);  div_2 = primals_2 = permute = None
    getitem_4 = matmul_backward_1[0]
    getitem_5 = matmul_backward_1[1];  matmul_backward_1 = None
    permute_1 = torch.ops.aten.permute.default(getitem_5, [0, 1, 3, 2]);  getitem_5 = None
    return pytree.tree_unflatten([view_5, permute_1, getitem_4, getitem_3], self._out_spec)
    