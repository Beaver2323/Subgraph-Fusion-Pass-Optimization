


def forward(self, primals, tangents):
    primals_1, primals_2, primals_3, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    permute = torch.ops.aten.permute.default(primals_2, [0, 1, 3, 2]);  primals_2 = None
    expand = torch.ops.aten.expand.default(primals_1, [4, 2, 16, 32])
    view = torch.ops.aten.view.default(expand, [8, 16, 32]);  expand = None
    expand_1 = torch.ops.aten.expand.default(permute, [4, 2, 32, 16])
    view_1 = torch.ops.aten.view.default(expand_1, [8, 32, 16]);  expand_1 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2 = torch.ops.aten.view.default(bmm, [4, 2, 16, 16]);  bmm = None
    div = torch.ops.aten.div.Tensor(view_2, 3.0);  view_2 = None
    amax = torch.ops.aten.amax.default(div, [-1], True)
    sub = torch.ops.aten.sub.Tensor(div, amax);  div = amax = None
    exp = torch.ops.aten.exp.default(sub);  sub = None
    sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    _npu_dropout = torch.ops.npu._npu_dropout.default(div_1, 0.4)
    getitem = _npu_dropout[0]
    getitem_1 = _npu_dropout[1];  _npu_dropout = None
    expand_2 = torch.ops.aten.expand.default(getitem, [4, 2, 16, 16])
    view_3 = torch.ops.aten.view.default(expand_2, [8, 16, 16]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(primals_3, [4, 2, 16, 32])
    view_4 = torch.ops.aten.view.default(expand_3, [8, 16, 32]);  expand_3 = None
    bmm_1 = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
    view_5 = torch.ops.aten.view.default(bmm_1, [4, 2, 16, 32]);  bmm_1 = None
    matmul_backward = torch.ops.aten.matmul_backward.default(tangents_1, getitem, primals_3, [True, True]);  tangents_1 = getitem = primals_3 = None
    getitem_2 = matmul_backward[0]
    getitem_3 = matmul_backward[1];  matmul_backward = None
    npu_dropout_backward = torch.ops.npu.npu_dropout_backward.default(getitem_2, getitem_1, 0.4);  getitem_2 = getitem_1 = None
    mul = torch.ops.aten.mul.Tensor(npu_dropout_backward, div_1);  npu_dropout_backward = None
    sum_2 = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
    mul_1 = torch.ops.aten.mul.Tensor(div_1, sum_2);  div_1 = sum_2 = None
    sub_1 = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
    div_2 = torch.ops.aten.div.Tensor(sub_1, 3.0);  sub_1 = None
    matmul_backward_1 = torch.ops.aten.matmul_backward.default(div_2, primals_1, permute, [True, True]);  div_2 = primals_1 = permute = None
    getitem_4 = matmul_backward_1[0]
    getitem_5 = matmul_backward_1[1];  matmul_backward_1 = None
    permute_1 = torch.ops.aten.permute.default(getitem_5, [0, 1, 3, 2]);  getitem_5 = None
    return pytree.tree_unflatten([view_5, getitem_4, permute_1, getitem_3], self._out_spec)
    