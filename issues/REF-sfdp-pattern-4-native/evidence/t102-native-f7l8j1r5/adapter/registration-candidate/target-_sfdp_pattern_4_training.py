


def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, tangents_1):
    primals_2 = primals_1
    primals_4 = primals_2
    primals_6 = primals_3
    primals_8 = primals_4
    primals_10 = primals_5
    tangents_2 = tangents_1
    permute = torch.ops.aten.permute.default(primals_4, [0, 1, 3, 2]);  primals_4 = None
    expand = torch.ops.aten.expand.default(primals_2, [2, 4, 8, 16])
    view = torch.ops.aten.view.default(expand, [8, 8, 16]);  expand = None
    expand_1 = torch.ops.aten.expand.default(permute, [2, 4, 16, 8])
    view_1 = torch.ops.aten.view.default(expand_1, [8, 16, 8]);  expand_1 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2 = torch.ops.aten.view.default(bmm, [2, 4, 8, 8]);  bmm = None
    mul = torch.ops.aten.mul.Tensor(view_2, primals_8);  view_2 = None
    amax = torch.ops.aten.amax.default(mul, [-1], True)
    sub = torch.ops.aten.sub.Tensor(mul, amax);  mul = amax = None
    exp = torch.ops.aten.exp.default(sub);  sub = None
    sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    _npu_dropout = torch.ops.npu._npu_dropout.default(div, 0.113377)
    getitem_1 = _npu_dropout[0]
    getitem_3 = _npu_dropout[1];  _npu_dropout = None
    expand_2 = torch.ops.aten.expand.default(getitem_1, [2, 4, 8, 8])
    view_3 = torch.ops.aten.view.default(expand_2, [8, 8, 8]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(primals_6, [2, 4, 8, 16])
    view_4 = torch.ops.aten.view.default(expand_3, [8, 8, 16]);  expand_3 = None
    bmm_1 = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
    view_5 = torch.ops.aten.view.default(bmm_1, [2, 4, 8, 16]);  bmm_1 = None
    matmul_backward = torch.ops.aten.matmul_backward.default(tangents_2, getitem_1, primals_6, [True, True]);  tangents_2 = getitem_1 = primals_6 = None
    getitem_5 = matmul_backward[0]
    getitem_7 = matmul_backward[1];  matmul_backward = None
    npu_dropout_backward = torch.ops.npu.npu_dropout_backward.default(getitem_5, getitem_3, 0.11337700000000006);  getitem_5 = getitem_3 = None
    mul_1 = torch.ops.aten.mul.Tensor(npu_dropout_backward, div);  npu_dropout_backward = None
    sum_2 = torch.ops.aten.sum.dim_IntList(mul_1, [-1], True)
    mul_2 = torch.ops.aten.mul.Tensor(div, sum_2);  div = sum_2 = None
    sub_1 = torch.ops.aten.sub.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    mul_3 = torch.ops.aten.mul.Tensor(sub_1, primals_8);  sub_1 = primals_8 = None
    matmul_backward_1 = torch.ops.aten.matmul_backward.default(mul_3, primals_2, permute, [True, True]);  mul_3 = primals_2 = permute = None
    getitem_9 = matmul_backward_1[0]
    getitem_11 = matmul_backward_1[1];  matmul_backward_1 = None
    permute_1 = torch.ops.aten.permute.default(getitem_11, [0, 1, 3, 2]);  getitem_11 = None
    return [view_5, getitem_9, permute_1, getitem_7, None, None]
    