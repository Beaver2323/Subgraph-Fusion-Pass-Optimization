


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    view = torch.ops.aten.view.default(arg1_1, [8, 16, 32]);  arg1_1 = None
    view_1 = torch.ops.aten.view.default(arg3_1, [8, 16, 32]);  arg3_1 = None
    permute = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    view_2 = torch.ops.aten.view.default(arg2_1, [8, 16, 32]);  arg2_1 = None
    bmm = torch.ops.aten.bmm.default(view_2, permute);  view_2 = permute = None
    view_3 = torch.ops.aten.view.default(bmm, [4, 2, 16, 16]);  bmm = None
    add = torch.ops.aten.add.Tensor(view_3, arg0_1);  view_3 = arg0_1 = None
    view_4 = torch.ops.aten.view.default(add, [8, 16, 16]);  add = None
    amax = torch.ops.aten.amax.default(view_4, [-1], True)
    sub = torch.ops.aten.sub.Tensor(view_4, amax);  view_4 = amax = None
    exp = torch.ops.aten.exp.default(sub);  sub = None
    sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    bmm_1 = torch.ops.aten.bmm.default(div, view_1);  div = view_1 = None
    view_5 = torch.ops.aten.view.default(bmm_1, [4, 2, 16, 32]);  bmm_1 = None
    return (view_5,)
    