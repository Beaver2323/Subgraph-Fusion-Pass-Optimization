


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    permute = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
    permute_1 = torch.ops.aten.permute.default(permute, [0, 1, 3, 2])
    permute_2 = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
    expand = torch.ops.aten.expand.default(permute_2, [1, 16, 2, 32]);  permute_2 = None
    view = torch.ops.aten.view.default(expand, [16, 2, 32]);  expand = None
    expand_1 = torch.ops.aten.expand.default(permute_1, [1, 16, 32, 2]);  permute_1 = None
    view_1 = torch.ops.aten.view.default(expand_1, [16, 32, 2]);  expand_1 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2 = torch.ops.aten.view.default(bmm, [1, 16, 2, 2]);  bmm = None
    permute_3 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1, 3]);  arg3_1 = None
    add = torch.ops.aten.add.Tensor(view_2, arg0_1);  view_2 = arg0_1 = None
    amax = torch.ops.aten.amax.default(add, [-1], True)
    sub = torch.ops.aten.sub.Tensor(add, amax);  add = amax = None
    exp = torch.ops.aten.exp.default(sub);  sub = None
    sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    expand_2 = torch.ops.aten.expand.default(div, [1, 16, 2, 2]);  div = None
    view_3 = torch.ops.aten.view.default(expand_2, [16, 2, 2]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(permute_3, [1, 16, 2, 32])
    view_4 = torch.ops.aten.view.default(expand_3, [16, 2, 32]);  expand_3 = None
    bmm_1 = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
    view_5 = torch.ops.aten.view.default(bmm_1, [1, 16, 2, 32]);  bmm_1 = None
    return (view_5, permute, permute_3)
    