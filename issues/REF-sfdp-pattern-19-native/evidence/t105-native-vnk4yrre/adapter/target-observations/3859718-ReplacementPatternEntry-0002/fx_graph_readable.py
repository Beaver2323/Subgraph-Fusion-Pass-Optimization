


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
    permute = torch.ops.aten.permute.default(arg2_1, [0, 1, 3, 2]);  arg2_1 = None
    expand = torch.ops.aten.expand.default(arg3_1, [4, 2, 16, 32]);  arg3_1 = None
    view = torch.ops.aten.view.default(expand, [8, 16, 32]);  expand = None
    expand_1 = torch.ops.aten.expand.default(permute, [4, 2, 32, 16]);  permute = None
    view_1 = torch.ops.aten.view.default(expand_1, [8, 32, 16]);  expand_1 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2 = torch.ops.aten.view.default(bmm, [4, 2, 16, 16]);  bmm = None
    full_default = torch.ops.aten.full.default([], 5.75685424949238, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
    div = torch.ops.aten.div.Tensor(view_2, full_default);  view_2 = full_default = None
    full_default_1 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
    where = torch.ops.aten.where.self(arg1_1, div, full_default_1);  arg1_1 = div = full_default_1 = None
    add = torch.ops.aten.add.Tensor(where, arg0_1);  where = arg0_1 = None
    amax = torch.ops.aten.amax.default(add, [-1], True)
    sub = torch.ops.aten.sub.Tensor(add, amax);  add = amax = None
    exp = torch.ops.aten.exp.default(sub);  sub = None
    sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    expand_2 = torch.ops.aten.expand.default(div_1, [4, 2, 16, 16]);  div_1 = None
    view_3 = torch.ops.aten.view.default(expand_2, [8, 16, 16]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(arg4_1, [4, 2, 16, 32]);  arg4_1 = None
    view_4 = torch.ops.aten.view.default(expand_3, [8, 16, 32]);  expand_3 = None
    bmm_1 = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
    view_5 = torch.ops.aten.view.default(bmm_1, [4, 2, 16, 32]);  bmm_1 = None
    return (view_5,)
    