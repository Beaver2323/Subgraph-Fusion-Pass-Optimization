


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    permute = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
    permute_1 = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
    permute_2 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1, 3]);  arg3_1 = None
    permute_3 = torch.ops.aten.permute.default(permute, [0, 1, 3, 2])
    expand = torch.ops.aten.expand.default(permute_1, [4, 16, 2, 32]);  permute_1 = None
    clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view = torch.ops.aten.view.default(clone, [64, 2, 32]);  clone = None
    expand_1 = torch.ops.aten.expand.default(permute_3, [4, 16, 32, 2]);  permute_3 = None
    clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_1 = torch.ops.aten.view.default(clone_1, [64, 32, 2]);  clone_1 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2 = torch.ops.aten.view.default(bmm, [4, 16, 2, 2]);  bmm = None
    full_default = torch.ops.aten.full.default([], 5.75685424949238, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
    div = torch.ops.aten.div.Tensor(view_2, full_default);  view_2 = full_default = None
    full_default_1 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
    permute_4 = torch.ops.aten.permute.default(permute, [0, 2, 1, 3]);  permute = None
    permute_5 = torch.ops.aten.permute.default(permute_2, [0, 2, 1, 3])
    where = torch.ops.aten.where.self(arg0_1, div, full_default_1);  arg0_1 = div = full_default_1 = None
    amax = torch.ops.aten.amax.default(where, [-1], True)
    sub = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp = torch.ops.aten.exp.default(sub);  sub = None
    sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    expand_2 = torch.ops.aten.expand.default(div_1, [4, 16, 2, 2]);  div_1 = None
    view_3 = torch.ops.aten.view.default(expand_2, [64, 2, 2]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(permute_2, [4, 16, 2, 32]);  permute_2 = None
    clone_2 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_4 = torch.ops.aten.view.default(clone_2, [64, 2, 32]);  clone_2 = None
    bmm_1 = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
    view_5 = torch.ops.aten.view.default(bmm_1, [4, 16, 2, 32]);  bmm_1 = None
    return (view_5, permute_4, permute_5)
    