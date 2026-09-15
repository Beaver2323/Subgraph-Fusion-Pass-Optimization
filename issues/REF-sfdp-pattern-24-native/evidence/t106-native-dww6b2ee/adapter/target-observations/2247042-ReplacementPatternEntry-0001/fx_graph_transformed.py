


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    mul_scalar = torch.ops.aten.mul.Scalar(arg2_1, 1.0);  arg2_1 = None
    permute_default = torch.ops.aten.permute.default(arg1_1, [0, 1, 3, 2]);  arg1_1 = None
    mul_scalar_1 = torch.ops.aten.mul.Scalar(permute_default, 1.0);  permute_default = None
    expand_default = torch.ops.aten.expand.default(mul_scalar, [4, 2, 16, 32]);  mul_scalar = None
    view_default = torch.ops.aten.view.default(expand_default, [8, 16, 32]);  expand_default = None
    expand_default_1 = torch.ops.aten.expand.default(mul_scalar_1, [4, 2, 32, 16]);  mul_scalar_1 = None
    view_default_1 = torch.ops.aten.view.default(expand_default_1, [8, 32, 16]);  expand_default_1 = None
    bmm_default = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
    view_default_2 = torch.ops.aten.view.default(bmm_default, [4, 2, 16, 16]);  bmm_default = None
    add_tensor = torch.ops.aten.add.Tensor(view_default_2, arg0_1);  view_default_2 = arg0_1 = None
    amax_default = torch.ops.aten.amax.default(add_tensor, [-1], True)
    sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  amax_default = None
    exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
    div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
    eq_scalar = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
    logical_not_default = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
    any_dim = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
    logical_not_default_1 = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
    full_default = torch.ops.aten.full.default([4, 2, 16, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
    where_self = torch.ops.aten.where.self(logical_not_default_1, full_default, div_tensor);  logical_not_default_1 = full_default = div_tensor = None
    expand_default_2 = torch.ops.aten.expand.default(where_self, [4, 2, 16, 16]);  where_self = None
    view_default_3 = torch.ops.aten.view.default(expand_default_2, [8, 16, 16]);  expand_default_2 = None
    expand_default_3 = torch.ops.aten.expand.default(arg3_1, [4, 2, 16, 32]);  arg3_1 = None
    view_default_4 = torch.ops.aten.view.default(expand_default_3, [8, 16, 32]);  expand_default_3 = None
    bmm_default_1 = torch.ops.aten.bmm.default(view_default_3, view_default_4);  view_default_3 = view_default_4 = None
    view_default_5 = torch.ops.aten.view.default(bmm_default_1, [4, 2, 16, 32]);  bmm_default_1 = None
    return (view_default_5,)
    