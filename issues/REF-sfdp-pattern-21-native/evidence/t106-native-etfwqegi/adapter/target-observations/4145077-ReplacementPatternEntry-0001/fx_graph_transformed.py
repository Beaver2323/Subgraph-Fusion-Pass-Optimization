


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    permute_default = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
    permute_default_1 = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
    permute_default_2 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1, 3]);  arg3_1 = None
    mul_scalar = torch.ops.aten.mul.Scalar(permute_default, 1.0);  permute_default = None
    permute_default_3 = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
    mul_scalar_1 = torch.ops.aten.mul.Scalar(permute_default_3, 1.0);  permute_default_3 = None
    expand_default = torch.ops.aten.expand.default(mul_scalar, [4, 16, 2, 32]);  mul_scalar = None
    clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
    view_default = torch.ops.aten.view.default(clone_default, [64, 2, 32]);  clone_default = None
    expand_default_1 = torch.ops.aten.expand.default(mul_scalar_1, [4, 16, 32, 2]);  mul_scalar_1 = None
    clone_default_1 = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
    view_default_1 = torch.ops.aten.view.default(clone_default_1, [64, 32, 2]);  clone_default_1 = None
    bmm_default = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
    view_default_2 = torch.ops.aten.view.default(bmm_default, [4, 16, 2, 2]);  bmm_default = None
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
    full_default = torch.ops.aten.full.default([4, 16, 2, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
    where_self = torch.ops.aten.where.self(logical_not_default_1, full_default, div_tensor);  logical_not_default_1 = full_default = div_tensor = None
    expand_default_2 = torch.ops.aten.expand.default(where_self, [4, 16, 2, 2]);  where_self = None
    view_default_3 = torch.ops.aten.view.default(expand_default_2, [64, 2, 2]);  expand_default_2 = None
    expand_default_3 = torch.ops.aten.expand.default(permute_default_2, [4, 16, 2, 32]);  permute_default_2 = None
    clone_default_2 = torch.ops.aten.clone.default(expand_default_3, memory_format = torch.contiguous_format);  expand_default_3 = None
    view_default_4 = torch.ops.aten.view.default(clone_default_2, [64, 2, 32]);  clone_default_2 = None
    bmm_default_1 = torch.ops.aten.bmm.default(view_default_3, view_default_4);  view_default_3 = view_default_4 = None
    view_default_5 = torch.ops.aten.view.default(bmm_default_1, [4, 16, 2, 32]);  bmm_default_1 = None
    return (view_default_5,)
    