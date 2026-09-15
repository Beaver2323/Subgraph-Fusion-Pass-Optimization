


def forward(self, primals, tangents):
    primals_1, primals_2, primals_3, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    full_default = torch.ops.aten.full.default([16, 16], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
    iota = torch.ops.prims.iota.default(16, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
    unsqueeze = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
    iota_1 = torch.ops.prims.iota.default(16, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
    sub = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
    le = torch.ops.aten.le.Scalar(sub, 0);  sub = None
    logical_and = torch.ops.aten.logical_and.default(le, full_default);  le = full_default = None
    logical_not = torch.ops.aten.logical_not.default(logical_and)
    full_default_1 = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
    where = torch.ops.aten.where.self(logical_not, full_default_1, logical_and);  logical_not = full_default_1 = logical_and = None
    convert_element_type_default = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
    mul_scalar = torch.ops.aten.mul.Scalar(primals_1, 0.42044820762685725);  primals_1 = None
    permute_default = torch.ops.aten.permute.default(primals_2, [0, 1, 3, 2]);  primals_2 = None
    mul_scalar_1 = torch.ops.aten.mul.Scalar(permute_default, 0.42044820762685725);  permute_default = None
    expand_default = torch.ops.aten.expand.default(mul_scalar, [4, 2, 16, 32])
    view_default = torch.ops.aten.view.default(expand_default, [8, 16, 32]);  expand_default = None
    expand_default_1 = torch.ops.aten.expand.default(mul_scalar_1, [4, 2, 32, 16])
    view_default_1 = torch.ops.aten.view.default(expand_default_1, [8, 32, 16]);  expand_default_1 = None
    bmm_default = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
    view_default_2 = torch.ops.aten.view.default(bmm_default, [4, 2, 16, 16]);  bmm_default = None
    add_tensor = torch.ops.aten.add.Tensor(view_default_2, convert_element_type_default);  view_default_2 = convert_element_type_default = None
    amax_default = torch.ops.aten.amax.default(add_tensor, [-1], True)
    sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  amax_default = None
    exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
    div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
    eq_scalar = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
    logical_not_default = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
    any_dim = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
    logical_not_default_1 = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
    full_default_2 = torch.ops.aten.full.default([4, 2, 16, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
    where_self = torch.ops.aten.where.self(logical_not_default_1, full_default_2, div_tensor);  logical_not_default_1 = full_default_2 = div_tensor = None
    expand_default_2 = torch.ops.aten.expand.default(where_self, [4, 2, 16, 16])
    view_default_3 = torch.ops.aten.view.default(expand_default_2, [8, 16, 16]);  expand_default_2 = None
    expand_default_3 = torch.ops.aten.expand.default(primals_3, [4, 2, 16, 32])
    view_default_4 = torch.ops.aten.view.default(expand_default_3, [8, 16, 32]);  expand_default_3 = None
    bmm_default_1 = torch.ops.aten.bmm.default(view_default_3, view_default_4);  view_default_3 = view_default_4 = None
    view_default_5 = torch.ops.aten.view.default(bmm_default_1, [4, 2, 16, 32]);  bmm_default_1 = None
    matmul_backward_default = torch.ops.aten.matmul_backward.default(tangents_1, where_self, primals_3, [True, True]);  tangents_1 = primals_3 = None
    getitem_4 = matmul_backward_default[0]
    getitem_5 = matmul_backward_default[1];  matmul_backward_default = None
    mul_tensor = torch.ops.aten.mul.Tensor(getitem_4, where_self);  getitem_4 = None
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(where_self, sum_dim_int_list_1);  where_self = sum_dim_int_list_1 = None
    sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
    matmul_backward_default_1 = torch.ops.aten.matmul_backward.default(sub_tensor_1, mul_scalar, mul_scalar_1, [True, True]);  sub_tensor_1 = mul_scalar = mul_scalar_1 = None
    getitem_6 = matmul_backward_default_1[0]
    getitem_7 = matmul_backward_default_1[1];  matmul_backward_default_1 = None
    mul_scalar_2 = torch.ops.aten.mul.Scalar(getitem_7, 0.42044820762685725);  getitem_7 = None
    permute_default_1 = torch.ops.aten.permute.default(mul_scalar_2, [0, 1, 3, 2]);  mul_scalar_2 = None
    mul_scalar_3 = torch.ops.aten.mul.Scalar(getitem_6, 0.42044820762685725);  getitem_6 = None
    return pytree.tree_unflatten([view_default_5, mul_scalar_3, permute_default_1, getitem_5], self._out_spec)
    