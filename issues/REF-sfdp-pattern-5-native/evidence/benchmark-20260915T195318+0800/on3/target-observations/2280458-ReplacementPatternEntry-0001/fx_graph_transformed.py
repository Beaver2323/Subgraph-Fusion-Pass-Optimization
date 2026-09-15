


def forward(self, primals, tangents):
    primals_1, primals_2, primals_3, primals_4, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    convert_element_type_default = torch.ops.prims.convert_element_type.default(primals_2, torch.float32);  primals_2 = None
    convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
    convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(primals_4, torch.float32);  primals_4 = None
    mul_scalar = torch.ops.aten.mul.Scalar(convert_element_type_default, 1.2247509951618742);  convert_element_type_default = None
    permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [0, 1, 3, 2]);  convert_element_type_default_1 = None
    mul_scalar_1 = torch.ops.aten.mul.Scalar(permute_default, 1.2247509951618742);  permute_default = None
    expand_default = torch.ops.aten.expand.default(mul_scalar, [2, 4, 8, 16])
    view_default = torch.ops.aten.view.default(expand_default, [8, 8, 16]);  expand_default = None
    expand_default_1 = torch.ops.aten.expand.default(mul_scalar_1, [2, 4, 16, 8])
    view_default_1 = torch.ops.aten.view.default(expand_default_1, [8, 16, 8]);  expand_default_1 = None
    bmm_default = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
    view_default_2 = torch.ops.aten.view.default(bmm_default, [2, 4, 8, 8]);  bmm_default = None
    add_tensor = torch.ops.aten.add.Tensor(view_default_2, primals_3);  view_default_2 = primals_3 = None
    amax_default = torch.ops.aten.amax.default(add_tensor, [-1], True)
    sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  amax_default = None
    exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
    div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
    eq_scalar = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
    logical_not_default = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
    any_dim = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
    logical_not_default_1 = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
    full_default = torch.ops.aten.full.default([2, 4, 8, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
    where_self = torch.ops.aten.where.self(logical_not_default_1, full_default, div_tensor);  logical_not_default_1 = full_default = div_tensor = None
    expand_default_2 = torch.ops.aten.expand.default(where_self, [2, 4, 8, 8])
    view_default_3 = torch.ops.aten.view.default(expand_default_2, [8, 8, 8]);  expand_default_2 = None
    expand_default_3 = torch.ops.aten.expand.default(convert_element_type_default_2, [2, 4, 8, 16])
    view_default_4 = torch.ops.aten.view.default(expand_default_3, [8, 8, 16]);  expand_default_3 = None
    bmm_default_1 = torch.ops.aten.bmm.default(view_default_3, view_default_4);  view_default_3 = view_default_4 = None
    view_default_5 = torch.ops.aten.view.default(bmm_default_1, [2, 4, 8, 16]);  bmm_default_1 = None
    convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float16);  view_default_5 = None
    convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32);  tangents_1 = None
    matmul_backward_default = torch.ops.aten.matmul_backward.default(convert_element_type_default_4, where_self, convert_element_type_default_2, [True, True]);  convert_element_type_default_4 = convert_element_type_default_2 = None
    getitem_4 = matmul_backward_default[0]
    getitem_5 = matmul_backward_default[1];  matmul_backward_default = None
    mul_tensor = torch.ops.aten.mul.Tensor(getitem_4, where_self);  getitem_4 = None
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(where_self, sum_dim_int_list_1);  where_self = sum_dim_int_list_1 = None
    sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
    matmul_backward_default_1 = torch.ops.aten.matmul_backward.default(sub_tensor_1, mul_scalar, mul_scalar_1, [True, True]);  sub_tensor_1 = mul_scalar = mul_scalar_1 = None
    getitem_6 = matmul_backward_default_1[0]
    getitem_7 = matmul_backward_default_1[1];  matmul_backward_default_1 = None
    mul_scalar_2 = torch.ops.aten.mul.Scalar(getitem_7, 1.2247509951618742);  getitem_7 = None
    permute_default_1 = torch.ops.aten.permute.default(mul_scalar_2, [0, 1, 3, 2]);  mul_scalar_2 = None
    mul_scalar_3 = torch.ops.aten.mul.Scalar(getitem_6, 1.2247509951618742);  getitem_6 = None
    convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(getitem_5, torch.float16);  getitem_5 = None
    convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(permute_default_1, torch.float16);  permute_default_1 = None
    convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_scalar_3, torch.float16);  mul_scalar_3 = None
    return pytree.tree_unflatten([convert_element_type_default_3, convert_element_type_default_6, convert_element_type_default_7, None, convert_element_type_default_5], self._out_spec)
    