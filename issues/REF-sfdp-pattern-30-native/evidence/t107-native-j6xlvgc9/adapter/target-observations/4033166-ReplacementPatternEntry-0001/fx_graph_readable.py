


def forward(self, arg0_1, arg1_1, arg2_1):
    permute = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
    permute_1 = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
    mul = torch.ops.aten.mul.Scalar(permute_1, 0.25);  permute_1 = None
    permute_2 = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
    permute_3 = torch.ops.aten.permute.default(permute, [0, 1, 3, 2]);  permute = None
    mul_1 = torch.ops.aten.mul.Scalar(permute_3, 0.25);  permute_3 = None
    expand = torch.ops.aten.expand.default(mul, [2, 4, 8, 16]);  mul = None
    clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view = torch.ops.aten.view.default(clone, [8, 8, 16]);  clone = None
    expand_1 = torch.ops.aten.expand.default(mul_1, [2, 4, 16, 8]);  mul_1 = None
    clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_1 = torch.ops.aten.view.default(clone_1, [8, 16, 8]);  clone_1 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2 = torch.ops.aten.view.default(bmm, [2, 4, 8, 8]);  bmm = None
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(view_2, torch.float32)
    amax = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
    sub = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
    exp = torch.ops.aten.exp.default(sub);  sub = None
    sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    convert_element_type_3 = torch.ops.prims.convert_element_type.default(div, torch.float16);  div = None
    eq = torch.ops.aten.eq.Scalar(view_2, -inf);  view_2 = None
    logical_not = torch.ops.aten.logical_not.default(eq);  eq = None
    any_1 = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
    logical_not_1 = torch.ops.aten.logical_not.default(any_1);  any_1 = None
    full_default = torch.ops.aten.full.default([2, 4, 8, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
    where = torch.ops.aten.where.self(logical_not_1, full_default, convert_element_type_3);  logical_not_1 = full_default = convert_element_type_3 = None
    expand_2 = torch.ops.aten.expand.default(where, [2, 4, 8, 8]);  where = None
    view_3 = torch.ops.aten.view.default(expand_2, [8, 8, 8]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(permute_2, [2, 4, 8, 16]);  permute_2 = None
    clone_2 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_4 = torch.ops.aten.view.default(clone_2, [8, 8, 16]);  clone_2 = None
    bmm_1 = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
    view_5 = torch.ops.aten.view.default(bmm_1, [2, 4, 8, 16]);  bmm_1 = None
    return (view_5,)
    