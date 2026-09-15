


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    permute = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
    permute_1 = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
    permute_2 = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
    permute_3 = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2])
    expand = torch.ops.aten.expand.default(permute, [2, 8, 4, 16]);  permute = None
    clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view = torch.ops.aten.view.default(clone, [16, 4, 16]);  clone = None
    expand_1 = torch.ops.aten.expand.default(permute_3, [2, 8, 16, 4]);  permute_3 = None
    clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_1 = torch.ops.aten.view.default(clone_1, [16, 16, 4]);  clone_1 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2 = torch.ops.aten.view.default(bmm, [2, 8, 4, 4]);  bmm = None
    full_default = torch.ops.aten.full.default([], 0.66666, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
    div = torch.ops.aten.div.Tensor(view_2, full_default);  view_2 = full_default = None
    full_default_1 = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
    where = torch.ops.aten.where.self(arg3_1, div, full_default_1);  arg3_1 = div = full_default_1 = None
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
    amax = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
    sub = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
    exp = torch.ops.aten.exp.default(sub);  sub = None
    sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    convert_element_type_3 = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None
    expand_2 = torch.ops.aten.expand.default(convert_element_type_3, [2, 8, 4, 4]);  convert_element_type_3 = None
    view_3 = torch.ops.aten.view.default(expand_2, [16, 4, 4]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(permute_2, [2, 8, 4, 16])
    clone_2 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_4 = torch.ops.aten.view.default(clone_2, [16, 4, 16]);  clone_2 = None
    bmm_1 = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
    view_5 = torch.ops.aten.view.default(bmm_1, [2, 8, 4, 16]);  bmm_1 = None
    return (view_5, permute_1, permute_2)
    