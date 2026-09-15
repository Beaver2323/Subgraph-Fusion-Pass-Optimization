


def forward(self, arg0_1):
    permute = torch.ops.aten.permute.default(arg0_1, [1, 0, 2, 4, 3]);  arg0_1 = None
    unbind = torch.ops.aten.unbind.int(permute);  permute = None
    getitem = unbind[0]
    getitem_1 = unbind[1]
    getitem_2 = unbind[2];  unbind = None
    permute_1 = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
    expand = torch.ops.aten.expand.default(getitem, [2, 4, 8, 16]);  getitem = None
    clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view = torch.ops.aten.view.default(clone, [8, 8, 16]);  clone = None
    expand_1 = torch.ops.aten.expand.default(permute_1, [2, 4, 16, 8]);  permute_1 = None
    clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_1 = torch.ops.aten.view.default(clone_1, [8, 16, 8]);  clone_1 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2 = torch.ops.aten.view.default(bmm, [2, 4, 8, 8]);  bmm = None
    mul = torch.ops.aten.mul.Tensor(view_2, 0.2);  view_2 = None
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
    amax = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
    sub = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
    exp = torch.ops.aten.exp.default(sub);  sub = None
    sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    convert_element_type_3 = torch.ops.prims.convert_element_type.default(div, torch.float16);  div = None
    expand_2 = torch.ops.aten.expand.default(convert_element_type_3, [2, 4, 8, 8]);  convert_element_type_3 = None
    view_3 = torch.ops.aten.view.default(expand_2, [8, 8, 8]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(getitem_2, [2, 4, 8, 16]);  getitem_2 = None
    clone_2 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_4 = torch.ops.aten.view.default(clone_2, [8, 8, 16]);  clone_2 = None
    bmm_1 = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
    view_5 = torch.ops.aten.view.default(bmm_1, [2, 4, 8, 16]);  bmm_1 = None
    return (view_5,)
    