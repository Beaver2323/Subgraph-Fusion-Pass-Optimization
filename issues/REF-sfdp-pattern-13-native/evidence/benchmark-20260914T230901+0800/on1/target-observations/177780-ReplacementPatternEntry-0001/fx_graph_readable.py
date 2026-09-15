


def forward(self, arg0_1, arg1_1, arg2_1):
    permute = torch.ops.aten.permute.default(arg0_1, [0, 2, 1]);  arg0_1 = None
    bmm = torch.ops.aten.bmm.default(arg1_1, permute);  arg1_1 = permute = None
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(bmm, torch.float32);  bmm = None
    amax = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
    sub = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
    exp = torch.ops.aten.exp.default(sub);  sub = None
    sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    convert_element_type_3 = torch.ops.prims.convert_element_type.default(div, torch.float16);  div = None
    bmm_1 = torch.ops.aten.bmm.default(convert_element_type_3, arg2_1);  convert_element_type_3 = arg2_1 = None
    return (bmm_1,)
    