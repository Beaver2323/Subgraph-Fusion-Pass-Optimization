


def forward(self, arg0_1, arg1_1, arg2_1):
    full_default = torch.ops.aten.full.default([2, 2], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
    iota_1 = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
    sub = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
    le = torch.ops.aten.le.Scalar(sub, 0);  sub = None
    logical_and = torch.ops.aten.logical_and.default(le, full_default);  le = full_default = None
    logical_not = torch.ops.aten.logical_not.default(logical_and)
    full_default_1 = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where = torch.ops.aten.where.self(logical_not, full_default_1, logical_and);  logical_not = full_default_1 = logical_and = None
    permute_default = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
    permute_default_1 = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
    permute_default_2 = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
    convert_element_type_default = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
    constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(convert_element_type_default, [0, 6], 0.0);  convert_element_type_default = None
    slice_tensor = torch.ops.aten.slice.Tensor(constant_pad_nd_default, -1, 0, 2);  constant_pad_nd_default = None
    expand_default = torch.ops.aten.expand.default(slice_tensor, [4, 16, 2, 2]);  slice_tensor = None
    _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default, permute_default_1, permute_default_2, expand_default, False, scale = 0.3333333333333333);  permute_default = permute_default_1 = permute_default_2 = expand_default = None
    getitem = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
    return (getitem,)
    