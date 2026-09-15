class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f16[2, 4, 8, 16]", primals_2: "f16[2, 4, 8, 16]", primals_3: "f16[2, 4, 8, 16]"):
        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:111 in _sfdp_pattern_4, code: torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1),
        permute: "f16[2, 4, 16, 8]" = torch.ops.aten.permute.default(primals_1, [0, 1, 3, 2]);  primals_1 = None
        expand: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(primals_2, [2, 4, 8, 16])
        view: "f16[8, 8, 16]" = torch.ops.aten.view.default(expand, [8, 8, 16]);  expand = None
        expand_1: "f16[2, 4, 16, 8]" = torch.ops.aten.expand.default(permute, [2, 4, 16, 8])
        view_1: "f16[8, 16, 8]" = torch.ops.aten.view.default(expand_1, [8, 16, 8]);  expand_1 = None
        bmm: "f16[8, 8, 8]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f16[2, 4, 8, 8]" = torch.ops.aten.view.default(bmm, [2, 4, 8, 8]);  bmm = None

        # No stacktrace found for following nodes
        mul_tensor: "f16[2, 4, 8, 8]" = torch.ops.aten.mul.Tensor(view_2, 2.0)
        convert_element_type_default: "f32[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        convert_element_type_default_1: "f32[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        mul_tensor_1: "f32[2, 4, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        amax_default: "f32[2, 4, 8, 1]" = torch.ops.aten.amax.default(mul_tensor_1, [-1], True)
        sub_tensor: "f32[2, 4, 8, 8]" = torch.ops.aten.sub.Tensor(mul_tensor_1, amax_default);  mul_tensor_1 = amax_default = None
        mul_tensor_2: "f32[2, 4, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, 2.0);  sub_tensor = None
        amax_default_1: "f32[2, 4, 8, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor_1: "f32[2, 4, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default_1);  amax_default_1 = None
        eq_tensor: "b8[2, 4, 8, 8]" = torch.ops.aten.eq.Tensor(convert_element_type_default, convert_element_type_default)
        abs_default: "f32[2, 4, 8, 8]" = torch.ops.aten.abs.default(convert_element_type_default);  convert_element_type_default = None
        ne_scalar: "b8[2, 4, 8, 8]" = torch.ops.aten.ne.Scalar(abs_default, inf);  abs_default = None
        mul_tensor_3: "b8[2, 4, 8, 8]" = torch.ops.aten.mul.Tensor(eq_tensor, ne_scalar);  eq_tensor = ne_scalar = None
        logical_not_default: "b8[2, 4, 8, 8]" = torch.ops.aten.logical_not.default(mul_tensor_3);  mul_tensor_3 = None
        any_dims: "b8[2, 4, 8, 1]" = torch.ops.aten.any.dims(logical_not_default, [-1], True);  logical_not_default = None
        logical_not_default_1: "b8[2, 4, 8, 1]" = torch.ops.aten.logical_not.default(any_dims);  any_dims = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:111 in _sfdp_pattern_4, code: torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1),
        where_self: "f32[2, 4, 8, 8]" = torch.ops.aten.where.self(logical_not_default_1, mul_tensor_2, sub_tensor_1);  logical_not_default_1 = mul_tensor_2 = sub_tensor_1 = None
        exp: "f32[2, 4, 8, 8]" = torch.ops.aten.exp.default(where_self);  where_self = None
        sum_1: "f32[2, 4, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[2, 4, 8, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "f16[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(div, torch.float16);  div = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:110 in _sfdp_pattern_4, code: return torch.nn.functional.dropout(
        _npu_dropout = torch.ops.npu._npu_dropout.default(convert_element_type_3, 1e-11)
        getitem: "f16[2, 4, 8, 8]" = _npu_dropout[0]
        getitem_1: "u8[64]" = _npu_dropout[1];  _npu_dropout = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:113 in _sfdp_pattern_4, code: ).matmul(value)
        expand_2: "f16[2, 4, 8, 8]" = torch.ops.aten.expand.default(getitem, [2, 4, 8, 8])
        view_3: "f16[8, 8, 8]" = torch.ops.aten.view.default(expand_2, [8, 8, 8]);  expand_2 = None
        expand_3: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(primals_3, [2, 4, 8, 16])
        view_4: "f16[8, 8, 16]" = torch.ops.aten.view.default(expand_3, [8, 8, 16]);  expand_3 = None
        bmm_1: "f16[8, 8, 16]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f16[2, 4, 8, 16]" = torch.ops.aten.view.default(bmm_1, [2, 4, 8, 16]);  bmm_1 = None
        return (view_5, primals_2, primals_3, permute, convert_element_type_3, getitem, getitem_1)
