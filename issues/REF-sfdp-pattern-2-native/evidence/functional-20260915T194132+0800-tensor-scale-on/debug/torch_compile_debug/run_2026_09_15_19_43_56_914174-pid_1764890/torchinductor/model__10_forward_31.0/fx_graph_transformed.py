class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f16[2, 4, 8, 16]", primals_2: "f16[2, 4, 8, 16]", primals_3: "f16[]", primals_4: "f16[2, 4, 8, 16]"):
        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:67 in _sfdp_pattern_2, code: torch.matmul(query, key.transpose(-2, -1))
        permute: "f16[2, 4, 16, 8]" = torch.ops.aten.permute.default(primals_1, [0, 1, 3, 2]);  primals_1 = None
        expand: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(primals_2, [2, 4, 8, 16])
        view: "f16[8, 8, 16]" = torch.ops.aten.reshape.default(expand, [8, 8, 16]);  expand = None
        expand_1: "f16[2, 4, 16, 8]" = torch.ops.aten.expand.default(permute, [2, 4, 16, 8])
        view_1: "f16[8, 16, 8]" = torch.ops.aten.reshape.default(expand_1, [8, 16, 8]);  expand_1 = None
        bmm: "f16[8, 8, 8]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f16[2, 4, 8, 8]" = torch.ops.aten.reshape.default(bmm, [2, 4, 8, 8]);  bmm = None

        # No stacktrace found for following nodes
        mul_tensor: "f16[2, 4, 8, 8]" = torch.ops.aten.mul.Tensor(view_2, primals_3)
        convert_element_type_default: "f32[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        convert_element_type_default_1: "f32[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='npu', index=0), pin_memory = False)
        ge_scalar: "b8[]" = torch.ops.aten.ge.Scalar(primals_3, 0)
        neg_default: "f32[]" = torch.ops.aten.neg.default(scalar_tensor_default)
        where_self: "f32[]" = torch.ops.aten.where.self(ge_scalar, scalar_tensor_default, neg_default);  ge_scalar = scalar_tensor_default = neg_default = None
        mul_tensor_1: "f32[2, 4, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, where_self);  convert_element_type_default_1 = None
        amax_default: "f32[2, 4, 8, 1]" = torch.ops.aten.amax.default(mul_tensor_1, [-1], True)
        sub_tensor: "f32[2, 4, 8, 8]" = torch.ops.aten.sub.Tensor(mul_tensor_1, amax_default);  mul_tensor_1 = amax_default = None
        mul_tensor_2: "f32[]" = torch.ops.aten.mul.Tensor(where_self, primals_3);  where_self = None
        mul_tensor_3: "f32[2, 4, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, mul_tensor_2);  sub_tensor = mul_tensor_2 = None
        amax_default_1: "f32[2, 4, 8, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor_1: "f32[2, 4, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default_1);  amax_default_1 = None
        eq_tensor: "b8[2, 4, 8, 8]" = torch.ops.aten.eq.Tensor(convert_element_type_default, convert_element_type_default)
        abs_default: "f32[2, 4, 8, 8]" = torch.ops.aten.abs.default(convert_element_type_default);  convert_element_type_default = None
        ne_scalar: "b8[2, 4, 8, 8]" = torch.ops.aten.ne.Scalar(abs_default, inf);  abs_default = None
        mul_tensor_4: "b8[2, 4, 8, 8]" = torch.ops.aten.mul.Tensor(eq_tensor, ne_scalar);  eq_tensor = ne_scalar = None
        logical_not_default: "b8[2, 4, 8, 8]" = torch.ops.aten.logical_not.default(mul_tensor_4);  mul_tensor_4 = None
        any_dims: "b8[2, 4, 8, 1]" = torch.ops.aten.any.dims(logical_not_default, [-1], True);  logical_not_default = None
        logical_not_default_1: "b8[2, 4, 8, 1]" = torch.ops.aten.logical_not.default(any_dims);  any_dims = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:69 in _sfdp_pattern_2, code: .softmax(dim=-1)
        where_self_1: "f32[2, 4, 8, 8]" = torch.ops.aten.where.self(logical_not_default_1, mul_tensor_3, sub_tensor_1);  logical_not_default_1 = mul_tensor_3 = sub_tensor_1 = None
        exp: "f32[2, 4, 8, 8]" = torch.ops.aten.exp.default(where_self_1);  where_self_1 = None
        sum_1: "f32[2, 4, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[2, 4, 8, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "f16[2, 4, 8, 8]" = torch.ops.prims.convert_element_type.default(div, torch.float16);  div = None

        # File: /home/z50063656/Pass/src/pytorch/torch/_inductor/fx_passes/fuse_attention.py:70 in _sfdp_pattern_2, code: .matmul(value)
        expand_2: "f16[2, 4, 8, 8]" = torch.ops.aten.expand.default(convert_element_type_3, [2, 4, 8, 8])
        view_3: "f16[8, 8, 8]" = torch.ops.aten.reshape.default(expand_2, [8, 8, 8]);  expand_2 = None
        expand_3: "f16[2, 4, 8, 16]" = torch.ops.aten.expand.default(primals_4, [2, 4, 8, 16])
        view_4: "f16[8, 8, 16]" = torch.ops.aten.reshape.default(expand_3, [8, 8, 16]);  expand_3 = None
        bmm_1: "f16[8, 8, 16]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f16[2, 4, 8, 16]" = torch.ops.aten.reshape.default(bmm_1, [2, 4, 8, 16]);  bmm_1 = None
        return (view_5, primals_2, primals_3, primals_4, permute, convert_element_type_3)
