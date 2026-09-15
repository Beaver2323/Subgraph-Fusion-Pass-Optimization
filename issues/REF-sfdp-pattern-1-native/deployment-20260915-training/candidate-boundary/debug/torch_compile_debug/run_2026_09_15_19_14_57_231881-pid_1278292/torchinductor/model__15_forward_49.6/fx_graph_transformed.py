class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2, 8, 16]", primals_2: "f32[2, 2, 8, 16]", primals_3: "f32[]", primals_4: "f32[2, 2, 8, 16]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/attention_training_boundary.py:54 in tensor_scale, code: return ((q @ k.transpose(-2,-1)) * extra).softmax(-1) @ v
        permute: "f32[2, 2, 16, 8]" = torch.ops.aten.permute.default(primals_1, [0, 1, 3, 2]);  primals_1 = None
        expand: "f32[2, 2, 8, 16]" = torch.ops.aten.expand.default(primals_2, [2, 2, 8, 16])
        view: "f32[4, 8, 16]" = torch.ops.aten.reshape.default(expand, [4, 8, 16]);  expand = None
        expand_1: "f32[2, 2, 16, 8]" = torch.ops.aten.expand.default(permute, [2, 2, 16, 8])
        view_1: "f32[4, 16, 8]" = torch.ops.aten.reshape.default(expand_1, [4, 16, 8]);  expand_1 = None
        bmm: "f32[4, 8, 8]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f32[2, 2, 8, 8]" = torch.ops.aten.reshape.default(bmm, [2, 2, 8, 8]);  bmm = None

        # No stacktrace found for following nodes
        mul_tensor: "f32[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(view_2, primals_3)
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(1, dtype = torch.float32, device = device(type='npu', index=0), pin_memory = False)
        ge_scalar: "b8[]" = torch.ops.aten.ge.Scalar(primals_3, 0)
        neg_default: "f32[]" = torch.ops.aten.neg.default(scalar_tensor_default)
        where_self: "f32[]" = torch.ops.aten.where.self(ge_scalar, scalar_tensor_default, neg_default);  ge_scalar = scalar_tensor_default = neg_default = None
        mul_tensor_1: "f32[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(view_2, where_self);  view_2 = None
        amax_default: "f32[2, 2, 8, 1]" = torch.ops.aten.amax.default(mul_tensor_1, [-1], True)
        sub_tensor: "f32[2, 2, 8, 8]" = torch.ops.aten.sub.Tensor(mul_tensor_1, amax_default);  mul_tensor_1 = amax_default = None
        mul_tensor_2: "f32[]" = torch.ops.aten.mul.Tensor(where_self, primals_3);  where_self = None
        mul_tensor_3: "f32[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, mul_tensor_2);  sub_tensor = mul_tensor_2 = None
        amax_default_1: "f32[2, 2, 8, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor_1: "f32[2, 2, 8, 8]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default_1);  amax_default_1 = None
        eq_tensor: "b8[2, 2, 8, 8]" = torch.ops.aten.eq.Tensor(mul_tensor, mul_tensor)
        abs_default: "f32[2, 2, 8, 8]" = torch.ops.aten.abs.default(mul_tensor);  mul_tensor = None
        ne_scalar: "b8[2, 2, 8, 8]" = torch.ops.aten.ne.Scalar(abs_default, inf);  abs_default = None
        mul_tensor_4: "b8[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(eq_tensor, ne_scalar);  eq_tensor = ne_scalar = None
        logical_not_default: "b8[2, 2, 8, 8]" = torch.ops.aten.logical_not.default(mul_tensor_4);  mul_tensor_4 = None
        any_dims: "b8[2, 2, 8, 1]" = torch.ops.aten.any.dims(logical_not_default, [-1], True);  logical_not_default = None
        logical_not_default_1: "b8[2, 2, 8, 1]" = torch.ops.aten.logical_not.default(any_dims);  any_dims = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/attention_training_boundary.py:54 in tensor_scale, code: return ((q @ k.transpose(-2,-1)) * extra).softmax(-1) @ v
        where_self_1: "f32[2, 2, 8, 8]" = torch.ops.aten.where.self(logical_not_default_1, mul_tensor_3, sub_tensor_1);  logical_not_default_1 = mul_tensor_3 = sub_tensor_1 = None
        exp: "f32[2, 2, 8, 8]" = torch.ops.aten.exp.default(where_self_1);  where_self_1 = None
        sum_1: "f32[2, 2, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[2, 2, 8, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        expand_2: "f32[2, 2, 8, 8]" = torch.ops.aten.expand.default(div, [2, 2, 8, 8])
        view_3: "f32[4, 8, 8]" = torch.ops.aten.reshape.default(expand_2, [4, 8, 8]);  expand_2 = None
        expand_3: "f32[2, 2, 8, 16]" = torch.ops.aten.expand.default(primals_4, [2, 2, 8, 16])
        view_4: "f32[4, 8, 16]" = torch.ops.aten.reshape.default(expand_3, [4, 8, 16]);  expand_3 = None
        bmm_1: "f32[4, 8, 16]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f32[2, 2, 8, 16]" = torch.ops.aten.reshape.default(bmm_1, [2, 2, 8, 16]);  bmm_1 = None
        return (view_5, primals_2, primals_3, primals_4, permute, div)
