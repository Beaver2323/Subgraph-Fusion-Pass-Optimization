class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2, 8, 16]", primals_2: "f32[2, 2, 8, 16]", primals_3: "f32[8, 8]", primals_4: "f32[2, 2, 8, 16]"):
        # No stacktrace found for following nodes
        mul_scalar: "f32[2, 2, 8, 16]" = torch.ops.aten.mul.Scalar(primals_2, 0.5);  primals_2 = None
        permute_default: "f32[2, 2, 16, 8]" = torch.ops.aten.permute.default(primals_1, [0, 1, 3, 2]);  primals_1 = None
        mul_scalar_1: "f32[2, 2, 16, 8]" = torch.ops.aten.mul.Scalar(permute_default, 0.5);  permute_default = None
        expand_default: "f32[2, 2, 8, 16]" = torch.ops.aten.expand.default(mul_scalar, [2, 2, 8, 16])
        view_default: "f32[4, 8, 16]" = torch.ops.aten.view.default(expand_default, [4, 8, 16]);  expand_default = None
        expand_default_1: "f32[2, 2, 16, 8]" = torch.ops.aten.expand.default(mul_scalar_1, [2, 2, 16, 8])
        view_default_1: "f32[4, 16, 8]" = torch.ops.aten.view.default(expand_default_1, [4, 16, 8]);  expand_default_1 = None
        bmm_default: "f32[4, 8, 8]" = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[2, 2, 8, 8]" = torch.ops.aten.view.default(bmm_default, [2, 2, 8, 8]);  bmm_default = None
        add_tensor: "f32[2, 2, 8, 8]" = torch.ops.aten.add.Tensor(view_default_2, primals_3);  view_default_2 = primals_3 = None
        amax_default: "f32[2, 2, 8, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[2, 2, 8, 8]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  amax_default = None
        exp_default: "f32[2, 2, 8, 8]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[2, 2, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[2, 2, 8, 8]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        eq_scalar: "b8[2, 2, 8, 8]" = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
        logical_not_default: "b8[2, 2, 8, 8]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[2, 2, 8, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[2, 2, 8, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default: "f32[2, 2, 8, 8]" = torch.ops.aten.full.default([2, 2, 8, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_self: "f32[2, 2, 8, 8]" = torch.ops.aten.where.self(logical_not_default_1, full_default, div_tensor);  logical_not_default_1 = full_default = div_tensor = None
        expand_default_2: "f32[2, 2, 8, 8]" = torch.ops.aten.expand.default(where_self, [2, 2, 8, 8])
        view_default_3: "f32[4, 8, 8]" = torch.ops.aten.view.default(expand_default_2, [4, 8, 8]);  expand_default_2 = None
        expand_default_3: "f32[2, 2, 8, 16]" = torch.ops.aten.expand.default(primals_4, [2, 2, 8, 16])
        view_default_4: "f32[4, 8, 16]" = torch.ops.aten.view.default(expand_default_3, [4, 8, 16]);  expand_default_3 = None
        bmm_default_1: "f32[4, 8, 16]" = torch.ops.aten.bmm.default(view_default_3, view_default_4);  view_default_3 = view_default_4 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/attention_training_boundary.py:57 in mask, code: return ((q @ k.transpose(-2,-1)).div(4.0) + extra).softmax(-1) @ v
        view_default_5: "f32[2, 2, 8, 16]" = torch.ops.aten.view.default(bmm_default_1, [2, 2, 8, 16]);  bmm_default_1 = None
        return (view_default_5, primals_4, mul_scalar, mul_scalar_1, where_self)
