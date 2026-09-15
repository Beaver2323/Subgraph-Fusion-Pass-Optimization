class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 3, 96]", primals_2: "f32[32]", primals_3: "f32[32]", primals_4: "f32[32, 3, 3]", primals_5: "f32[32]", mul_1: "f32[32, 3, 3]", tangents_1: "f32[4, 32, 47]"):
        # No stacktrace found for following nodes
        convolution_backward = torch.ops.aten.convolution_backward.default(tangents_1, primals_1, mul_1, [32], [2], [0], [1], False, [0], 1, [False, True, True]);  tangents_1 = primals_1 = mul_1 = None
        getitem_1: "f32[32, 3, 3]" = convolution_backward[1]
        getitem_2: "f32[32]" = convolution_backward[2];  convolution_backward = None
        full_default: "f32[32]" = torch.ops.aten.full.default([32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        sub: "f32[32]" = torch.ops.aten.sub.Tensor(full_default, primals_2);  full_default = primals_2 = None
        mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(getitem_2, sub);  sub = None
        view_3: "f32[32, 1, 1]" = torch.ops.aten.view.default(mul_3, [32, 1, 1]);  mul_3 = None
        mul_4: "f32[32, 3, 3]" = torch.ops.aten.mul.Tensor(getitem_1, primals_4);  primals_4 = None
        add: "f32[32]" = torch.ops.aten.add.Tensor(primals_3, 1e-05);  primals_3 = None
        rsqrt: "f32[32]" = torch.ops.aten.rsqrt.default(add);  add = None
        view: "f32[32, 1, 1]" = torch.ops.aten.view.default(rsqrt, [-1, 1, 1]);  rsqrt = None
        view_1: "f32[32, 1, 1]" = torch.ops.aten.view.default(primals_5, [32, 1, 1]);  primals_5 = None
        mul: "f32[32, 1, 1]" = torch.ops.aten.mul.Tensor(view_1, view);  view_1 = None
        mul_5: "f32[32, 3, 3]" = torch.ops.aten.mul.Tensor(getitem_1, mul);  getitem_1 = mul = None
        sum_1: "f32[32, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [1, 2], True);  mul_4 = None
        add_2: "f32[32, 1, 1]" = torch.ops.aten.add.Tensor(view_3, sum_1);  view_3 = sum_1 = None
        mul_6: "f32[32, 1, 1]" = torch.ops.aten.mul.Tensor(add_2, view);  add_2 = view = None
        view_4: "f32[32]" = torch.ops.aten.view.default(mul_6, [32]);  mul_6 = None
        return (None, None, None, mul_5, view_4, getitem_2)
