class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32]", primals_2: "f32[32]", primals_4: "f32[32]", primals_5: "f32[32]", primals_6: "f32[3, 32, 3, 3]", primals_7: "f32[4, 3, 4, 4]", mul_1: "f32[3, 32, 3, 3]", tangents_1: "f32[4, 32, 9, 9]"):
        # No stacktrace found for following nodes
        convolution_backward = torch.ops.aten.convolution_backward.default(tangents_1, primals_7, mul_1, [32], [2, 2], [0, 0], [1, 1], True, [0, 0], 1, [False, True, True]);  tangents_1 = primals_7 = mul_1 = None
        getitem_1: "f32[3, 32, 3, 3]" = convolution_backward[1]
        getitem_2: "f32[32]" = convolution_backward[2];  convolution_backward = None
        add: "f32[32]" = torch.ops.aten.add.Tensor(primals_2, 1e-05);  primals_2 = None
        rsqrt: "f32[32]" = torch.ops.aten.rsqrt.default(add);  add = None
        view: "f32[1, 32, 1, 1]" = torch.ops.aten.view.default(rsqrt, [1, -1, 1, 1]);  rsqrt = None
        view_1: "f32[1, 32, 1, 1]" = torch.ops.aten.view.default(primals_4, [1, 32, 1, 1]);  primals_4 = None
        mul: "f32[1, 32, 1, 1]" = torch.ops.aten.mul.Tensor(view_1, view);  view_1 = None
        view_2: "f32[32]" = torch.ops.aten.view.default(mul, [32])
        mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(getitem_2, view_2);  view_2 = None
        sub: "f32[32]" = torch.ops.aten.sub.Tensor(primals_5, primals_1);  primals_5 = primals_1 = None
        mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(getitem_2, sub);  sub = None
        view_3: "f32[1, 32, 1, 1]" = torch.ops.aten.view.default(mul_4, [1, 32, 1, 1]);  mul_4 = None
        mul_5: "f32[3, 32, 3, 3]" = torch.ops.aten.mul.Tensor(getitem_1, primals_6);  primals_6 = None
        mul_6: "f32[3, 32, 3, 3]" = torch.ops.aten.mul.Tensor(getitem_1, mul);  getitem_1 = mul = None
        sum_1: "f32[1, 32, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 2, 3], True);  mul_5 = None
        add_2: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(view_3, sum_1);  view_3 = sum_1 = None
        mul_7: "f32[1, 32, 1, 1]" = torch.ops.aten.mul.Tensor(add_2, view);  add_2 = view = None
        view_4: "f32[32]" = torch.ops.aten.view.default(mul_7, [32]);  mul_7 = None
        return (None, None, getitem_2, view_4, mul_3, mul_6, None)
