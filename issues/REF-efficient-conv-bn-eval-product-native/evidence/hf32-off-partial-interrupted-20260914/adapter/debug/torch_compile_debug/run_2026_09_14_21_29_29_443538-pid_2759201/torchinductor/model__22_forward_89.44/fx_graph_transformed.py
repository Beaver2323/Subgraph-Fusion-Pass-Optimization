class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32]", primals_2: "f32[32]", primals_3: "f32[32]", primals_4: "f32[32]", primals_5: "f32[32]", primals_6: "f32[3, 32, 3, 3]", primals_7: "f32[4, 3, 4, 4]"):
        # No stacktrace found for following nodes
        add: "f32[32]" = torch.ops.aten.add.Tensor(primals_2, 1e-05)
        rsqrt: "f32[32]" = torch.ops.aten.rsqrt.default(add);  add = None
        view: "f32[1, 32, 1, 1]" = torch.ops.aten.reshape.default(rsqrt, [1, -1, 1, 1]);  rsqrt = None
        view_1: "f32[1, 32, 1, 1]" = torch.ops.aten.reshape.default(primals_4, [1, 32, 1, 1])
        mul: "f32[1, 32, 1, 1]" = torch.ops.aten.mul.Tensor(view_1, view);  view_1 = view = None
        mul_1: "f32[3, 32, 3, 3]" = torch.ops.aten.mul.Tensor(primals_6, mul)
        view_2: "f32[32]" = torch.ops.aten.reshape.default(mul, [32]);  mul = None
        sub: "f32[32]" = torch.ops.aten.sub.Tensor(primals_5, primals_1)
        mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(view_2, sub);  view_2 = sub = None
        add_1: "f32[32]" = torch.ops.aten.add.Tensor(primals_3, mul_2);  primals_3 = mul_2 = None
        convolution: "f32[4, 32, 9, 9]" = torch.ops.aten.convolution.default(primals_7, mul_1, add_1, [2, 2], [0, 0], [1, 1], True, [0, 0], 1);  add_1 = None
        return (convolution, primals_1, primals_2, primals_4, primals_5, primals_6, primals_7, mul_1)
