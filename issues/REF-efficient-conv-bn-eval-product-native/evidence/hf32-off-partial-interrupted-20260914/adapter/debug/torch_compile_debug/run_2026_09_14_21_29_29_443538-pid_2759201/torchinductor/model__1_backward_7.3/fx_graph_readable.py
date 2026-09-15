class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 3]", primals_2: "f32[32]", primals_3: "f32[32]", primals_4: "f32[32, 3]", primals_5: "f32[32]", primals_6: "f32[32]", tangents_1: "f32[4, 32]"):
        # No stacktrace found for following nodes
        permute_1: "f32[32, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm: "f32[32, 3]" = torch.ops.aten.mm.default(permute_1, primals_1);  permute_1 = primals_1 = None
        sum_1: "f32[1, 32]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_3: "f32[32]" = torch.ops.aten.view.default(sum_1, [32]);  sum_1 = None
        add: "f32[32]" = torch.ops.aten.add.Tensor(primals_3, 1e-05);  primals_3 = None
        rsqrt: "f32[32]" = torch.ops.aten.rsqrt.default(add);  add = None
        view: "f32[32, 1]" = torch.ops.aten.view.default(rsqrt, [-1, 1]);  rsqrt = None
        view_1: "f32[32, 1]" = torch.ops.aten.view.default(primals_6, [32, 1]);  primals_6 = None
        mul: "f32[32, 1]" = torch.ops.aten.mul.Tensor(view_1, view);  view_1 = None
        view_2: "f32[32]" = torch.ops.aten.view.default(mul, [32])
        mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(view_3, view_2);  view_2 = None
        sub: "f32[32]" = torch.ops.aten.sub.Tensor(primals_5, primals_2);  primals_5 = primals_2 = None
        mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(view_3, sub);  sub = None
        view_4: "f32[32, 1]" = torch.ops.aten.view.default(mul_4, [32, 1]);  mul_4 = None
        mul_5: "f32[32, 3]" = torch.ops.aten.mul.Tensor(mm, primals_4);  primals_4 = None
        mul_6: "f32[32, 3]" = torch.ops.aten.mul.Tensor(mm, mul);  mm = mul = None
        sum_2: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [1], True);  mul_5 = None
        add_2: "f32[32, 1]" = torch.ops.aten.add.Tensor(view_4, sum_2);  view_4 = sum_2 = None
        mul_7: "f32[32, 1]" = torch.ops.aten.mul.Tensor(add_2, view);  add_2 = view = None
        view_5: "f32[32]" = torch.ops.aten.view.default(mul_7, [32]);  mul_7 = None
        return (None, None, None, mul_6, mul_3, view_5, view_3)
