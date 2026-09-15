class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 3]", primals_2: "f32[32]", primals_3: "f32[32]", primals_4: "f32[32, 3]", primals_5: "f32[32]", primals_6: "f32[32]"):
        # No stacktrace found for following nodes
        full_default: "f32[32]" = torch.ops.aten.full.default([32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        add: "f32[32]" = torch.ops.aten.add.Tensor(primals_3, 1e-05)
        rsqrt: "f32[32]" = torch.ops.aten.rsqrt.default(add);  add = None
        view: "f32[32, 1]" = torch.ops.aten.reshape.default(rsqrt, [-1, 1]);  rsqrt = None
        view_1: "f32[32, 1]" = torch.ops.aten.reshape.default(primals_5, [32, 1])
        mul: "f32[32, 1]" = torch.ops.aten.mul.Tensor(view_1, view);  view_1 = view = None
        mul_1: "f32[32, 3]" = torch.ops.aten.mul.Tensor(primals_4, mul)
        view_2: "f32[32]" = torch.ops.aten.reshape.default(mul, [32]);  mul = None
        sub: "f32[32]" = torch.ops.aten.sub.Tensor(full_default, primals_2);  full_default = None
        mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(view_2, sub);  view_2 = sub = None
        add_1: "f32[32]" = torch.ops.aten.add.Tensor(primals_6, mul_2);  primals_6 = mul_2 = None
        permute: "f32[3, 32]" = torch.ops.aten.permute.default(mul_1, [1, 0]);  mul_1 = None
        addmm: "f32[4, 32]" = torch.ops.aten.addmm.default(add_1, primals_1, permute);  add_1 = permute = None
        return (addmm, primals_1, primals_2, primals_3, primals_4, primals_5)
