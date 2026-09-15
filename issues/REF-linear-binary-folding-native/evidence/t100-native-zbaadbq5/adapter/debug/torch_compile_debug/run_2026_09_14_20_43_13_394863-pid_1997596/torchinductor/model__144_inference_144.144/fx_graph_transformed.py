class <lambda>(torch.nn.Module):
    def forward(self, arg2_1: "f32[2, 4, 3]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:252 in forward, code: x = self.linear(x)
        mul_tensor: "f32[3, 32]" = self._frozen_param3
        view: "f32[8, 3]" = torch.ops.aten.reshape.default(arg2_1, [8, 3]);  arg2_1 = None
        mm_default: "f32[8, 32]" = torch.ops.aten.mm.default(view, mul_tensor);  view = mul_tensor = None
        view_1: "f32[2, 4, 32]" = torch.ops.aten.reshape.default(mm_default, [2, 4, 32]);  mm_default = None
        return (view_1,)
