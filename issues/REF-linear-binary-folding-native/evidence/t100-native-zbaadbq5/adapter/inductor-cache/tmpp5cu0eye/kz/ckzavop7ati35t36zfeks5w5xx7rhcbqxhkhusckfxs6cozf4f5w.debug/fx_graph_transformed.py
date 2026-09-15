class <lambda>(torch.nn.Module):
    def forward(self, arg2_1: "f32[4, 3]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:252 in forward, code: x = self.linear(x)
        mul_tensor: "f32[3, 32]" = self._frozen_param3
        mm_default: "f32[4, 32]" = torch.ops.aten.mm.default(arg2_1, mul_tensor);  arg2_1 = mul_tensor = None
        return (mm_default,)
