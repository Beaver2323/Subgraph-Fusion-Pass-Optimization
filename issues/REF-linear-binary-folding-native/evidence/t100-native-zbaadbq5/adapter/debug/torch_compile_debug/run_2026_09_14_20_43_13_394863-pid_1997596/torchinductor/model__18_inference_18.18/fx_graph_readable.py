class <lambda>(torch.nn.Module):
    def forward(self, arg2_1: "f32[4, 3]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:252 in forward, code: x = self.linear(x)
        mul_tensor: "f32[3, 32]" = self._frozen_param3

        # No stacktrace found for following nodes
        mul_tensor_1: "f32[32]" = self._frozen_param4

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:252 in forward, code: x = self.linear(x)
        addmm_default: "f32[4, 32]" = torch.ops.aten.addmm.default(mul_tensor_1, arg2_1, mul_tensor);  mul_tensor_1 = arg2_1 = mul_tensor = None
        return (addmm_default,)
