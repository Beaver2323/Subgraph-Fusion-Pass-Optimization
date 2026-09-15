class <lambda>(torch.nn.Module):
    def forward(self, arg3_1: "f32[2, 4, 3]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:252 in forward, code: x = self.linear(x)
        view: "f32[8, 3]" = torch.ops.aten.reshape.default(arg3_1, [8, 3]);  arg3_1 = None
        mul_tensor: "f32[3, 32]" = self._frozen_param4

        # No stacktrace found for following nodes
        mul_tensor_1: "f32[32]" = self._frozen_param5

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:252 in forward, code: x = self.linear(x)
        addmm_default: "f32[8, 32]" = torch.ops.aten.addmm.default(mul_tensor_1, view, mul_tensor);  mul_tensor_1 = view = mul_tensor = None
        view_1: "f32[2, 4, 32]" = torch.ops.aten.reshape.default(addmm_default, [2, 4, 32]);  addmm_default = None
        return (view_1,)
