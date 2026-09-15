class <lambda>(torch.nn.Module):
    def forward(self, arg1_1: "f32[4, 3]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:252 in forward, code: x = self.linear(x)
        permute: "f32[3, 32]" = self._frozen_param1

        # No stacktrace found for following nodes
        add_tensor: "f32[32]" = self._frozen_param2

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:252 in forward, code: x = self.linear(x)
        addmm_default: "f32[4, 32]" = torch.ops.aten.addmm.default(add_tensor, arg1_1, permute);  add_tensor = arg1_1 = permute = None
        return (addmm_default,)
