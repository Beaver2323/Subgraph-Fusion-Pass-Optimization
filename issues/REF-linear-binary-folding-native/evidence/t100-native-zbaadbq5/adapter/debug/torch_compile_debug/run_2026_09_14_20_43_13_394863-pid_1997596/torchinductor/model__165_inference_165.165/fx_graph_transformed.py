class <lambda>(torch.nn.Module):
    def forward(self, arg3_1: "f32[4, 3]"):
        # No stacktrace found for following nodes
        arg0_1: "f32[32]" = self._frozen_param0
        arg2_1: "f32[4, 1]" = self._frozen_param2

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:252 in forward, code: x = self.linear(x)
        permute: "f32[3, 32]" = self._frozen_param3
        mm_default: "f32[4, 32]" = torch.ops.aten.mm.default(arg3_1, permute);  arg3_1 = permute = None
        add_tensor: "f32[4, 32]" = torch.ops.aten.add.Tensor(arg0_1, mm_default);  arg0_1 = mm_default = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:256 in forward, code: return self.op(x, self.tensor)
        mul: "f32[4, 32]" = torch.ops.aten.mul.Tensor(add_tensor, arg2_1);  add_tensor = arg2_1 = None
        return (mul,)
