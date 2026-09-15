class <lambda>(torch.nn.Module):
    def forward(self, arg2_1: "f32[4, 3]"):
        # No stacktrace found for following nodes
        arg1_1: "f32[4, 32]" = self._frozen_param1

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:252 in forward, code: x = self.linear(x)
        permute: "f32[3, 32]" = self._frozen_param2
        mm: "f32[4, 32]" = torch.ops.aten.mm.default(arg2_1, permute);  arg2_1 = permute = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:256 in forward, code: return self.op(x, self.tensor)
        div: "f32[4, 32]" = torch.ops.aten.div.Tensor(mm, arg1_1);  mm = arg1_1 = None
        return (div,)
