class <lambda>(torch.nn.Module):
    def forward(self, arg1_1: "f32[2, 4, 3]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:252 in forward, code: x = self.linear(x)
        permute: "f32[3, 32]" = self._frozen_param1
        view: "f32[8, 3]" = torch.ops.aten.reshape.default(arg1_1, [8, 3]);  arg1_1 = None

        # No stacktrace found for following nodes
        sub_tensor: "f32[32]" = self._frozen_param2

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:252 in forward, code: x = self.linear(x)
        addmm_default: "f32[8, 32]" = torch.ops.aten.addmm.default(sub_tensor, view, permute);  sub_tensor = view = permute = None
        view_1: "f32[2, 4, 32]" = torch.ops.aten.reshape.default(addmm_default, [2, 4, 32]);  addmm_default = None
        return (view_1,)
