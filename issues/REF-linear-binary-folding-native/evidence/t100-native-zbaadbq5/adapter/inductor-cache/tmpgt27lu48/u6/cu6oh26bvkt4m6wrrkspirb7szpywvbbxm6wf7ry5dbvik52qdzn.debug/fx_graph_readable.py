class <lambda>(torch.nn.Module):
    def forward(self, arg1_1: "f32[2, 4, 3]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_binary_folding.py:252 in forward, code: x = self.linear(x)
        view: "f32[8, 3]" = torch.ops.aten.reshape.default(arg1_1, [8, 3]);  arg1_1 = None
        div_tensor: "f32[3, 32]" = self._frozen_param2
        mm_default: "f32[8, 32]" = torch.ops.aten.mm.default(view, div_tensor);  view = div_tensor = None
        view_1: "f32[2, 4, 32]" = torch.ops.aten.reshape.default(mm_default, [2, 4, 32]);  mm_default = None
        return (view_1,)
