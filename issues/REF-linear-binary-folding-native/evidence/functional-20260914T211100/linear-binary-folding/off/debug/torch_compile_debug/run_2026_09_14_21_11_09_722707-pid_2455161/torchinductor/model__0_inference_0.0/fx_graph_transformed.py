class <lambda>(torch.nn.Module):
    def forward(self, arg3_1: "f32[4, 3]"):
        # No stacktrace found for following nodes
        arg1_1: "f32[32]" = self._frozen_param1
        arg2_1: "f32[32]" = self._frozen_param2

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t091_t100_performance_worker.py:295 in forward, code: return self.linear(x) + self.other
        permute: "f32[3, 32]" = self._frozen_param3
        mm_default: "f32[4, 32]" = torch.ops.aten.mm.default(arg3_1, permute);  arg3_1 = permute = None
        add_tensor: "f32[4, 32]" = torch.ops.aten.add.Tensor(arg1_1, mm_default);  arg1_1 = mm_default = None
        add: "f32[4, 32]" = torch.ops.aten.add.Tensor(add_tensor, arg2_1);  add_tensor = arg2_1 = None
        return (add,)
