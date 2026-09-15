class <lambda>(torch.nn.Module):
    def forward(self, arg3_1: "f32[4, 3]"):
        # No stacktrace found for following nodes
        arg1_1: "f32[32]" = self._frozen_param1
        arg2_1: "f32[32]" = self._frozen_param2

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t091_t100_performance_worker.py:295 in forward, code: return self.linear(x) + self.other
        permute: "f32[3, 32]" = self._frozen_param3
        addmm: "f32[4, 32]" = torch.ops.aten.addmm.default(arg1_1, arg3_1, permute);  arg1_1 = arg3_1 = permute = None
        add: "f32[4, 32]" = torch.ops.aten.add.Tensor(addmm, arg2_1);  addmm = arg2_1 = None
        return (add,)
