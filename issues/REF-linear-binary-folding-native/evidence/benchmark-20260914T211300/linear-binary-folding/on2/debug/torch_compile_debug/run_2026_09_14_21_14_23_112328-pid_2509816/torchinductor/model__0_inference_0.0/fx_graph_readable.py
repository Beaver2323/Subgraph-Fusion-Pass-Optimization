class <lambda>(torch.nn.Module):
    def forward(self, arg3_1: "f32[4, 3]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t091_t100_performance_worker.py:295 in forward, code: return self.linear(x) + self.other
        permute: "f32[3, 32]" = self._frozen_param3

        # No stacktrace found for following nodes
        add_tensor: "f32[32]" = self._frozen_param4

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t091_t100_performance_worker.py:295 in forward, code: return self.linear(x) + self.other
        addmm_default: "f32[4, 32]" = torch.ops.aten.addmm.default(add_tensor, arg3_1, permute);  add_tensor = arg3_1 = permute = None
        return (addmm_default,)
