class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 16, 8]", arg1_1: "f32[1, 8, 32]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/verify_t079_bmm_gate.py:100 in fn, code: return torch.bmm(a, b)
        bmm: "f32[1, 16, 32]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        return (bmm,)
