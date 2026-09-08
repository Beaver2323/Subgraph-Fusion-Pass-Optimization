class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 64]", arg1_1: "bf16[64, 64]", arg2_1: "bf16[64, 64]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t078_performance_worker.py:199 in fn, code: return torch.addcdiv(inp, tensor1, tensor2, value=value)
        addcdiv_default: "bf16[64, 64]" = torch.ops.aten.addcdiv.default(arg0_1, arg1_1, arg2_1, value = 2.0);  arg0_1 = arg1_1 = arg2_1 = None
        return (addcdiv_default,)
