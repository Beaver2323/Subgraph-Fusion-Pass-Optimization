class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 64]", arg1_1: "bf16[64, 64]", arg2_1: "bf16[64, 64]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t078_performance_worker.py:199 in fn, code: return torch.addcdiv(inp, tensor1, tensor2, value=value)
        div: "bf16[64, 64]" = torch.ops.aten.div.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "bf16[64, 64]" = torch.ops.aten.mul.Tensor(div, 2.0);  div = None
        add: "bf16[64, 64]" = torch.ops.aten.add.Tensor(arg0_1, mul);  arg0_1 = mul = None
        return (add,)
