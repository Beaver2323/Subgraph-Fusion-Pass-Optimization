class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[64, 64]", arg1_1: "f16[64, 64]", arg2_1: "f16[64, 64]"):
        # File: /data/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t078_addcdiv_dtype_reference.py:64 in fn, code: return torch.addcdiv(s, t1, t2, value=args.value)
        div: "f16[64, 64]" = torch.ops.aten.div.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        add: "f16[64, 64]" = torch.ops.aten.add.Tensor(arg0_1, div);  arg0_1 = div = None
        return (add,)
