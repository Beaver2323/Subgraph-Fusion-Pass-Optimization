class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 64]", arg1_1: "bf16[64, 64]", arg2_1: "bf16[64, 64]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/issues/REF-addcdiv-fma-codegen-native/verify_lowp_three_arm.py:125 in addcdiv_fn, code: return torch.addcdiv(s, t1, t2, value=2.0)
        div: "bf16[64, 64]" = torch.ops.aten.div.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "bf16[64, 64]" = torch.ops.aten.mul.Tensor(div, 2.0);  div = None
        add: "bf16[64, 64]" = torch.ops.aten.add.Tensor(arg0_1, mul);  arg0_1 = mul = None
        return (add,)
