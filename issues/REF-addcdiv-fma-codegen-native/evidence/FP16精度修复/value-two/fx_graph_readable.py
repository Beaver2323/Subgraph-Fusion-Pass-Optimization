class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[64, 64]", arg1_1: "f16[64, 64]", arg2_1: "f16[64, 64]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/issues/REF-addcdiv-fma-codegen-native/verify_lowp_three_arm.py:137 in addcdiv_fn, code: return torch.addcdiv(s, t1, t2, value=args.value)
        div: "f16[64, 64]" = torch.ops.aten.div.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "f16[64, 64]" = torch.ops.aten.mul.Tensor(div, 2.0);  div = None
        add: "f16[64, 64]" = torch.ops.aten.add.Tensor(arg0_1, mul);  arg0_1 = mul = None
        return (add,)
