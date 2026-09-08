class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[64, 64]", arg1_1: "f16[64, 64]", arg2_1: "f16[64, 64]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/issues/REF-addcdiv-fma-codegen-native/verify_lowp_three_arm.py:128 in decomposed_fn, code: return s + (t1 / t2) * 2.0
        div: "f16[64, 64]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mul: "f16[64, 64]" = torch.ops.aten.mul.Tensor(div, 2.0);  div = None
        add: "f16[64, 64]" = torch.ops.aten.add.Tensor(arg2_1, mul);  arg2_1 = mul = None
        return (add,)
