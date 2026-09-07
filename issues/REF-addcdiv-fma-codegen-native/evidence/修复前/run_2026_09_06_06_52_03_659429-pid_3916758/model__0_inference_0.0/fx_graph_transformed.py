class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 64]", arg1_1: "f32[64, 64]", arg2_1: "f32[64, 64]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_torchinductor.py:19053 in fn, code: return torch.addcdiv(s, t1, t2, value=2.0)
        div: "f32[64, 64]" = torch.ops.aten.div.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "f32[64, 64]" = torch.ops.aten.mul.Tensor(div, 2.0);  div = None
        add: "f32[64, 64]" = torch.ops.aten.add.Tensor(arg0_1, mul);  arg0_1 = mul = None
        return (add,)
