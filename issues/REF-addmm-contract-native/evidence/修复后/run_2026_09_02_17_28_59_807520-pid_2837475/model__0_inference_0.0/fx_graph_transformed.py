class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 16]", arg1_1: "f32[16, 16]", arg2_1: "f32[16, 16]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_pattern_matcher.py:577 in fn, code: return torch.add(a, torch.mm(b, c)), torch.mm(b, c) + a
        addmm_default_1: "f32[16, 16]" = torch.ops.aten.addmm.default(arg2_1, arg0_1, arg1_1)
        addmm_default: "f32[16, 16]" = torch.ops.aten.addmm.default(arg2_1, arg0_1, arg1_1);  arg2_1 = arg0_1 = arg1_1 = None
        return (addmm_default_1, addmm_default)
