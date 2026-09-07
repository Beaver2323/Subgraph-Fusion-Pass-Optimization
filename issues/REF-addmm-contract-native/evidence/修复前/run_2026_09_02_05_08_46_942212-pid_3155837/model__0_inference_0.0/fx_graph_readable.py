class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 16]", arg1_1: "f32[16, 16]", arg2_1: "f32[16, 16]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_pattern_matcher.py:577 in fn, code: return torch.add(a, torch.mm(b, c)), torch.mm(b, c) + a
        mm: "f32[16, 16]" = torch.ops.aten.mm.default(arg0_1, arg1_1)
        add: "f32[16, 16]" = torch.ops.aten.add.Tensor(arg2_1, mm);  mm = None
        mm_1: "f32[16, 16]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        add_1: "f32[16, 16]" = torch.ops.aten.add.Tensor(mm_1, arg2_1);  mm_1 = arg2_1 = None
        return (add, add_1)
