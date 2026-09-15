class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s21)", arg1_1: "Sym(s67)", arg2_1: "f32[1, s21, s67]", arg3_1: "Sym(s26)", arg4_1: "Sym(s3)", arg5_1: "Sym(s32)", arg6_1: "f32[s3, s21, s32]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/attention_select_slice_boundary.py:88 in fn, code: return source.select(-1, lane).unsqueeze(-1) + addend
        select: "f32[1, s21]" = torch.ops.aten.select.int(arg2_1, -1, arg3_1);  arg2_1 = arg3_1 = None
        unsqueeze: "f32[1, s21, 1]" = torch.ops.aten.unsqueeze.default(select, -1);  select = None
        add_5: "f32[s3, s21, s32]" = torch.ops.aten.add.Tensor(unsqueeze, arg6_1);  unsqueeze = arg6_1 = None
        return (add_5,)
