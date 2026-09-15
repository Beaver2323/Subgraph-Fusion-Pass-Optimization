class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 7, 4]", arg1_1: "f32[3, 7, 5]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/attention_select_slice_boundary.py:88 in fn, code: return source.select(-1, lane).unsqueeze(-1) + addend
        select: "f32[3, 7]" = torch.ops.aten.select.int(arg0_1, -1, 3);  arg0_1 = None
        unsqueeze: "f32[3, 7, 1]" = torch.ops.aten.unsqueeze.default(select, -1);  select = None
        add: "f32[3, 7, 5]" = torch.ops.aten.add.Tensor(unsqueeze, arg1_1);  unsqueeze = arg1_1 = None
        return (add,)
