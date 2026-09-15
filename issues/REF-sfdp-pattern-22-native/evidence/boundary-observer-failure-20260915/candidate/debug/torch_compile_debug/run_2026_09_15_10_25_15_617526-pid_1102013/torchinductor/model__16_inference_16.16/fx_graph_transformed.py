class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 7, 8]", arg1_1: "bf16[3, 7, 5]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/attention_select_slice_boundary.py:83 in fn, code: return source.select(-1, lane).unsqueeze(-1) + addend
        select: "bf16[1, 7]" = torch.ops.aten.select.int(arg0_1, -1, 0);  arg0_1 = None
        unsqueeze: "bf16[1, 7, 1]" = torch.ops.aten.unsqueeze.default(select, -1);  select = None
        add: "bf16[3, 7, 5]" = torch.ops.aten.add.Tensor(unsqueeze, arg1_1);  unsqueeze = arg1_1 = None
        return (add,)
