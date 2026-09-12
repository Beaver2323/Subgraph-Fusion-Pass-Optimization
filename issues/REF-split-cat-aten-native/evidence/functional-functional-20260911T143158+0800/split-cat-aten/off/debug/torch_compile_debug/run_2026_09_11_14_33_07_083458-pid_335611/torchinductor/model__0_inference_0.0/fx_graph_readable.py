class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024, 128]", arg1_1: "f32[1024, 128]", arg2_1: "f32[1024, 32]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:143 in fn, code: parts = torch.ops.aten.split.Tensor(torch.ops.aten.cat.default([x, y], 1), 32, 1)
        cat: "f32[1024, 256]" = torch.ops.aten.cat.default([arg0_1, arg1_1], 1);  arg0_1 = arg1_1 = None
        split = torch.ops.aten.split.Tensor(cat, 32, 1);  cat = None
        getitem: "f32[1024, 32]" = split[0]
        getitem_1: "f32[1024, 32]" = split[1]
        getitem_2: "f32[1024, 32]" = split[2]
        getitem_3: "f32[1024, 32]" = split[3]
        getitem_4: "f32[1024, 32]" = split[4]
        getitem_5: "f32[1024, 32]" = split[5]
        getitem_6: "f32[1024, 32]" = split[6]
        getitem_7: "f32[1024, 32]" = split[7];  split = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:144 in fn, code: merged = torch.ops.aten.cat.default(list(parts[:8]), 1)
        cat_1: "f32[1024, 256]" = torch.ops.aten.cat.default([getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7], 1);  getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:145 in fn, code: side = torch.ops.aten.cat.default([parts[0], z], 1)
        cat_2: "f32[1024, 64]" = torch.ops.aten.cat.default([getitem, arg2_1], 1);  getitem = arg2_1 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:146 in fn, code: return torch.ops.aten.cat.default([merged, side], 1)
        cat_3: "f32[1024, 320]" = torch.ops.aten.cat.default([cat_1, cat_2], 1);  cat_1 = cat_2 = None
        return (cat_3,)
