class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024, 128]", arg1_1: "f32[1024, 128]", arg2_1: "f32[1024, 32]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:143 in fn, code: parts = torch.ops.aten.split.Tensor(torch.ops.aten.cat.default([x, y], 1), 32, 1)
        cat_default_3: "f32[1024, 256]" = torch.ops.aten.cat.default([arg0_1, arg1_1], dim = 1);  arg0_1 = arg1_1 = None
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(cat_default_3, [32, 32, 32, 32, 32, 32, 32, 32], dim = 1)
        getitem: "f32[1024, 32]" = split_with_sizes_default[0];  split_with_sizes_default = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:144 in fn, code: merged = torch.ops.aten.cat.default(list(parts[:8]), 1)
        cat_default_2: "f32[1024, 256]" = torch.ops.aten.cat.default([cat_default_3], dim = 1);  cat_default_3 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:145 in fn, code: side = torch.ops.aten.cat.default([parts[0], z], 1)
        cat_default_1: "f32[1024, 64]" = torch.ops.aten.cat.default([getitem, arg2_1], dim = 1);  getitem = arg2_1 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:146 in fn, code: return torch.ops.aten.cat.default([merged, side], 1)
        cat_default: "f32[1024, 320]" = torch.ops.aten.cat.default([cat_default_2, cat_default_1], dim = 1);  cat_default_2 = cat_default_1 = None
        return (cat_default,)
