class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[7, 8, 96]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:161 in fn, code: parts = torch.ops.aten.split_with_sizes.default(x, [1] * 7)
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(arg0_1, [1, 1, 1, 1, 1, 1, 1], dim = 0);  arg0_1 = None
        getitem: "f32[1, 8, 96]" = split_with_sizes_default[0]
        getitem_1: "f32[1, 8, 96]" = split_with_sizes_default[1]
        getitem_2: "f32[1, 8, 96]" = split_with_sizes_default[2]
        getitem_3: "f32[1, 8, 96]" = split_with_sizes_default[3]
        getitem_4: "f32[1, 8, 96]" = split_with_sizes_default[4]
        getitem_5: "f32[1, 8, 96]" = split_with_sizes_default[5]
        getitem_6: "f32[1, 8, 96]" = split_with_sizes_default[6];  split_with_sizes_default = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:162 in <listcomp>, code: views = [torch.ops.aten.view.default(item, [8, 96]) for item in parts]
        view: "f32[8, 96]" = torch.ops.aten.reshape.default(getitem, [8, 96]);  getitem = None
        view_1: "f32[8, 96]" = torch.ops.aten.reshape.default(getitem_1, [8, 96]);  getitem_1 = None
        view_2: "f32[8, 96]" = torch.ops.aten.reshape.default(getitem_2, [8, 96]);  getitem_2 = None
        view_3: "f32[8, 96]" = torch.ops.aten.reshape.default(getitem_3, [8, 96]);  getitem_3 = None
        view_4: "f32[8, 96]" = torch.ops.aten.reshape.default(getitem_4, [8, 96]);  getitem_4 = None
        view_5: "f32[8, 96]" = torch.ops.aten.reshape.default(getitem_5, [8, 96]);  getitem_5 = None
        view_6: "f32[8, 96]" = torch.ops.aten.reshape.default(getitem_6, [8, 96]);  getitem_6 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:164 in fn, code: merged = torch.ops.aten.cat.default(views, 1)
        cat_default_2: "f32[8, 672]" = torch.ops.aten.cat.default([view, view_1, view_2, view_3, view_4, view_5, view_6], dim = 1);  view_1 = view_2 = view_3 = view_4 = view_5 = view_6 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:165 in fn, code: return torch.ops.aten.cat.default([clone, torch.ops.aten.cat.default([clone, merged], 1)], 1)
        cat_default_1: "f32[8, 768]" = torch.ops.aten.cat.default([view, cat_default_2], dim = 1);  cat_default_2 = None
        cat_default: "f32[8, 864]" = torch.ops.aten.cat.default([view, cat_default_1], dim = 1);  view = cat_default_1 = None
        return (cat_default,)
