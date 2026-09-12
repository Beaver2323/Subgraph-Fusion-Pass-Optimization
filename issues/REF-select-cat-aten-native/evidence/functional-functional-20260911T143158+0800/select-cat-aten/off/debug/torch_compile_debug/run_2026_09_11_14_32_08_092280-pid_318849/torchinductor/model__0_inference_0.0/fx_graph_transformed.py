class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024, 6, 128]", arg1_1: "f32[1024, 6, 128]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:150 in <listcomp>, code: xs = [torch.ops.aten.select.int(x, 1, i) for i in range(6)]
        select: "f32[1024, 128]" = torch.ops.aten.select.int(arg0_1, 1, 0)
        select_1: "f32[1024, 128]" = torch.ops.aten.select.int(arg0_1, 1, 1)
        select_2: "f32[1024, 128]" = torch.ops.aten.select.int(arg0_1, 1, 2)
        select_3: "f32[1024, 128]" = torch.ops.aten.select.int(arg0_1, 1, 3)
        select_4: "f32[1024, 128]" = torch.ops.aten.select.int(arg0_1, 1, 4)
        select_5: "f32[1024, 128]" = torch.ops.aten.select.int(arg0_1, 1, 5);  arg0_1 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:153 in fn, code: torch.ops.aten.cat.default(xs, 1),
        cat_default_3: "f32[1024, 768]" = torch.ops.aten.cat.default([select, select_1, select_2, select_3, select_4, select_5], dim = 1);  select_5 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:154 in fn, code: torch.ops.aten.cat.default(xs[:5], 1),
        cat_default_2: "f32[1024, 640]" = torch.ops.aten.cat.default([select, select_1, select_2, select_3, select_4], dim = 1);  select_1 = select_3 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:155 in fn, code: torch.ops.aten.cat.default(xs[::2], 1),
        cat_default_1: "f32[1024, 384]" = torch.ops.aten.cat.default([select, select_2, select_4], dim = 1);  select = select_2 = select_4 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:151 in <listcomp>, code: ys = [torch.ops.aten.select.int(y, 1, i) for i in range(5)]
        select_6: "f32[1024, 128]" = torch.ops.aten.select.int(arg1_1, 1, 0)
        select_7: "f32[1024, 128]" = torch.ops.aten.select.int(arg1_1, 1, 1)
        select_8: "f32[1024, 128]" = torch.ops.aten.select.int(arg1_1, 1, 2)
        select_9: "f32[1024, 128]" = torch.ops.aten.select.int(arg1_1, 1, 3)
        select_10: "f32[1024, 128]" = torch.ops.aten.select.int(arg1_1, 1, 4);  arg1_1 = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:156 in fn, code: torch.ops.aten.cat.default(ys, 1),
        cat_default: "f32[1024, 640]" = torch.ops.aten.cat.default([select_6, select_7, select_8, select_9, select_10], dim = 1);  select_6 = select_7 = select_8 = select_9 = select_10 = None
        return (cat_default_3, cat_default_2, cat_default_1, cat_default)
