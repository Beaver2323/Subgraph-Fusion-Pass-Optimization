class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 4]", arg1_1: "f32[4, 4]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_passes.py:1557 in fn, code: return torch.stack([x, y], axis=1)
        cat: "f32[4, 8]" = torch.ops.aten.cat.default([arg0_1, arg1_1], 1);  arg0_1 = arg1_1 = None
        view: "f32[4, 2, 4]" = torch.ops.aten.view.default(cat, [4, 2, 4]);  cat = None
        return (view,)
