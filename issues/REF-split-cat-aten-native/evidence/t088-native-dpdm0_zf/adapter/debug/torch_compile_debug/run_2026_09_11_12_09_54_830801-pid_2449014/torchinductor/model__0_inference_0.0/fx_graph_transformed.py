class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024, 128]", arg1_1: "f32[1024, 128]", arg2_1: "f32[1024, 32]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:25 in forward, code: cat = torch.ops.aten.cat.default([x, y], 1)
        cat_default_3: "f32[1024, 256]" = torch.ops.aten.cat.default([arg0_1, arg1_1], dim = 1);  arg0_1 = arg1_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:26 in forward, code: split = torch.ops.aten.split.Tensor(cat, 32, 1)
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(cat_default_3, [32, 32, 32, 32, 32, 32, 32, 32], dim = 1)
        getitem: "f32[1024, 32]" = split_with_sizes_default[0];  split_with_sizes_default = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:35 in forward, code: cat_1 = torch.ops.aten.cat.default(
        cat_default_2: "f32[1024, 256]" = torch.ops.aten.cat.default([cat_default_3], dim = 1);  cat_default_3 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:48 in forward, code: cat_2 = torch.ops.aten.cat.default([getitem, z], 1)
        cat_default_1: "f32[1024, 64]" = torch.ops.aten.cat.default([getitem, arg2_1], dim = 1);  getitem = arg2_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:49 in forward, code: return torch.ops.aten.cat.default([cat_1, cat_2], 1)
        cat_default: "f32[1024, 320]" = torch.ops.aten.cat.default([cat_default_2, cat_default_1], dim = 1);  cat_default_2 = cat_default_1 = None
        return (cat_default,)
