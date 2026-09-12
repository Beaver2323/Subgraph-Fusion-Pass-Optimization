class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024, 128]", arg1_1: "f32[1024, 128]", arg2_1: "f32[1024, 32]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:57 in forward, code: cat = torch.ops.aten.cat.default([x, y], 1)
        cat_default_2: "f32[1024, 256]" = torch.ops.aten.cat.default([arg0_1, arg1_1], dim = 1);  arg0_1 = arg1_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:58 in forward, code: split = torch.ops.aten.split.Tensor(cat, 32, 1)
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(cat_default_2, [32, 32, 32, 32, 32, 32, 32, 32], dim = 1);  cat_default_2 = None
        getitem: "f32[1024, 32]" = split_with_sizes_default[0];  split_with_sizes_default = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:60 in forward, code: cat_1 = torch.ops.aten.cat.default(
        clone: "f32[1024, 32]" = torch.ops.aten.clone.default(getitem)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:64 in forward, code: cat_2 = torch.ops.aten.cat.default([getitem, z], 1)
        cat_default_1: "f32[1024, 64]" = torch.ops.aten.cat.default([getitem, arg2_1], dim = 1);  getitem = arg2_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:65 in forward, code: return torch.ops.aten.cat.default([cat_1, cat_2], 1)
        cat_default: "f32[1024, 96]" = torch.ops.aten.cat.default([clone, cat_default_1], dim = 1);  clone = cat_default_1 = None
        return (cat_default,)
