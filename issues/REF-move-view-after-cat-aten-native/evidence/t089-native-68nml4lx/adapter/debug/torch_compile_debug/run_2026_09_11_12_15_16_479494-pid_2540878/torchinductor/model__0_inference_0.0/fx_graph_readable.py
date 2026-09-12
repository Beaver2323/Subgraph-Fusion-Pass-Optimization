class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[7, 8, 96]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:159 in forward, code: split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(
        split_with_sizes = torch.ops.aten.split_with_sizes.default(arg0_1, [1, 1, 1, 1, 1, 1, 1]);  arg0_1 = None
        getitem: "f32[1, 8, 96]" = split_with_sizes[0]
        getitem_1: "f32[1, 8, 96]" = split_with_sizes[1]
        getitem_2: "f32[1, 8, 96]" = split_with_sizes[2]
        getitem_3: "f32[1, 8, 96]" = split_with_sizes[3]
        getitem_4: "f32[1, 8, 96]" = split_with_sizes[4]
        getitem_5: "f32[1, 8, 96]" = split_with_sizes[5]
        getitem_6: "f32[1, 8, 96]" = split_with_sizes[6];  split_with_sizes = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:169 in forward, code: view_1 = torch.ops.aten.view.default(getitem_71, [8, 96])
        view: "f32[8, 96]" = torch.ops.aten.view.default(getitem, [8, 96]);  getitem = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:170 in forward, code: view_2 = torch.ops.aten.view.default(getitem_72, [8, 96])
        view_1: "f32[8, 96]" = torch.ops.aten.view.default(getitem_1, [8, 96]);  getitem_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:171 in forward, code: view_3 = torch.ops.aten.view.default(getitem_73, [8, 96])
        view_2: "f32[8, 96]" = torch.ops.aten.view.default(getitem_2, [8, 96]);  getitem_2 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:172 in forward, code: view_4 = torch.ops.aten.view.default(getitem_74, [8, 96])
        view_3: "f32[8, 96]" = torch.ops.aten.view.default(getitem_3, [8, 96]);  getitem_3 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:173 in forward, code: view_5 = torch.ops.aten.view.default(getitem_75, [8, 96])
        view_4: "f32[8, 96]" = torch.ops.aten.view.default(getitem_4, [8, 96]);  getitem_4 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:174 in forward, code: view_6 = torch.ops.aten.view.default(getitem_76, [8, 96])
        view_5: "f32[8, 96]" = torch.ops.aten.view.default(getitem_5, [8, 96]);  getitem_5 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:175 in forward, code: view_7 = torch.ops.aten.view.default(getitem_77, [8, 96])
        view_6: "f32[8, 96]" = torch.ops.aten.view.default(getitem_6, [8, 96]);  getitem_6 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:178 in forward, code: cat = torch.ops.aten.cat.default(
        cat: "f32[8, 672]" = torch.ops.aten.cat.default([view, view_1, view_2, view_3, view_4, view_5, view_6], 1);  view_1 = view_2 = view_3 = view_4 = view_5 = view_6 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:190 in forward, code: cat_1 = torch.ops.aten.cat.default([clone, cat], 1)
        cat_1: "f32[8, 768]" = torch.ops.aten.cat.default([view, cat], 1);  cat = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:191 in forward, code: return torch.cat([clone, cat_1], 1)
        cat_2: "f32[8, 864]" = torch.ops.aten.cat.default([view, cat_1], 1);  view = cat_1 = None
        return (cat_2,)
