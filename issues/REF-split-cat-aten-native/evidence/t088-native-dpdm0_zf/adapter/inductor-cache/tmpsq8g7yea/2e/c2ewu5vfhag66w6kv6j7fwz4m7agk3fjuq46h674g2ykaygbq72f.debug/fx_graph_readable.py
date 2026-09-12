class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024, 128]", arg1_1: "f32[1024, 128]", arg2_1: "f32[1024, 32]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:25 in forward, code: cat = torch.ops.aten.cat.default([x, y], 1)
        cat: "f32[1024, 256]" = torch.ops.aten.cat.default([arg0_1, arg1_1], 1);  arg0_1 = arg1_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:26 in forward, code: split = torch.ops.aten.split.Tensor(cat, 32, 1)
        split = torch.ops.aten.split.Tensor(cat, 32, 1);  cat = None
        getitem: "f32[1024, 32]" = split[0]
        getitem_1: "f32[1024, 32]" = split[1]
        getitem_2: "f32[1024, 32]" = split[2]
        getitem_3: "f32[1024, 32]" = split[3]
        getitem_4: "f32[1024, 32]" = split[4]
        getitem_5: "f32[1024, 32]" = split[5]
        getitem_6: "f32[1024, 32]" = split[6]
        getitem_7: "f32[1024, 32]" = split[7];  split = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:48 in forward, code: cat_2 = torch.ops.aten.cat.default([getitem, z], 1)
        cat_1: "f32[1024, 64]" = torch.ops.aten.cat.default([getitem, arg2_1], 1);  arg2_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:35 in forward, code: cat_1 = torch.ops.aten.cat.default(
        cat_2: "f32[1024, 256]" = torch.ops.aten.cat.default([getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7], 1);  getitem = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:49 in forward, code: return torch.ops.aten.cat.default([cat_1, cat_2], 1)
        cat_3: "f32[1024, 320]" = torch.ops.aten.cat.default([cat_2, cat_1], 1);  cat_2 = cat_1 = None
        return (cat_3,)
