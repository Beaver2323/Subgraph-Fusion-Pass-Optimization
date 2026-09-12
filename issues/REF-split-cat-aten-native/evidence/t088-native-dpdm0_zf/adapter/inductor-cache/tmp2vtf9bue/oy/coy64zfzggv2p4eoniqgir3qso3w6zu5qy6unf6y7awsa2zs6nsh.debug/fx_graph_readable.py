class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024, 2016]", arg1_1: "f32[1024, 384]", arg2_1: "f32[1024, 96]", arg3_1: "f32[1024, 96]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:75 in forward, code: split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(
        split_with_sizes = torch.ops.aten.split_with_sizes.default(arg0_1, [96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96], 1);  arg0_1 = None
        getitem: "f32[1024, 96]" = split_with_sizes[0]
        getitem_1: "f32[1024, 96]" = split_with_sizes[1]
        getitem_2: "f32[1024, 96]" = split_with_sizes[2]
        getitem_3: "f32[1024, 96]" = split_with_sizes[3]
        getitem_4: "f32[1024, 96]" = split_with_sizes[4]
        getitem_5: "f32[1024, 96]" = split_with_sizes[5]
        getitem_10: "f32[1024, 96]" = split_with_sizes[10]
        getitem_11: "f32[1024, 96]" = split_with_sizes[11]
        getitem_12: "f32[1024, 96]" = split_with_sizes[12]
        getitem_13: "f32[1024, 96]" = split_with_sizes[13]
        getitem_14: "f32[1024, 96]" = split_with_sizes[14]
        getitem_15: "f32[1024, 96]" = split_with_sizes[15]
        getitem_16: "f32[1024, 96]" = split_with_sizes[16]
        getitem_17: "f32[1024, 96]" = split_with_sizes[17];  split_with_sizes = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:102 in forward, code: split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(arg1_1, [96, 96, 96, 96], 1);  arg1_1 = None
        getitem_21: "f32[1024, 96]" = split_with_sizes_1[0]
        getitem_22: "f32[1024, 96]" = split_with_sizes_1[1]
        getitem_23: "f32[1024, 96]" = split_with_sizes_1[2]
        getitem_24: "f32[1024, 96]" = split_with_sizes_1[3];  split_with_sizes_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:124 in forward, code: cat = torch.ops.aten.cat.default(
        cat: "f32[1024, 2112]" = torch.ops.aten.cat.default([arg3_1, getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_15, getitem_16, getitem_17, arg2_1, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, arg2_1, getitem_21, getitem_22, getitem_23, getitem_24, arg3_1], 1);  arg3_1 = getitem = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_15 = getitem_16 = getitem_17 = arg2_1 = getitem_10 = getitem_11 = getitem_12 = getitem_13 = getitem_14 = getitem_21 = getitem_22 = getitem_23 = getitem_24 = None
        return (cat,)
