class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024, 2016]", arg1_1: "f32[1024, 384]", arg2_1: "f32[1024, 96]", arg3_1: "f32[1024, 96]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:75 in forward, code: split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(
        split_with_sizes_default_1 = torch.ops.aten.split_with_sizes.default(arg0_1, [96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96], dim = 1)
        getitem_15: "f32[1024, 96]" = split_with_sizes_default_1[15]
        getitem_16: "f32[1024, 96]" = split_with_sizes_default_1[16]
        getitem_17: "f32[1024, 96]" = split_with_sizes_default_1[17];  split_with_sizes_default_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:102 in forward, code: split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(arg1_1, [96, 96, 96, 96], dim = 1)
        getitem_21: "f32[1024, 96]" = split_with_sizes_default[0];  getitem_21 = None
        getitem_22: "f32[1024, 96]" = split_with_sizes_default[1];  getitem_22 = None
        getitem_23: "f32[1024, 96]" = split_with_sizes_default[2];  getitem_23 = None
        getitem_24: "f32[1024, 96]" = split_with_sizes_default[3];  split_with_sizes_default = getitem_24 = None

        # No stacktrace found for following nodes
        slice_tensor: "f32[1024, 576]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 576)
        slice_tensor_1: "f32[1024, 480]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 960, 1440);  arg0_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:124 in forward, code: cat = torch.ops.aten.cat.default(
        cat_default: "f32[1024, 2112]" = torch.ops.aten.cat.default([arg3_1, slice_tensor, getitem_15, getitem_16, getitem_17, arg2_1, slice_tensor_1, arg2_1, arg1_1, arg3_1], dim = 1);  arg3_1 = slice_tensor = getitem_15 = getitem_16 = getitem_17 = arg2_1 = slice_tensor_1 = arg1_1 = None
        return (cat_default,)
