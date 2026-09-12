class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024, 6, 128]", arg1_1: "f32[1024, 6, 128]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:199 in forward, code: select = torch.ops.aten.select.int(x, 1, 0)
        select: "f32[1024, 128]" = torch.ops.aten.select.int(arg0_1, 1, 0)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:200 in forward, code: select_1 = torch.ops.aten.select.int(x, 1, 1)
        select_1: "f32[1024, 128]" = torch.ops.aten.select.int(arg0_1, 1, 1)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:201 in forward, code: select_2 = torch.ops.aten.select.int(x, 1, 2)
        select_2: "f32[1024, 128]" = torch.ops.aten.select.int(arg0_1, 1, 2)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:202 in forward, code: select_3 = torch.ops.aten.select.int(x, 1, 3)
        select_3: "f32[1024, 128]" = torch.ops.aten.select.int(arg0_1, 1, 3)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:203 in forward, code: select_4 = torch.ops.aten.select.int(x, 1, 4)
        select_4: "f32[1024, 128]" = torch.ops.aten.select.int(arg0_1, 1, 4)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:208 in forward, code: cat1 = torch.ops.aten.cat.default(
        cat: "f32[1024, 640]" = torch.ops.aten.cat.default([select, select_1, select_2, select_3, select_4], 1)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:211 in forward, code: cat2 = torch.ops.aten.cat.default([select, select_2, select_4], 1)
        cat_1: "f32[1024, 384]" = torch.ops.aten.cat.default([select, select_2, select_4], 1)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:204 in forward, code: select_5 = torch.ops.aten.select.int(x, 1, 5)
        select_5: "f32[1024, 128]" = torch.ops.aten.select.int(arg0_1, 1, 5);  arg0_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:205 in forward, code: cat = torch.ops.aten.cat.default(
        cat_2: "f32[1024, 768]" = torch.ops.aten.cat.default([select, select_1, select_2, select_3, select_4, select_5], 1);  select = select_1 = select_2 = select_3 = select_4 = select_5 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:212 in forward, code: select_6 = torch.ops.aten.select.int(y, 1, 0)
        select_6: "f32[1024, 128]" = torch.ops.aten.select.int(arg1_1, 1, 0)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:213 in forward, code: select_7 = torch.ops.aten.select.int(y, 1, 1)
        select_7: "f32[1024, 128]" = torch.ops.aten.select.int(arg1_1, 1, 1)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:214 in forward, code: select_8 = torch.ops.aten.select.int(y, 1, 2)
        select_8: "f32[1024, 128]" = torch.ops.aten.select.int(arg1_1, 1, 2)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:215 in forward, code: select_9 = torch.ops.aten.select.int(y, 1, 3)
        select_9: "f32[1024, 128]" = torch.ops.aten.select.int(arg1_1, 1, 3)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:216 in forward, code: select_10 = torch.ops.aten.select.int(y, 1, 4)
        select_10: "f32[1024, 128]" = torch.ops.aten.select.int(arg1_1, 1, 4);  arg1_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_split_cat_fx_aten_passes.py:217 in forward, code: cat3 = torch.ops.aten.cat.default(
        cat_3: "f32[1024, 640]" = torch.ops.aten.cat.default([select_6, select_7, select_8, select_9, select_10], 1);  select_6 = select_7 = select_8 = select_9 = select_10 = None
        return (cat_2, cat, cat_1, cat_3)
