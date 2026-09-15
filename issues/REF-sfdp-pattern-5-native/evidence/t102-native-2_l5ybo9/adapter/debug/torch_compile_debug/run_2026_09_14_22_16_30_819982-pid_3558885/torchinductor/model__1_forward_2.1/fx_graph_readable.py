class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 2, 16, 32]", primals_2: "f32[4, 2, 16, 32]", primals_3: "f32[4, 2, 16, 32]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:362 in sfdp_pattern_5_v1, code: attn_mask = torch.ones(
        full_default: "b8[16, 16]" = torch.ops.aten.full.default([16, 16], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:369 in sfdp_pattern_5_v1, code: (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
        permute: "f32[4, 2, 32, 16]" = torch.ops.aten.permute.default(primals_1, [0, 1, 3, 2]);  primals_1 = None
        expand: "f32[4, 2, 16, 32]" = torch.ops.aten.expand.default(primals_2, [4, 2, 16, 32])
        view: "f32[8, 16, 32]" = torch.ops.aten.view.default(expand, [8, 16, 32]);  expand = None
        expand_1: "f32[4, 2, 32, 16]" = torch.ops.aten.expand.default(permute, [4, 2, 32, 16])
        view_1: "f32[8, 32, 16]" = torch.ops.aten.view.default(expand_1, [8, 32, 16]);  expand_1 = None
        bmm: "f32[8, 16, 16]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f32[4, 2, 16, 16]" = torch.ops.aten.view.default(bmm, [4, 2, 16, 16]);  bmm = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:364 in sfdp_pattern_5_v1, code: ).tril(diagonal=0)
        iota: "i64[16]" = torch.ops.prims.iota.default(16, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze: "i64[1, 16]" = torch.ops.aten.unsqueeze.default(iota, -2)
        unsqueeze_1: "i64[16, 1]" = torch.ops.aten.unsqueeze.default(iota, -1);  iota = None
        sub: "i64[16, 16]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        le: "b8[16, 16]" = torch.ops.aten.le.Scalar(sub, 0);  sub = None
        logical_and: "b8[16, 16]" = torch.ops.aten.logical_and.default(le, full_default);  le = full_default = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:366 in sfdp_pattern_5_v1, code: torch.logical_not(attn_mask), -float("inf")
        logical_not: "b8[16, 16]" = torch.ops.aten.logical_not.default(logical_and)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:365 in sfdp_pattern_5_v1, code: attn_mask = attn_mask.masked_fill(
        full_default_1: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "b8[16, 16]" = torch.ops.aten.where.self(logical_not, full_default_1, logical_and);  logical_not = full_default_1 = logical_and = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:369 in sfdp_pattern_5_v1, code: (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
        div: "f32[4, 2, 16, 16]" = torch.ops.aten.div.Tensor(view_2, 5.656854249492381);  view_2 = None
        add: "f32[4, 2, 16, 16]" = torch.ops.aten.add.Tensor(div, where);  div = where = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:368 in sfdp_pattern_5_v1, code: attn_weight = torch.softmax(
        amax: "f32[4, 2, 16, 1]" = torch.ops.aten.amax.default(add, [-1], True)
        sub_1: "f32[4, 2, 16, 16]" = torch.ops.aten.sub.Tensor(add, amax);  add = amax = None
        exp: "f32[4, 2, 16, 16]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[4, 2, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[4, 2, 16, 16]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:372 in sfdp_pattern_5_v1, code: return attn_weight @ value
        expand_2: "f32[4, 2, 16, 16]" = torch.ops.aten.expand.default(div_1, [4, 2, 16, 16])
        view_3: "f32[8, 16, 16]" = torch.ops.aten.view.default(expand_2, [8, 16, 16]);  expand_2 = None
        expand_3: "f32[4, 2, 16, 32]" = torch.ops.aten.expand.default(primals_3, [4, 2, 16, 32])
        view_4: "f32[8, 16, 32]" = torch.ops.aten.view.default(expand_3, [8, 16, 32]);  expand_3 = None
        bmm_1: "f32[8, 16, 32]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f32[4, 2, 16, 32]" = torch.ops.aten.view.default(bmm_1, [4, 2, 16, 32]);  bmm_1 = None
        return (view_5, primals_2, primals_3, permute, div_1)
