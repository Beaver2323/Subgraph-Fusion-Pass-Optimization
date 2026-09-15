class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 2, 16, 32]", primals_2: "f32[4, 2, 16, 32]", primals_3: "f32[4, 2, 16, 32]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:950 in dot_prod_attention, code: attn_mask = torch.ones(
        full_default: "b8[2, 2]" = torch.ops.aten.full.default([2, 2], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:957 in dot_prod_attention, code: k = key.permute(0, 2, 1, 3)
        permute: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(primals_1, [0, 2, 1, 3]);  primals_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:956 in dot_prod_attention, code: q = query.permute(0, 2, 1, 3)
        permute_1: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(primals_2, [0, 2, 1, 3]);  primals_2 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:958 in dot_prod_attention, code: v = value.permute(0, 2, 1, 3)
        permute_2: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(primals_3, [0, 2, 1, 3]);  primals_3 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:960 in dot_prod_attention, code: (torch.matmul(q, k.transpose(-2, -1)).div(3.0) + attn_mask)
        permute_3: "f32[4, 16, 32, 2]" = torch.ops.aten.permute.default(permute, [0, 1, 3, 2]);  permute = None
        expand: "f32[4, 16, 2, 32]" = torch.ops.aten.expand.default(permute_1, [4, 16, 2, 32])
        clone: "f32[4, 16, 2, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f32[64, 2, 32]" = torch.ops.aten.reshape.default(clone, [64, 2, 32]);  clone = None
        expand_1: "f32[4, 16, 32, 2]" = torch.ops.aten.expand.default(permute_3, [4, 16, 32, 2])
        clone_1: "f32[4, 16, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(clone_1, [64, 32, 2]);  clone_1 = None
        bmm: "f32[64, 2, 2]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f32[4, 16, 2, 2]" = torch.ops.aten.reshape.default(bmm, [4, 16, 2, 2]);  bmm = None
        div: "f32[4, 16, 2, 2]" = torch.ops.aten.div.Tensor(view_2, 3.0);  view_2 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:952 in dot_prod_attention, code: ).tril(diagonal=0)
        iota: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze: "i64[1, 2]" = torch.ops.aten.unsqueeze.default(iota, -2)
        unsqueeze_1: "i64[2, 1]" = torch.ops.aten.unsqueeze.default(iota, -1);  iota = None
        sub: "i64[2, 2]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        le: "b8[2, 2]" = torch.ops.aten.le.Scalar(sub, 0);  sub = None
        logical_and: "b8[2, 2]" = torch.ops.aten.logical_and.default(le, full_default);  le = full_default = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:954 in dot_prod_attention, code: torch.logical_not(attn_mask), -float("inf")
        logical_not: "b8[2, 2]" = torch.ops.aten.logical_not.default(logical_and)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:953 in dot_prod_attention, code: attn_mask = attn_mask.masked_fill(
        full_default_1: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "b8[2, 2]" = torch.ops.aten.where.self(logical_not, full_default_1, logical_and);  logical_not = full_default_1 = logical_and = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:960 in dot_prod_attention, code: (torch.matmul(q, k.transpose(-2, -1)).div(3.0) + attn_mask)
        add: "f32[4, 16, 2, 2]" = torch.ops.aten.add.Tensor(div, where);  div = where = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:961 in dot_prod_attention, code: .softmax(dim=-1)
        amax: "f32[4, 16, 2, 1]" = torch.ops.aten.amax.default(add, [-1], True)
        sub_1: "f32[4, 16, 2, 2]" = torch.ops.aten.sub.Tensor(add, amax);  add = amax = None
        exp: "f32[4, 16, 2, 2]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[4, 16, 2, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[4, 16, 2, 2]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:962 in dot_prod_attention, code: .matmul(v)
        expand_2: "f32[4, 16, 2, 2]" = torch.ops.aten.expand.default(div_1, [4, 16, 2, 2])
        view_3: "f32[64, 2, 2]" = torch.ops.aten.reshape.default(expand_2, [64, 2, 2]);  expand_2 = None
        expand_3: "f32[4, 16, 2, 32]" = torch.ops.aten.expand.default(permute_2, [4, 16, 2, 32])
        clone_2: "f32[4, 16, 2, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_4: "f32[64, 2, 32]" = torch.ops.aten.reshape.default(clone_2, [64, 2, 32]);  clone_2 = None
        bmm_1: "f32[64, 2, 32]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f32[4, 16, 2, 32]" = torch.ops.aten.reshape.default(bmm_1, [4, 16, 2, 32]);  bmm_1 = None
        return (view_5, permute_1, permute_2, permute_3, div_1)
