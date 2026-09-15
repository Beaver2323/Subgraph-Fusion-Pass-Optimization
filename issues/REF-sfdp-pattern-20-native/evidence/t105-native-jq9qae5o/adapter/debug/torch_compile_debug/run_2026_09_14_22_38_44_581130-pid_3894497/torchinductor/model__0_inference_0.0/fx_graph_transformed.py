class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 2, 16, 32]", arg1_1: "f32[4, 2, 16, 32]", arg2_1: "f32[4, 2, 16, 32]"):
        # No stacktrace found for following nodes
        permute_default: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
        permute_default_1: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        permute_default_2: "f32[4, 16, 2, 32]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1334 in dot_prod_attention, code: ).tril(diagonal=0)
        iota: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze: "i64[1, 2]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze_1: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub: "i64[4, 2]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        le: "b8[4, 2]" = torch.ops.aten.le.Scalar(sub, 0);  sub = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1332 in dot_prod_attention, code: attn_mask = torch.ones(
        full_default: "b8[4, 2]" = torch.ops.aten.full.default([4, 2], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1334 in dot_prod_attention, code: ).tril(diagonal=0)
        logical_and: "b8[4, 2]" = torch.ops.aten.logical_and.default(le, full_default);  le = full_default = None

        # No stacktrace found for following nodes
        eq_scalar: "b8[4, 2]" = torch.ops.aten.eq.Scalar(logical_and, 1);  logical_and = None
        view_default: "b8[4, 1, 1, 2]" = torch.ops.aten.reshape.default(eq_scalar, [4, 1, 1, 2]);  eq_scalar = None
        expand_default: "b8[4, 16, 2, 2]" = torch.ops.aten.expand.default(view_default, [4, 16, 2, 2]);  view_default = None
        logical_not_default: "b8[4, 16, 2, 2]" = torch.ops.aten.logical_not.default(expand_default);  expand_default = None
        npu_fusion_attention_v3_default = torch.ops.npu.npu_fusion_attention_v3.default(permute_default, permute_default_1, permute_default_2, 16, 'BNSD', None, None, logical_not_default, 0.17677669529663687, 1.0, 2147483647, 2147483647, 2);  permute_default = permute_default_1 = permute_default_2 = logical_not_default = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fused_attention.py:1344 in dot_prod_attention, code: return torch.matmul(weights, v)
        getitem: "f32[4, 16, 2, 32]" = npu_fusion_attention_v3_default[0];  npu_fusion_attention_v3_default = None
        return (getitem,)
