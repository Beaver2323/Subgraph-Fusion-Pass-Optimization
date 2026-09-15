class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2, 8, 16]", primals_2: "f32[2, 2, 8, 16]", primals_3: "f32[]", primals_4: "f32[2, 2, 8, 16]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/attention_training_boundary.py:57 in mask, code: return ((q @ k.transpose(-2,-1)).div(4.0) + extra).softmax(-1) @ v
        permute: "f32[2, 2, 16, 8]" = torch.ops.aten.permute.default(primals_1, [0, 1, 3, 2]);  primals_1 = None
        expand: "f32[2, 2, 8, 16]" = torch.ops.aten.expand.default(primals_2, [2, 2, 8, 16])
        view: "f32[4, 8, 16]" = torch.ops.aten.view.default(expand, [4, 8, 16]);  expand = None
        expand_1: "f32[2, 2, 16, 8]" = torch.ops.aten.expand.default(permute, [2, 2, 16, 8])
        view_1: "f32[4, 16, 8]" = torch.ops.aten.view.default(expand_1, [4, 16, 8]);  expand_1 = None
        bmm: "f32[4, 8, 8]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f32[2, 2, 8, 8]" = torch.ops.aten.view.default(bmm, [2, 2, 8, 8]);  bmm = None
        div: "f32[2, 2, 8, 8]" = torch.ops.aten.div.Tensor(view_2, 4.0);  view_2 = None
        add: "f32[2, 2, 8, 8]" = torch.ops.aten.add.Tensor(div, primals_3);  div = primals_3 = None
        amax: "f32[2, 2, 8, 1]" = torch.ops.aten.amax.default(add, [-1], True)
        sub: "f32[2, 2, 8, 8]" = torch.ops.aten.sub.Tensor(add, amax);  add = amax = None
        exp: "f32[2, 2, 8, 8]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[2, 2, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[2, 2, 8, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        expand_2: "f32[2, 2, 8, 8]" = torch.ops.aten.expand.default(div_1, [2, 2, 8, 8])
        view_3: "f32[4, 8, 8]" = torch.ops.aten.view.default(expand_2, [4, 8, 8]);  expand_2 = None
        expand_3: "f32[2, 2, 8, 16]" = torch.ops.aten.expand.default(primals_4, [2, 2, 8, 16])
        view_4: "f32[4, 8, 16]" = torch.ops.aten.view.default(expand_3, [4, 8, 16]);  expand_3 = None
        bmm_1: "f32[4, 8, 16]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f32[2, 2, 8, 16]" = torch.ops.aten.view.default(bmm_1, [2, 2, 8, 16]);  bmm_1 = None
        return (view_5, primals_2, primals_4, permute, div_1)
