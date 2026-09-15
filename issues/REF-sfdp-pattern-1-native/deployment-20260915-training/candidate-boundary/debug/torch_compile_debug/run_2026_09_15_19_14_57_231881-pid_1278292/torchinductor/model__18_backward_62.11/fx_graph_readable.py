class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[2, 2, 8, 16]", primals_4: "f32[2, 2, 8, 16]", permute: "f32[2, 2, 16, 8]", div_1: "f32[2, 2, 8, 8]", tangents_1: "f32[2, 2, 8, 16]"):
        # Backward of forward node: File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/attention_training_boundary.py:57 in mask, code: return ((q @ k.transpose(-2,-1)).div(4.0) + extra).softmax(-1) @ v
        matmul_backward = torch.ops.aten.matmul_backward.default(tangents_1, div_1, primals_4, [True, True]);  tangents_1 = primals_4 = None
        getitem: "f32[2, 2, 8, 8]" = matmul_backward[0]
        getitem_1: "f32[2, 2, 8, 16]" = matmul_backward[1];  matmul_backward = None
        mul: "f32[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(getitem, div_1);  getitem = None
        sum_2: "f32[2, 2, 8, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        mul_1: "f32[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(div_1, sum_2);  div_1 = sum_2 = None
        sub_1: "f32[2, 2, 8, 8]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        div_2: "f32[2, 2, 8, 8]" = torch.ops.aten.div.Tensor(sub_1, 4.0);  sub_1 = None
        matmul_backward_1 = torch.ops.aten.matmul_backward.default(div_2, primals_2, permute, [True, True]);  div_2 = primals_2 = permute = None
        getitem_2: "f32[2, 2, 8, 16]" = matmul_backward_1[0]
        getitem_3: "f32[2, 2, 16, 8]" = matmul_backward_1[1];  matmul_backward_1 = None
        permute_1: "f32[2, 2, 8, 16]" = torch.ops.aten.permute.default(getitem_3, [0, 1, 3, 2]);  getitem_3 = None
        return (permute_1, getitem_2, None, getitem_1)
