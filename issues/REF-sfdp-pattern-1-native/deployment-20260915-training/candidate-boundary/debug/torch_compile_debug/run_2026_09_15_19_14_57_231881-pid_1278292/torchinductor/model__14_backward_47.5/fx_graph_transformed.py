class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[2, 2, 8, 16]", primals_3: "f32[2, 2, 8, 16]", permute: "f32[2, 2, 16, 8]", div_1: "f32[2, 2, 8, 8]", tangents_1: "f32[2, 2, 8, 16]", tangents_2: "f32[2, 2, 8, 8]"):
        # Backward of forward node: File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/attention_training_boundary.py:51 in reuse, code: return weights @ v, weights
        matmul_backward = torch.ops.aten.matmul_backward.default(tangents_1, div_1, primals_3, [True, True]);  tangents_1 = primals_3 = None
        getitem: "f32[2, 2, 8, 8]" = matmul_backward[0]
        getitem_1: "f32[2, 2, 8, 16]" = matmul_backward[1];  matmul_backward = None
        add: "f32[2, 2, 8, 8]" = torch.ops.aten.add.Tensor(tangents_2, getitem);  tangents_2 = getitem = None

        # Backward of forward node: File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/attention_training_boundary.py:50 in reuse, code: weights = (q @ k.transpose(-2,-1)).div(4.0).softmax(-1)
        mul: "f32[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(add, div_1);  add = None
        sum_2: "f32[2, 2, 8, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        mul_1: "f32[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(div_1, sum_2);  div_1 = sum_2 = None
        sub_1: "f32[2, 2, 8, 8]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        div_2: "f32[2, 2, 8, 8]" = torch.ops.aten.div.Tensor(sub_1, 4.0);  sub_1 = None
        matmul_backward_1 = torch.ops.aten.matmul_backward.default(div_2, primals_2, permute, [True, True]);  div_2 = primals_2 = permute = None
        getitem_2: "f32[2, 2, 8, 16]" = matmul_backward_1[0]
        getitem_3: "f32[2, 2, 16, 8]" = matmul_backward_1[1];  matmul_backward_1 = None
        permute_1: "f32[2, 2, 8, 16]" = torch.ops.aten.permute.default(getitem_3, [0, 1, 3, 2]);  getitem_3 = None
        return (permute_1, getitem_2, getitem_1)
