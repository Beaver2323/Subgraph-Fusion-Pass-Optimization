class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[2, 2, 8, 16]", primals_3: "f32[]", primals_4: "f32[2, 2, 8, 16]", permute: "f32[2, 2, 16, 8]", div: "f32[2, 2, 8, 8]", tangents_1: "f32[2, 2, 8, 16]"):
        # Backward of forward node: File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/attention_training_boundary.py:54 in tensor_scale, code: return ((q @ k.transpose(-2,-1)) * extra).softmax(-1) @ v
        matmul_backward = torch.ops.aten.matmul_backward.default(tangents_1, div, primals_4, [True, True]);  tangents_1 = primals_4 = None
        getitem: "f32[2, 2, 8, 8]" = matmul_backward[0]
        getitem_1: "f32[2, 2, 8, 16]" = matmul_backward[1];  matmul_backward = None
        mul_1: "f32[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(getitem, div);  getitem = None
        sum_2: "f32[2, 2, 8, 1]" = torch.ops.aten.sum.dim_IntList(mul_1, [-1], True)
        mul_2: "f32[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(div, sum_2);  div = sum_2 = None
        sub_1: "f32[2, 2, 8, 8]" = torch.ops.aten.sub.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        mul_3: "f32[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(sub_1, primals_3);  sub_1 = primals_3 = None
        matmul_backward_1 = torch.ops.aten.matmul_backward.default(mul_3, primals_2, permute, [True, True]);  mul_3 = primals_2 = permute = None
        getitem_2: "f32[2, 2, 8, 16]" = matmul_backward_1[0]
        getitem_3: "f32[2, 2, 16, 8]" = matmul_backward_1[1];  matmul_backward_1 = None
        permute_1: "f32[2, 2, 8, 16]" = torch.ops.aten.permute.default(getitem_3, [0, 1, 3, 2]);  getitem_3 = None
        return (permute_1, getitem_2, None, getitem_1)
