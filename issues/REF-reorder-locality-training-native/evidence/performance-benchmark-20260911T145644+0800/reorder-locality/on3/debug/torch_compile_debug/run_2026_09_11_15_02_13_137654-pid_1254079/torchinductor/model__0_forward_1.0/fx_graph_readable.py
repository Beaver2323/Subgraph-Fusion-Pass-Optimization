class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 16]", primals_2: "f32[8, 16]", primals_3: "f32[16, 16]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:265 in forward, code: a = torch.tanh(x @ self.w1)
        mm: "f32[8, 16]" = torch.ops.aten.mm.default(primals_2, primals_1)
        tanh: "f32[8, 16]" = torch.ops.aten.tanh.default(mm)

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:266 in forward, code: b = torch.sigmoid(x @ self.w2)
        mm_1: "f32[8, 16]" = torch.ops.aten.mm.default(primals_2, primals_3)
        sigmoid: "f32[8, 16]" = torch.ops.aten.sigmoid.default(mm_1)

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t087_t090_performance_worker.py:267 in forward, code: return (a * b + torch.sin(a) * torch.cos(b)).sum(dim=1)
        mul: "f32[8, 16]" = torch.ops.aten.mul.Tensor(tanh, sigmoid)
        sin: "f32[8, 16]" = torch.ops.aten.sin.default(tanh);  tanh = None
        cos: "f32[8, 16]" = torch.ops.aten.cos.default(sigmoid);  sigmoid = None
        mul_1: "f32[8, 16]" = torch.ops.aten.mul.Tensor(sin, cos);  sin = cos = None
        add: "f32[8, 16]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        sum_1: "f32[8]" = torch.ops.aten.sum.dim_IntList(add, [1]);  add = None
        return (sum_1, primals_1, primals_2, primals_3, mm, mm_1)
