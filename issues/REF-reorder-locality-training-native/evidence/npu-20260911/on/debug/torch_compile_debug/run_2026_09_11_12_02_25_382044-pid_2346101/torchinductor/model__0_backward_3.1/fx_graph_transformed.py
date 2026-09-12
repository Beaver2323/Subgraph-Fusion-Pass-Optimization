class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 16]", primals_2: "f32[8, 16]", primals_3: "f32[16, 16]", mm: "f32[8, 16]", mm_1: "f32[8, 16]", tangents_1: "f32[8]"):
        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_reorder_for_locality_in_training.py:74 in forward, code: return (a * b + torch.sin(a) * torch.cos(b)).sum(dim=1)
        unsqueeze: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(tangents_1, 1);  tangents_1 = None
        expand: "f32[8, 16]" = torch.ops.aten.expand.default(unsqueeze, [8, 16]);  unsqueeze = None

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_reorder_for_locality_in_training.py:72 in forward, code: a = torch.tanh(x @ self.w1)
        tanh: "f32[8, 16]" = torch.ops.aten.tanh.default(mm);  mm = None

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_reorder_for_locality_in_training.py:74 in forward, code: return (a * b + torch.sin(a) * torch.cos(b)).sum(dim=1)
        sin: "f32[8, 16]" = torch.ops.aten.sin.default(tanh)
        mul_2: "f32[8, 16]" = torch.ops.aten.mul.Tensor(expand, sin);  sin = None

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_reorder_for_locality_in_training.py:73 in forward, code: b = torch.sigmoid(x @ self.w2)
        sigmoid: "f32[8, 16]" = torch.ops.aten.sigmoid.default(mm_1);  mm_1 = None

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_reorder_for_locality_in_training.py:74 in forward, code: return (a * b + torch.sin(a) * torch.cos(b)).sum(dim=1)
        sin_1: "f32[8, 16]" = torch.ops.aten.sin.default(sigmoid)
        neg: "f32[8, 16]" = torch.ops.aten.neg.default(sin_1);  sin_1 = None
        mul_4: "f32[8, 16]" = torch.ops.aten.mul.Tensor(mul_2, neg);  mul_2 = neg = None
        mul_6: "f32[8, 16]" = torch.ops.aten.mul.Tensor(expand, tanh)
        add_2: "f32[8, 16]" = torch.ops.aten.add.Tensor(mul_4, mul_6);  mul_4 = mul_6 = None

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_reorder_for_locality_in_training.py:73 in forward, code: b = torch.sigmoid(x @ self.w2)
        sub: "f32[8, 16]" = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul_8: "f32[8, 16]" = torch.ops.aten.mul.Tensor(sigmoid, sub);  sub = None
        mul_9: "f32[8, 16]" = torch.ops.aten.mul.Tensor(add_2, mul_8);  add_2 = mul_8 = None
        matmul_backward = torch.ops.aten.matmul_backward.default(mul_9, primals_2, primals_3, [False, True]);  mul_9 = primals_3 = None
        getitem_1: "f32[16, 16]" = matmul_backward[1];  matmul_backward = None

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_reorder_for_locality_in_training.py:74 in forward, code: return (a * b + torch.sin(a) * torch.cos(b)).sum(dim=1)
        cos: "f32[8, 16]" = torch.ops.aten.cos.default(sigmoid)
        mul_3: "f32[8, 16]" = torch.ops.aten.mul.Tensor(expand, cos);  cos = None
        cos_1: "f32[8, 16]" = torch.ops.aten.cos.default(tanh)
        mul_5: "f32[8, 16]" = torch.ops.aten.mul.Tensor(mul_3, cos_1);  mul_3 = cos_1 = None
        mul_7: "f32[8, 16]" = torch.ops.aten.mul.Tensor(expand, sigmoid);  expand = sigmoid = None
        add_1: "f32[8, 16]" = torch.ops.aten.add.Tensor(mul_5, mul_7);  mul_5 = mul_7 = None

        # Backward of forward node: File: /home/z50063656/Pass/src/pytorch/test/inductor/test_reorder_for_locality_in_training.py:72 in forward, code: a = torch.tanh(x @ self.w1)
        mul_10: "f32[8, 16]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_1: "f32[8, 16]" = torch.ops.aten.sub.Tensor(1, mul_10);  mul_10 = None
        mul_11: "f32[8, 16]" = torch.ops.aten.mul.Tensor(add_1, sub_1);  add_1 = sub_1 = None
        matmul_backward_1 = torch.ops.aten.matmul_backward.default(mul_11, primals_2, primals_1, [False, True]);  mul_11 = primals_2 = primals_1 = None
        getitem_3: "f32[16, 16]" = matmul_backward_1[1];  matmul_backward_1 = None
        return (getitem_3, None, getitem_1)
