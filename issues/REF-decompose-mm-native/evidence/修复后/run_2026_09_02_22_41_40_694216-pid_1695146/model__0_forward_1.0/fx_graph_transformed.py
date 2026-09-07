class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[20480, 5]", primals_2: "f32[5, 2]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/issues/REF-decompose-mm-native/npu_adapter.py:20 in mm_fn, code: return torch.mm(left, right)
        mm: "f32[20480, 2]" = torch.ops.aten.mm.default(primals_1, primals_2)

        # Backward of forward node: File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/issues/REF-decompose-mm-native/npu_adapter.py:20 in mm_fn, code: return torch.mm(left, right)
        permute: "f32[5, 20480]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        permute_1: "f32[2, 5]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        return (mm, permute, permute_1)
