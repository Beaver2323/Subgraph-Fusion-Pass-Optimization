class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2048, 2]", primals_2: "f32[2, 2]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/issues/REF-decompose-mm-native/npu_adapter.py:20 in mm_fn, code: return torch.mm(left, right)
        mm: "f32[2048, 2]" = torch.ops.aten.mm.default(primals_1, primals_2)

        # Backward of forward node: File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/issues/REF-decompose-mm-native/npu_adapter.py:20 in mm_fn, code: return torch.mm(left, right)
        permute: "f32[2, 2048]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        permute_1: "f32[2, 2]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        return (mm, permute, permute_1)
