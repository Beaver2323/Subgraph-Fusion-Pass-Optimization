class GraphModule(torch.nn.Module):
    def forward(self, permute: "f32[2, 2048]", permute_1: "f32[2, 2]", tangents_1: "f32[2048, 2]"):
        # Backward of forward node: File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/issues/REF-decompose-mm-native/npu_adapter.py:20 in mm_fn, code: return torch.mm(left, right)
        mm_1: "f32[2, 2]" = torch.ops.aten.mm.default(permute, tangents_1);  permute = None
        mm_2: "f32[2048, 2]" = torch.ops.aten.mm.default(tangents_1, permute_1);  tangents_1 = permute_1 = None
        return (mm_2, mm_1)
