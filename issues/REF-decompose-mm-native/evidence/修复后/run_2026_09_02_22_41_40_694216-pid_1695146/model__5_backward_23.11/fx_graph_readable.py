class GraphModule(torch.nn.Module):
    def forward(self, permute: "bf16[2, 2048]", permute_1: "bf16[2, 2]", tangents_1: "bf16[2048, 2]"):
        # Backward of forward node: File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/issues/REF-decompose-mm-native/npu_adapter.py:20 in mm_fn, code: return torch.mm(left, right)
        mm_1: "bf16[2, 2]" = torch.ops.aten.mm.default(permute, tangents_1);  permute = None
        mm_2: "bf16[2048, 2]" = torch.ops.aten.mm.default(tangents_1, permute_1);  tangents_1 = permute_1 = None
        convert_element_type_8: "f32[2, 2]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_9: "f32[2048, 2]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        return (convert_element_type_9, convert_element_type_8)
