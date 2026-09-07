class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2048, 2]", primals_2: "f32[2, 2]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/issues/REF-decompose-mm-native/npu_adapter.py:20 in mm_fn, code: return torch.mm(left, right)
        convert_element_type: "bf16[2048, 2]" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_1: "bf16[2, 2]" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        mm: "bf16[2048, 2]" = torch.ops.aten.mm.default(convert_element_type, convert_element_type_1)

        # Backward of forward node: File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/issues/REF-decompose-mm-native/npu_adapter.py:20 in mm_fn, code: return torch.mm(left, right)
        permute: "bf16[2, 2048]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        permute_1: "bf16[2, 2]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        return (mm, permute, permute_1)
