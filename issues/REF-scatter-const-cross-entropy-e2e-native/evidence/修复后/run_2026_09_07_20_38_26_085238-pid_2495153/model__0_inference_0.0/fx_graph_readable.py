class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[2, 1024, 1]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/verify_t080_scatter_gate.py:99 in fn, code: output = torch.full((2, 1024, 2048), 3.14, device="npu")
        full_default: "f32[2, 1024, 2048]" = torch.ops.aten.full.default([2, 1024, 2048], 3.14, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/verify_t080_scatter_gate.py:100 in fn, code: return output.scatter(2, index, 2.718)
        scatter: "f32[2, 1024, 2048]" = torch.ops.aten.scatter.value(full_default, 2, arg0_1, 2.718);  full_default = arg0_1 = None
        return (scatter,)
