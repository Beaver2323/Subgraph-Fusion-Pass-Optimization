class <lambda>(torch.nn.Module):
    def forward(self):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:233 in function, code: value = torch.full([5000], 1.0, dtype=torch.float16, device=device)
        full_default: "f16[5000]" = torch.ops.aten.full.default([5000], 1.0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:234 in function, code: return torch.cumsum(value, 0, dtype=torch.float32)
        cumsum: "f32[5000]" = torch.ops.aten.cumsum.default(full_default, 0, dtype = torch.float32);  full_default = None
        return (cumsum,)
