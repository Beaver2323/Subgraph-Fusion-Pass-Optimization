class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 256, 64]", arg1_1: "f32[1, 64, 256]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t079_performance_worker.py:124 in fn, code: return torch.bmm(a, b)
        squeeze_dim: "f32[256, 64]" = torch.ops.aten.squeeze.dim(arg0_1, 0);  arg0_1 = None
        squeeze_dim_1: "f32[64, 256]" = torch.ops.aten.squeeze.dim(arg1_1, 0);  arg1_1 = None
        mm_default: "f32[256, 256]" = torch.ops.aten.mm.default(squeeze_dim, squeeze_dim_1);  squeeze_dim = squeeze_dim_1 = None
        unsqueeze_default: "f32[1, 256, 256]" = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None
        return (unsqueeze_default,)
