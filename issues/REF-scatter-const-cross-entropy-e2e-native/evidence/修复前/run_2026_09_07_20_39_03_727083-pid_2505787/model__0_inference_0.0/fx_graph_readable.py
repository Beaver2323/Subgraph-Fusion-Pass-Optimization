class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[2, 1024, 1]"):
        # No stacktrace found for following nodes
        iota_default: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        view_default: "i64[1, 1, 2048]" = torch.ops.aten.view.default(iota_default, [1, 1, 2048]);  iota_default = None
        expand_default: "i64[2, 1024, 2048]" = torch.ops.aten.expand.default(arg0_1, [2, 1024, 2048]);  arg0_1 = None
        eq_tensor: "b8[2, 1024, 2048]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(3.14, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(2.718, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0))

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/verify_t080_scatter_gate.py:100 in fn, code: return output.scatter(2, index, 2.718)
        where_self: "f32[2, 1024, 2048]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        return (where_self,)
