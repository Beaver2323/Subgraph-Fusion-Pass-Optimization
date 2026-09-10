class <lambda>(torch.nn.Module):
    def forward(self):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:233 in function, code: value = torch.full([5000], 1.0, dtype=torch.float16, device=device)
        full_default: "f16[5000]" = torch.ops.aten.full.default([5000], 1.0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  full_default = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t085_performance_worker.py:234 in function, code: return torch.cumsum(value, 0, dtype=torch.float32)
        iota_default: "i64[5000]" = torch.ops.prims.iota.default(5000, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        mul_tensor: "i64[5000]" = torch.ops.aten.mul.Tensor(iota_default, 1);  iota_default = None
        add_tensor: "i64[5000]" = torch.ops.aten.add.Tensor(mul_tensor, 1);  mul_tensor = None
        convert_element_type_default: "f64[5000]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float64);  add_tensor = None
        mul_tensor_1: "f64[5000]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0);  convert_element_type_default = None
        expand_default: "f64[5000]" = torch.ops.aten.expand.default(mul_tensor_1, [5000]);  mul_tensor_1 = None
        convert_element_type_default_1: "f32[5000]" = torch.ops.prims.convert_element_type.default(expand_default, torch.float32);  expand_default = None
        return (convert_element_type_default_1,)
