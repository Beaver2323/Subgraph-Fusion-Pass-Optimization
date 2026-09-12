class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[7]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t096_installed_worker.py:87 in fn, code: return (torch.clamp(torch.ceil(torch.log2(x)), -127, 127) + 127).to(torch.uint8)
        log2: "f16[7]" = torch.ops.aten.log2.default(arg0_1);  arg0_1 = None
        ceil: "f16[7]" = torch.ops.aten.ceil.default(log2);  log2 = None
        convert_element_type: "f32[7]" = torch.ops.prims.convert_element_type.default(ceil, torch.float32);  ceil = None
        clamp_min: "f32[7]" = torch.ops.aten.clamp_min.default(convert_element_type, -127);  convert_element_type = None
        clamp_max: "f32[7]" = torch.ops.aten.clamp_max.default(clamp_min, 127);  clamp_min = None
        convert_element_type_1: "f16[7]" = torch.ops.prims.convert_element_type.default(clamp_max, torch.float16);  clamp_max = None
        add: "f16[7]" = torch.ops.aten.add.Tensor(convert_element_type_1, 127);  convert_element_type_1 = None
        convert_element_type_2: "u8[7]" = torch.ops.prims.convert_element_type.default(add, torch.uint8);  add = None
        return (convert_element_type_2,)
