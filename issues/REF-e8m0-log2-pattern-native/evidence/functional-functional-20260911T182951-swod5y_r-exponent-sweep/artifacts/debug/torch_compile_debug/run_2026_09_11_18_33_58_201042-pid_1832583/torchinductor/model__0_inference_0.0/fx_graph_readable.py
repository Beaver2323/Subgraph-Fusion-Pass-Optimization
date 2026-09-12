class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1016]"):
        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t096_installed_worker.py:87 in fn, code: return (torch.clamp(torch.ceil(torch.log2(x)), -127, 127) + 127).to(torch.uint8)
        log2: "f32[1016]" = torch.ops.aten.log2.default(arg0_1);  arg0_1 = None
        ceil: "f32[1016]" = torch.ops.aten.ceil.default(log2);  log2 = None
        clamp_min: "f32[1016]" = torch.ops.aten.clamp_min.default(ceil, -127);  ceil = None
        clamp_max: "f32[1016]" = torch.ops.aten.clamp_max.default(clamp_min, 127);  clamp_min = None
        add: "f32[1016]" = torch.ops.aten.add.Tensor(clamp_max, 127);  clamp_max = None
        convert_element_type: "u8[1016]" = torch.ops.prims.convert_element_type.default(add, torch.uint8);  add = None
        return (convert_element_type,)
