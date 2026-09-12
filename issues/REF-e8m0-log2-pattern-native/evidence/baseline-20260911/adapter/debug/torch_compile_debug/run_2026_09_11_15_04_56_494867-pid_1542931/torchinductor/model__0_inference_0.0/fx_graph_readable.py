class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[7]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fp8.py:2157 in fn, code: log2_val = torch.log2(inp)
        log2: "f32[7]" = torch.ops.aten.log2.default(arg0_1);  arg0_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fp8.py:2158 in fn, code: ceil_val = torch.ceil(log2_val)
        ceil: "f32[7]" = torch.ops.aten.ceil.default(log2);  log2 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fp8.py:2159 in fn, code: clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
        clamp_min: "f32[7]" = torch.ops.aten.clamp_min.default(ceil, -127);  ceil = None
        clamp_max: "f32[7]" = torch.ops.aten.clamp_max.default(clamp_min, 127);  clamp_min = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fp8.py:2160 in fn, code: biased = clamped + E8M0_BIAS
        add: "f32[7]" = torch.ops.aten.add.Tensor(clamp_max, 127);  clamp_max = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fp8.py:2161 in fn, code: return biased.to(torch.uint8)
        convert_element_type: "u8[7]" = torch.ops.prims.convert_element_type.default(add, torch.uint8);  add = None
        return (convert_element_type,)
