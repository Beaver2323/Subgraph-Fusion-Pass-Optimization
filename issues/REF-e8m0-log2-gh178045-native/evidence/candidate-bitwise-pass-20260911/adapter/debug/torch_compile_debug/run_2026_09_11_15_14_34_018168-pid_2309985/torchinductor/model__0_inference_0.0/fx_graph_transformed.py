class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[7]"):
        # No stacktrace found for following nodes
        view_dtype: "i32[7]" = torch.ops.aten.view.dtype(arg0_1, torch.int32);  arg0_1 = None
        __rshift___scalar: "i32[7]" = torch.ops.aten.__rshift__.Scalar(view_dtype, 23)
        bitwise_and_scalar: "i32[7]" = torch.ops.aten.bitwise_and.Scalar(__rshift___scalar, 255);  __rshift___scalar = None
        bitwise_and_scalar_1: "i32[7]" = torch.ops.aten.bitwise_and.Scalar(view_dtype, 8388607);  view_dtype = None
        ne_scalar: "b8[7]" = torch.ops.aten.ne.Scalar(bitwise_and_scalar_1, 0);  bitwise_and_scalar_1 = None
        convert_element_type_default: "i32[7]" = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None
        add_tensor: "i32[7]" = torch.ops.aten.add.Tensor(bitwise_and_scalar, convert_element_type_default);  bitwise_and_scalar = convert_element_type_default = None
        clamp_min_default: "i32[7]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "i32[7]" = torch.ops.aten.clamp_max.default(clamp_min_default, 254);  clamp_min_default = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_fp8.py:2223 in encode_fn, code: return biased.to(torch.uint8)
        convert_element_type_default_1: "u8[7]" = torch.ops.prims.convert_element_type.default(clamp_max_default, torch.uint8);  clamp_max_default = None
        return (convert_element_type_default_1,)
