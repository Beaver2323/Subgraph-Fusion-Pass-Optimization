class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1016]"):
        # No stacktrace found for following nodes
        view_dtype: "i32[1016]" = torch.ops.aten.view.dtype(arg0_1, torch.int32);  arg0_1 = None
        __rshift___scalar: "i32[1016]" = torch.ops.aten.__rshift__.Scalar(view_dtype, 23)
        bitwise_and_scalar: "i32[1016]" = torch.ops.aten.bitwise_and.Scalar(__rshift___scalar, 255);  __rshift___scalar = None
        bitwise_and_scalar_1: "i32[1016]" = torch.ops.aten.bitwise_and.Scalar(view_dtype, 8388607)
        ne_scalar: "b8[1016]" = torch.ops.aten.ne.Scalar(bitwise_and_scalar_1, 0)
        convert_element_type_default: "i32[1016]" = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None
        add_tensor: "i32[1016]" = torch.ops.aten.add.Tensor(bitwise_and_scalar, convert_element_type_default);  convert_element_type_default = None
        clamp_min_default: "i32[1016]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "i32[1016]" = torch.ops.aten.clamp_max.default(clamp_min_default, 254);  clamp_min_default = None
        gt_scalar: "b8[1016]" = torch.ops.aten.gt.Scalar(bitwise_and_scalar_1, 4194304)
        convert_element_type_default_1: "i32[1016]" = torch.ops.prims.convert_element_type.default(gt_scalar, torch.int32);  gt_scalar = None
        eq_scalar: "b8[1016]" = torch.ops.aten.eq.Scalar(bitwise_and_scalar, 0)
        where_self: "i32[1016]" = torch.ops.aten.where.self(eq_scalar, convert_element_type_default_1, clamp_max_default);  eq_scalar = convert_element_type_default_1 = clamp_max_default = None
        lt_scalar: "b8[1016]" = torch.ops.aten.lt.Scalar(view_dtype, 0);  view_dtype = None
        eq_scalar_1: "b8[1016]" = torch.ops.aten.eq.Scalar(bitwise_and_scalar, 255);  bitwise_and_scalar = None
        ne_scalar_1: "b8[1016]" = torch.ops.aten.ne.Scalar(bitwise_and_scalar_1, 0);  bitwise_and_scalar_1 = None
        bitwise_and_tensor: "b8[1016]" = torch.ops.aten.bitwise_and.Tensor(eq_scalar_1, ne_scalar_1);  eq_scalar_1 = ne_scalar_1 = None
        bitwise_or_tensor: "b8[1016]" = torch.ops.aten.bitwise_or.Tensor(lt_scalar, bitwise_and_tensor);  lt_scalar = bitwise_and_tensor = None
        scalar_tensor_default: "i32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int32, layout = torch.strided, device = device(type='npu', index=0))
        where_self_1: "i32[1016]" = torch.ops.aten.where.self(bitwise_or_tensor, scalar_tensor_default, where_self);  bitwise_or_tensor = scalar_tensor_default = where_self = None

        # File: /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/runners/t096_installed_worker.py:87 in fn, code: return (torch.clamp(torch.ceil(torch.log2(x)), -127, 127) + 127).to(torch.uint8)
        convert_element_type_default_2: "u8[1016]" = torch.ops.prims.convert_element_type.default(where_self_1, torch.uint8);  where_self_1 = None
        return (convert_element_type_default_2,)
