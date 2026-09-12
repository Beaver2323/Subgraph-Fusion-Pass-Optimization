class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 8]"):
        # File: /home/z50063656/Pass/src/pytorch/test/distributed/tensor/test_compile_on_one_rank.py:487 in _coor_inductor_fn, code: z = torch.zeros(4, x.shape[1], device=x.device, dtype=x.dtype)
        full: "f32[4, 8]" = torch.ops.aten.full.default([4, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=1), pin_memory = False)

        # File: /home/z50063656/Pass/src/pytorch/test/distributed/tensor/test_compile_on_one_rank.py:488 in _coor_inductor_fn, code: return z + x.sum()
        sum_1: "f32[]" = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
        add: "f32[4, 8]" = torch.ops.aten.add.Tensor(full, sum_1);  full = sum_1 = None
        return (add,)
