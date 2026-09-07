class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4096, 512]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_pattern_matcher.py:564 in test_fn, code: partial = partial_fn(x, [0], True)
        amax: "f32[1, 512]" = torch.ops.aten.amax.default(arg0_1, [0], True)

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_pattern_matcher.py:565 in test_fn, code: full = full_fn(x)
        max_1: "f32[]" = torch.ops.aten.max.default(arg0_1);  arg0_1 = None
        return (amax, max_1)
