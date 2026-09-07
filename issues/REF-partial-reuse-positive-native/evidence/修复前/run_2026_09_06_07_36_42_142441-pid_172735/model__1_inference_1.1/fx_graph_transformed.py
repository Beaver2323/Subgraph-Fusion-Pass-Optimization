class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024, 1024]"):
        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_pattern_matcher.py:564 in test_fn, code: partial = partial_fn(x, [0], True)
        amin_default: "f32[1, 1024]" = torch.ops.aten.amin.default(arg0_1, [0], True);  arg0_1 = None

        # File: /home/z50063656/Pass/src/pytorch/test/inductor/test_pattern_matcher.py:565 in test_fn, code: full = full_fn(x)
        min_default: "f32[]" = torch.ops.aten.min.default(amin_default)
        return (amin_default, min_default)
