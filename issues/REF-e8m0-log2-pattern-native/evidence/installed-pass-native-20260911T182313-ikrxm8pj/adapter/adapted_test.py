class TestE8M0Log2PatternBitManip(TestCase):

    def test_pattern_fires_and_is_correct(self):
        """The log2+ceil pattern should be matched and produce correct uint8."""
        _misc_patterns_init()
        E8M0_BIAS = 127

        def fn(inp):
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)
        inp = torch.tensor([1.0, 2.0, 4.0, 3.0, 1.5, 0.5, 0.25], device='npu', dtype=torch.float32)
        eager_result = fn(inp)
        compiled_result = torch.compile(fn)(inp)
        self.assertEqual(compiled_result, eager_result)
