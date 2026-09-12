class TestE8M0Log2PatternBitManip(TestCase):

    def test_correct_for_values_one_ulp_above_power_of_two(self):
        """Values 1 ULP above a power of 2 must produce the correct (higher) exponent.

        Software log2 for x = 2^e + 1 ULP may round to exactly e when e is
        large (the ULP of the log2 output exceeds the true fractional part).
        The bit-manipulation replacement reads the exponent field directly and
        is always correct.
        """
        E8M0_BIAS = 127

        def fn(inp):
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)
        powers = [float(2 ** e) for e in range(8)]
        one_ulp_above = [torch.nextafter(torch.tensor(p), torch.tensor(float('inf'))).item() for p in powers]
        inp = torch.tensor(one_ulp_above, device='npu', dtype=torch.float32)
        expected = torch.tensor([E8M0_BIAS + e + 1 for e in range(8)], dtype=torch.uint8, device='npu')
        compiled_result = torch.compile(fn)(inp)
        self.assertEqual(compiled_result, expected)
