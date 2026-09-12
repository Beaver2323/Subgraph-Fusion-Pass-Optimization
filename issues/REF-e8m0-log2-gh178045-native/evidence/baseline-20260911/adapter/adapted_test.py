class TestE8M0Log2PatternBitManip(TestCase):

    def test_regression_gh178045_encoding_correctness(self):
        """Regression for gh-178045: encoding must be correct for inputs near power-of-2.

        When a compiled kernel (e.g. fused GELU) produces a float value that is
        just above a power of 2 (say 2^e + 1 ULP), the software log2+ceil
        pipeline may give the WRONG result because float32 log2 rounds the
        result down to exactly e, making ceil return e instead of the correct
        e+1.  The bit-manipulation replacement always returns the correct value.

        This test specifically validates that the bit-manipulation codepath gives
        mathematically correct results for such boundary values (i.e. that it
        fixes the correctness bug distinct from the GELU input discrepancy).
        """
        E8M0_BIAS = 127

        def encode_fn(inp):
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)
        powers = [float(2 ** e) for e in range(7)]
        one_ulp_above = [torch.nextafter(torch.tensor(p), torch.tensor(float('inf'))).item() for p in powers]
        inp = torch.tensor(one_ulp_above, device='npu', dtype=torch.float32)
        expected = torch.tensor([E8M0_BIAS + e + 1 for e in range(7)], dtype=torch.uint8, device='npu')
        torch._dynamo.reset()
        compiled_result = torch.compile(encode_fn)(inp)
        self.assertEqual(compiled_result, expected, msg='Bit-manipulation replacement must return the correct e8m0 ceiling for values 1 ULP above a power of 2 (gh-178045 regression).')
