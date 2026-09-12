class TestCompileOnOneRankDeviceAsParameter(TestCase):

    @staticmethod
    def _coor_inductor_fn(x):
        z = torch.zeros(4, x.shape[1], device=x.device, dtype=x.dtype)
        return z + x.sum()

    def _assert_no_baked_device(self, code):
        self.assertNotRegex(code, 'npu:\\d')
        self.assertNotRegex(code, 'device\\(type=.npu., index=\\d')
        self.assertNotRegex(code, 'DeviceProperties\\([^)]*index=\\d')

    def _inductor_code_on_device(self, dev):
        from torch._inductor.utils import run_and_get_code
        torch._dynamo.reset()
        with torch.npu.device(dev):
            compiled = torch.compile(self._coor_inductor_fn, backend='inductor', fullgraph=True)
            _, codes = run_and_get_code(compiled, torch.randn(2, 8, device=f'npu:{dev}'))
        return '\n'.join(codes)
