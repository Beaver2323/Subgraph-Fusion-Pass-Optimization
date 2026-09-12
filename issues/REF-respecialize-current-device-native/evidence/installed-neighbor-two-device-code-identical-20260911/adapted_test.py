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

    @unittest.skipIf(torch.npu.device_count() < 2, 'requires >= 2 GPUs')
    @compiler_config.patch(compile_on_one_rank=True)
    def test_inductor_code_identical_across_devices(self):
        import re

        def norm(s):
            return re.sub("AOT ID: \\['\\d+_", "AOT ID: ['N_", s)
        for cfg in ({'benchmark_kernel': True}, {'triton.autotune_at_compile_time': True}):
            with self.subTest(cfg=cfg), torch._inductor.config.patch(**cfg):
                code0 = self._inductor_code_on_device(0)
                code1 = self._inductor_code_on_device(1)
                if norm(code0) != norm(code1):
                    diff = ''.join(difflib.unified_diff(norm(code0).splitlines(keepends=True), norm(code1).splitlines(keepends=True), fromfile='npu:0', tofile='npu:1'))
                    self.fail(f'inductor code differs across devices under CooR with {cfg}:\n{diff}')
                self._assert_no_baked_device(code0)
