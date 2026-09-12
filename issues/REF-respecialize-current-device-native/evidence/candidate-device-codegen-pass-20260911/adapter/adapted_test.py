class TestCompileOnOneRankDeviceAsParameter(TestCase):

    @staticmethod
    def _coor_inductor_fn(x):
        z = torch.zeros(4, x.shape[1], device=x.device, dtype=x.dtype)
        return z + x.sum()

    @unittest.skipIf(not torch.npu.is_available(), 'requires NPU')
    @compiler_config.patch(compile_on_one_rank=True)
    def test_inductor_compiles_under_coor(self):
        from torch._C import FileCheck
        from torch._inductor.utils import run_and_get_code
        torch._dynamo.reset()
        compiled = torch.compile(self._coor_inductor_fn, backend='inductor', fullgraph=True)
        out, codes = run_and_get_code(compiled, torch.randn(2, 8, device='npu'))
        self.assertEqual(out.device.type, 'npu')
        code = '\n'.join(codes)
        FileCheck().check('torch.npu.current_device()').run(code)
        self._assert_no_baked_device(code)

    def _assert_no_baked_device(self, code):
        self.assertNotRegex(code, 'npu:\\d')
        self.assertNotRegex(code, 'device\\(type=.npu., index=\\d')
        self.assertNotRegex(code, 'DeviceProperties\\([^)]*index=\\d')
