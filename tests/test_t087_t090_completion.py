"""完成态证据门禁的零设备回归；不导入 torch，不执行生成代码。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("completion", ROOT / "scripts/review_t087_t090_completion.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class CompletionTests(unittest.TestCase):
    def test_archived_records_are_hash_bound(self):
        review.validate_archived()

    def test_no_module_file_is_safe(self):
        # 复核工具自己不加载 torch；sys.modules 内 None __file__ 不应使采集崩溃。
        self.assertIsInstance(review.WORKER.source_hashes(), dict)

    def test_generated_modules_are_not_implementation_sources(self):
        row = {"loaded_source_sha256": {"/home/z50063656/tmp/run/cache.py":"a", "/src/device.py":"b"}}
        self.assertEqual(review.implementation_sources(row), {"/src/device.py":"b"})

    def test_codegen_rejects_cpu_and_unknown_external_ops(self):
        for body in ('x.cpu()', 'torch.ops.aten.unsupported.default(x)'):
            with self.subTest(body=body), tempfile.TemporaryDirectory(dir='/home/z50063656/tmp') as tmp:
                base = Path(tmp)
                (base/'debug').mkdir()
                (base/'debug/output_code.py').write_text('import torch\n# torch.npu\n'+body)
                (base/'stderr.log').write_text('')
                with self.assertRaises(ValueError):
                    review.audit_code(base)

    def test_extern_npu_cat_is_explicit_not_cpu_fallback(self):
        with tempfile.TemporaryDirectory(dir='/home/z50063656/tmp') as tmp:
            base = Path(tmp)
            (base/'debug').mkdir()
            (base/'debug/output_code.py').write_text('import torch\n# torch.npu\ntorch.ops.aten.cat.default(xs, 1)')
            (base/'stderr.log').write_text('')
            row = review.audit_code(base)
            self.assertEqual(row['registered_npu_aten_calls'], {'aten.cat.default':1})
            self.assertEqual(row['unexpected_cpu_or_graph_external_fallbacks'], 0)


if __name__ == '__main__':
    unittest.main()
