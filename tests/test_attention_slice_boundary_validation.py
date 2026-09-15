"""边界验收必须核对原件、完整执行矩阵与无候选安装态。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('boundary_validation',
                                            ROOT / 'scripts/validate_attention_slice_boundary.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
spec = importlib.util.spec_from_file_location('boundary_fixture',
                                            ROOT / 'runners/attention_select_slice_boundary.py')
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


class BoundaryValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('installed_triton_source.py', 'runner_source.py', 'emitted-000.txt'):
            (self.root / name).write_text('raise RuntimeError("must not execute")\n# _es_full.shape[0] >= 0\n')
        self.product = validation.sha(self.root / 'installed_triton_source.py')
        self.runner = validation.sha(self.root / 'runner_source.py')
        cases = []
        for index, case in enumerate(boundary.scenarios()):
            rows = [7, 11] if case['dynamic'] else [7]
            cases.append(dict(case, index=index, status='passed', executions=[
                dict(rows=n, bitwise_equal=True, mismatch=0, output_shape=[3,n,5],
                     storage_offset=int(case['layout']=='offset'),
                     stride=[56,8,2] if case['layout']=='strided' else [28,4,1]) for n in rows]))
        self.record = dict(status='passed', arm='installed', device_execution=True,
                           target_exercised=True, backend='triton_experimental',
                           pytorch_commit=validation.COMMIT, installed_before=self.product,
                           installed_after=self.product, source_sha256=self.runner,
                           candidate=None, cases=cases,
                           emitted=[dict(case=0,file='emitted-000.txt',sha256=self.product,
                                         per_load_shape=True,bounded_load=True)])

    def verify(self):
        path = self.root / 'result.json'
        path.write_text(json.dumps(self.record))
        return validation.verify(path, self.product, self.runner, 'installed')

    def test_complete_fixture_is_read_without_execution(self):
        self.assertEqual(self.verify()['executions'], 23)

    def test_rejects_incomplete_dynamic_execution(self):
        self.record['cases'][-1]['executions'].pop()
        with self.assertRaisesRegex(ValueError, '第二形状'):
            self.verify()

    def test_rejects_numerical_mismatch(self):
        self.record['cases'][0]['executions'][0]['mismatch'] = 1
        with self.assertRaisesRegex(ValueError, '数值'):
            self.verify()

    def test_rejects_changed_product(self):
        self.record['installed_after'] = 'bad'
        with self.assertRaisesRegex(ValueError, '安装态源码'):
            self.verify()

    def test_rejects_candidate_in_installed_run(self):
        self.record['candidate'] = {'candidate_sha256': validation.CANDIDATE}
        with self.assertRaisesRegex(ValueError, '不得加载候选'):
            self.verify()

    def test_rejects_unexercised_path(self):
        self.record['emitted'][0]['bounded_load'] = False
        with self.assertRaisesRegex(ValueError, '未实际执行'):
            self.verify()

    def test_rejects_modified_generated_evidence(self):
        (self.root / 'emitted-000.txt').write_text('modified')
        with self.assertRaisesRegex(ValueError, '哈希不符'):
            self.verify()

    def test_rejects_wrong_backend(self):
        self.record['backend'] = 'default'
        with self.assertRaisesRegex(ValueError, '后端'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
