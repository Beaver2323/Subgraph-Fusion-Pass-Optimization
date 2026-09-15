"""窄部署校验不把已部署当作完成；只读测试不加载产品或设备。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import validate_attention_slice_deployment as deployment


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        before = 'class NPUTritonKernel:\n'
        after = before
        for name in sorted(deployment.METHODS):
            before += f'    def {name}(self):\n        return 0\n'
            after += f'    def {name}(self):\n        return 1\n'
        (self.root / 'before.py').write_text(before)
        (self.root / 'after.py').write_text(after)
        (self.root / 'pre.json').write_text('{}')
        (self.root / 'candidate').mkdir()
        for name in deployment.METHODS:
            (self.root / 'candidate' / f'after-{name}.py').write_text(f'def {name}(self):\n    return 1\n')
        self.before = deployment.sha(self.root / 'before.py')
        self.after = deployment.sha(self.root / 'after.py')
        self.record = dict(task_id='T-106', backend='triton_experimental',
            acceptance_unit_id='AU-fuse-attention-sfdp-pattern-22', deployed=True,
            pytorch_modified=False, product_disable_bypassed=False, performance_gate_issued=False,
            backup='before.py', installed_snapshot='after.py', before_sha256=self.before,
            after_sha256=self.after, candidate_sha256=deployment.CANDIDATE,
            changed_methods=sorted(deployment.METHODS), candidate_boundary_result='pre.json',
            candidate_boundary_result_sha256=deployment.sha(self.root / 'pre.json'),
            candidate_boundary_runner_sha256='fixture',
            verification_status='pending-fresh-installed-original-neighbors-boundary')

    def verify(self):
        p = self.root / 'deployment.json'; p.write_text(json.dumps(self.record))
        with patch.object(deployment, 'BEFORE', self.before), patch.object(deployment, 'AFTER', self.after), \
                patch.object(deployment, 'verify_boundary') as check:
            result = deployment.verify(self.root, dict(path=p.name, sha256=deployment.sha(p)))
            self.assertEqual(check.call_count, 2 if result['installed_passed'] else 1)
            return result

    def completed_fixture(self):
        self.record.update(verification_status='passed-original-and-three-neighbors-and-boundary',
                           target='/installed/triton.py', installed_cases={})
        for number, tensors, rewrites in ((21, 2, 2), (22, 12, 4), (23, 6, 2), (24, 1, 1)):
            r = dict(status='community-contract-passed', tests_ran=1, tests_skipped=0,
                     case_id=f'REF-sfdp-pattern-{number}-native', task_id='T-106',
                     backend='triton_experimental', pytorch_commit=deployment.COMMIT,
                     native_assertions_passed=True, numerical_assertions_modified=False,
                     isolated_codegen_candidate=False, isolated_registration_candidate=False,
                     product_gate_bypassed=False, pid=number,
                     loaded_source_sha256={'/installed/triton.py': self.after},
                     tensor_assertions=[dict(passed=True, actual_device='npu:0',
                                             expected_device='npu:0') for _ in range(tensors)],
                     exact_target_observations=rewrites)
            p = self.root / f'{number}.json'; p.write_text(json.dumps(r))
            self.record['installed_cases'][str(number)] = dict(path=p.name, sha256=deployment.sha(p))
        p = self.root / 'post.json'; p.write_text(json.dumps(dict(pid=99)))
        self.record.update(installed_boundary_result=dict(path=p.name, sha256=deployment.sha(p)),
                           installed_boundary_runner_sha256='installed-fixture')

    def change_installed(self, number, **updates):
        p = self.root / f'{number}.json'
        r = json.loads(p.read_text()); r.update(updates); p.write_text(json.dumps(r))
        self.record['installed_cases'][str(number)]['sha256'] = deployment.sha(p)

    def test_complete_chain_passes_both_boundary_checks(self):
        self.completed_fixture()
        self.assertEqual(self.verify(), dict(deployed=True, installed_passed=True))

    def test_candidate_cannot_replace_installed_original(self):
        self.completed_fixture(); self.change_installed(22, isolated_codegen_candidate=True)
        with self.assertRaisesRegex(ValueError, '无候选安装态'):
            self.verify()

    def test_installed_runs_must_use_fresh_processes(self):
        self.completed_fixture(); self.change_installed(24, pid=22)
        with self.assertRaisesRegex(ValueError, '复用了进程'):
            self.verify()

    def test_stale_installed_source_is_rejected(self):
        self.completed_fixture()
        self.change_installed(22, loaded_source_sha256={'/installed/triton.py': self.before})
        with self.assertRaisesRegex(ValueError, '实际部署文件'):
            self.verify()

    def test_pending_deployment_is_not_installed_pass(self):
        self.assertEqual(self.verify(), dict(deployed=True, installed_passed=False))

    def test_changed_non_target_code_is_rejected(self):
        p = self.root / 'after.py'; p.write_text(p.read_text()+'outside = 1\n')
        self.after = self.record['after_sha256'] = deployment.sha(p)
        with self.assertRaisesRegex(ValueError, '两个方法以外'):
            self.verify()

    def test_deployed_methods_must_equal_tested_candidate(self):
        p = self.root / 'candidate/after-_maybe_record_select_lane_load.py'; p.write_text('different')
        with self.assertRaisesRegex(ValueError, '实测候选不同'):
            self.verify()

    def test_completed_claim_without_regressions_is_rejected(self):
        self.record.update(verification_status='passed-original-and-three-neighbors-and-boundary', installed_cases={})
        with self.assertRaisesRegex(ValueError, '原例或邻接缺失'):
            self.verify()

    def test_wrong_backend_is_rejected(self):
        self.record['backend'] = 'default'
        with self.assertRaisesRegex(ValueError, '范围不符'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
