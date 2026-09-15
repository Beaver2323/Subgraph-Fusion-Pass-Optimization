import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from npu_stage_review import verify


class StageReviewTests(unittest.TestCase):
    def fixture(self,root,mode,**extra):
        data=dict(task_id='T-102',acceptance_unit_id='AU',case_id='REF',backend='triton_experimental',
                  pytorch_commit='8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b',product_gate_bypassed=False,
                  isolated_registration_candidate=mode=='candidate',status='failed-or-target-missing',
                  tests_ran=1,tests_skipped=0,native_assertions_passed=False)
        data.update(extra)
        path=root/f'{mode}.json';path.write_text(json.dumps(data))
        return dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def review(self,root):
        return dict(task_id='T-102',acceptance_unit_id='AU',case_id='REF',backend='triton_experimental',
                    performance_gate_issued=False,candidate_verified=False,baseline=self.fixture(root,'baseline'))

    def test_failed_baseline_stays_failed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.review(root)
            self.assertFalse(verify(root,'T-102','AU',r)['baseline_passed'])

    def test_candidate_never_becomes_installed_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.review(root)
            r['candidate']=self.fixture(root,'candidate',status='community-contract-passed',native_assertions_passed=True)
            r['candidate_verified']=True
            state=verify(root,'T-102','AU',r)
            self.assertTrue(state['candidate_passed']);self.assertFalse(state['baseline_passed'])

    def test_candidate_cannot_replace_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.review(root);r['baseline']=self.fixture(root,'candidate')
            with self.assertRaisesRegex(ValueError,'冒充'):
                verify(root,'T-102','AU',r)

    def test_codegen_candidate_cannot_replace_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.review(root)
            r['baseline']=self.fixture(root,'baseline',isolated_codegen_candidate=True)
            with self.assertRaisesRegex(ValueError,'冒充'):
                verify(root,'T-102','AU',r)

    def test_codegen_candidate_requires_explicit_kind_and_stays_isolated(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.review(root)
            r['candidate']=self.fixture(root,'candidate',isolated_registration_candidate=False,
                isolated_codegen_candidate=True,status='community-contract-passed',native_assertions_passed=True)
            r['candidate_verified']=True
            with self.assertRaisesRegex(ValueError,'类型不符'):
                verify(root,'T-102','AU',r)
            r['candidate_kind']='codegen'
            state=verify(root,'T-102','AU',r)
            self.assertTrue(state['candidate_passed']);self.assertFalse(state['baseline_passed'])

    def test_tampered_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.review(root);(root/'baseline.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'哈希'):
                verify(root,'T-102','AU',r)

    def test_skip_not_candidate_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.review(root)
            r['candidate']=self.fixture(root,'candidate',status='community-contract-passed',native_assertions_passed=True,tests_skipped=1)
            r['candidate_verified']=True
            with self.assertRaisesRegex(ValueError,'通过状态'):
                verify(root,'T-102','AU',r)

    def test_numerical_failure_requires_real_failure_stack(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.review(root)
            r['failure_layer']='numerical-contract'
            with self.assertRaisesRegex(ValueError,'数值失败分类'):
                verify(root,'T-102','AU',r)
            r['baseline']=self.fixture(root,'baseline',test_failures=['AssertionError: Tensor-likes are not close!'])
            self.assertFalse(verify(root,'T-102','AU',r)['baseline_passed'])
