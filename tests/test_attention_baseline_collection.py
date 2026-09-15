"""安装态基线收集只读原件，不接受候选、异后端、错源码或非NPU数值。"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import record_attention_baseline_review as collector


class BaselineCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=collector.WORK)
        self.addCleanup(self.temp.cleanup)
        base=Path(self.temp.name);self.root=base/'repo';self.raw=base/'raw/adapter'
        self.raw.mkdir(parents=True)
        self.case='REF-sfdp-pattern-2-native'
        self.parent_path=self.root/'issues'/self.case/'adapter_runs/run/run_result.json'
        source_sha=hashlib.sha256(b'frozen').hexdigest()
        self.parent={'case_id':self.case,'task_id':'T-102','mode':'adapter',
                     'attention_registration_candidate':False,'return_code':0,
                     'installed_product_before':{'snapshot':'same'},'installed_product_after':{'snapshot':'same'},
                     'raw_artifact_dir':str(self.raw.parent),'source_sha256':source_sha}
        self.record={'case_id':self.case,'task_id':'T-102','isolated_registration_candidate':False,
                     'backend':'triton_experimental','pytorch_commit':'8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b',
                     'product_gate_bypassed':False,'numerical_assertions_modified':False,
                     'source_test':'test/inductor/test_fused_attention.py::Test.test_case','source_sha256':source_sha,
                     'harness_snapshot_sha256':hashlib.sha256(b'harness').hexdigest(),
                     'tensor_assertions':[],'status':'community-contract-passed','tests_ran':1,'tests_skipped':0,
                     'native_assertions_passed':True}
        (self.raw/'harness_source.py').write_bytes(b'harness')
        self.write(self.root/'results/current/T-102/gpu_reference_review.json',
                   {'cases':[{'case_id':self.case,'target_review':{'status':'gpu-contract-reviewed-awaiting-npu'}}]})

    def write(self,path,data):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data))

    def collect(self):
        self.write(self.parent_path,self.parent);self.write(self.raw/'result.json',self.record)
        with patch.object(collector.subprocess,'check_output',return_value=b'frozen'), \
             patch.object(collector,'inspect',return_value={'output_code_count':1}):
            return collector.collect(self.parent_path,self.root)

    def test_zero_numerical_oracle_is_not_invented(self):
        result=self.collect()
        self.assertTrue(result[-2])  # 只表示原社区执行/结构合同，不增加数值证据。
        self.assertEqual(result[2]['tensor_assertions'],[])

    def test_candidate_cannot_replace_installed_baseline(self):
        self.parent['attention_registration_candidate']=True
        with self.assertRaises(ValueError):self.collect()

    def test_codegen_candidate_cannot_replace_installed_baseline(self):
        self.parent['attention_select_slice_candidate']=True
        with self.assertRaises(ValueError):self.collect()
        self.parent['attention_select_slice_candidate']=False
        self.record['isolated_codegen_candidate']=True
        with self.assertRaises(ValueError):self.collect()

    def test_wrong_backend_and_source_are_rejected(self):
        self.record['backend']='default'
        with self.assertRaises(ValueError):self.collect()
        self.record['backend']='triton_experimental';self.record['source_sha256']='0'*64
        with self.assertRaises(ValueError):self.collect()

    def test_cpu_comparison_cannot_count_as_npu(self):
        self.record['tensor_assertions']=[{'passed':True,'actual_device':'npu:0','expected_device':'cpu'}]
        with self.assertRaises(ValueError):self.collect()

    def test_changed_harness_rejected(self):
        (self.raw/'harness_source.py').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.collect()

    def test_supersession_requires_explicit_flag_and_keeps_history(self):
        old={'candidate_verified':False,'performance_gate_issued':False,'run':'old'}
        new=dict(old,run='new')
        with self.assertRaisesRegex(ValueError,'supersede'):
            collector.preserve_previous_stage(self.root,'T-102','AU-test',old,new,False)
        collector.preserve_previous_stage(self.root,'T-102','AU-test',old,new,True)
        collector.preserve_previous_stage(self.root,'T-102','AU-test',old,new,True)
        snapshots=list((self.root/'results/history/T-102').glob('*.json'))
        self.assertEqual(len(snapshots),1)
        self.assertEqual(json.loads(snapshots[0].read_text()),old)

    def test_verified_candidate_cannot_be_overwritten_by_baseline(self):
        old={'candidate_verified':True,'performance_gate_issued':False}
        with self.assertRaisesRegex(ValueError,'候选验证'):
            collector.preserve_previous_stage(self.root,'T-102','AU-test',old,{},True)
