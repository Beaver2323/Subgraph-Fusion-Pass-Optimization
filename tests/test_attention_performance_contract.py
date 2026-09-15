"""性能准备只能选择同编号正确阶段；不导入torch或假装已实测。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys

spec=importlib.util.spec_from_file_location('attention_worker_contract',
    Path(__file__).resolve().parents[1]/'runners/t102_t107_attention_performance_worker.py')
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class AttentionPerformanceContractTests(unittest.TestCase):
    def test_scale_placeholder_conversion_is_name_scoped(self):
        class Scalar:
            ndim = 0
            def item(self):
                return 3.0
        value = Scalar()
        self.assertEqual(worker.normalize_scale_argument('inv_scale', value), 3.0)
        self.assertEqual(worker.normalize_scale_argument('scale_factor', value), 3.0)
        self.assertEqual(worker.normalize_scale_argument('inv_scale_factor', value), 3.0)
        self.assertIs(worker.normalize_scale_argument('attn_mask', value), value)
        self.assertEqual(worker.input_description('inv_scale', 3.0),
                         dict(parameter='inv_scale',kind='python-scalar',python_type='float',value=3.0))

    def test_pattern19_uses_original_fp32_domain(self):
        candidates=[('_sfdp_pattern_19_half_inference',{}),('_sfdp_pattern_19_inference',{})]
        name,_,_,_=worker.select_registration(candidates,19)
        self.assertEqual(name,'_sfdp_pattern_19_inference')
        self.assertIn('fp32',worker.workload_for_pattern(19))
        with self.assertRaises(ValueError):
            worker.select_registration(candidates[:1],19)

    def gate_fixture(self, root, *, candidate=False, wrong_gpu=False, wrong_off=False):
        au='AU-fuse-attention-sfdp-pattern-1'
        common=dict(task_id='T-102',acceptance_unit_id=au,backend='triton_experimental',
                    pytorch_commit=worker.COMMIT,correctness='passed',numerical_execution=True,
                    target_rewrite='confirmed',graph_breaks=0,fallbacks=0,product_disabled=False,
                    measurement_workload=worker.workload_for_pattern(1),worker_sha256=worker.sha256(Path(worker.__file__)),
                    target_control_sha256=worker.sha256(Path(worker.__file__).with_name('target_entry_control.py')),
                    observer_sha256=worker.sha256(Path(worker.__file__).with_name('native_contract_observer.py')),
                    input_spec=[{'shape':[4,4],'requires_grad':False}],registration_name='_sfdp_pattern_1_half_inference',
                    dropout_p=0.0, pytorch_worktree_status='', backend_selected_before_import=True)
        payloads={
            'gpu_reference':dict(expected_pytorch_commit=worker.COMMIT,cases=[dict(acceptance_unit_id=au,
                tests_ran=1,tests_skipped=0,target_review=dict(expected_target='_sfdp_pattern_2' if wrong_gpu else '_sfdp_pattern_1',
                exact_target_observations=1))]),
            'target_functional':dict(common,mode='on',phase='functional',exact_pattern_counter=1,general_fuse_attention_counter=1),
            'off_functional':dict(common,mode='off',phase='functional',exact_pattern_counter=0,general_fuse_attention_counter=0),
            'community_functional':dict(status='community-contract-passed',tests_ran=1,tests_skipped=0,native_assertions_passed=True,
                isolated_registration_candidate=candidate,backend='triton_experimental',acceptance_unit_id=au,
                pytorch_commit=worker.COMMIT,product_gate_bypassed=False,exact_target_observations=1),
        }
        runtime = root/'runtime-source.py'
        runtime.write_text('# fixture, not a device result\n')
        for pid, name in enumerate(('target_functional','off_functional','community_functional'), 1001):
            payloads[name].update(pid=pid, python_executable=sys.executable,
                                  loaded_source_sha256={str(runtime):worker.sha256(runtime)})
        if wrong_off:
            payloads['off_functional']['input_spec']=[]
        gate=dict(common,reviewed_at='2026-09-14T21:49:00+08:00',reviewer='unit-test-fixture-not-device-evidence')
        for name,data in payloads.items():
            path=root/f'{name}.json'
            path.write_text(json.dumps(data))
            gate[name]={'path':path.name,'sha256':worker.sha256(path)}
        path=root/'gate.json';path.write_text(json.dumps(gate))
        return path

    def test_complete_bound_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            gate=self.gate_fixture(Path(temp))
            self.assertEqual(worker.read_gate(gate,1,'npu')['task_id'],'T-102')

    def test_isolated_candidate_cannot_sign_installed_performance(self):
        with tempfile.TemporaryDirectory() as temp:
            gate=self.gate_fixture(Path(temp),candidate=True)
            with self.assertRaisesRegex(ValueError,'隔离候选'):
                worker.read_gate(gate,1,'npu')

    def test_codegen_candidate_cannot_sign_installed_performance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);gate=self.gate_fixture(root)
            self.mutate_arm(root,gate,'community_functional',isolated_codegen_candidate=True)
            with self.assertRaisesRegex(ValueError,'隔离候选'):
                worker.read_gate(gate,1,'npu')

    def test_neighbor_gpu_target_cannot_sign_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            gate=self.gate_fixture(Path(temp),wrong_gpu=True)
            with self.assertRaisesRegex(ValueError,'本编号'):
                worker.read_gate(gate,1,'npu')

    def test_off_input_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            gate=self.gate_fixture(Path(temp),wrong_off=True)
            with self.assertRaisesRegex(ValueError,'input_spec'):
                worker.read_gate(gate,1,'npu')

    def mutate_arm(self, root, gate, name, **changes):
        path=root/f'{name}.json'
        data=json.loads(path.read_text());data.update(changes)
        path.write_text(json.dumps(data))
        meta=json.loads(gate.read_text());meta[name]['sha256']=worker.sha256(path)
        gate.write_text(json.dumps(meta))

    def test_same_process_control_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);gate=self.gate_fixture(root)
            self.mutate_arm(root,gate,'off_functional',pid=1001)
            with self.assertRaisesRegex(ValueError,'独立进程'):
                worker.read_gate(gate,1,'npu')

    def test_source_changed_after_functional_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);gate=self.gate_fixture(root)
            (root/'runtime-source.py').write_text('# changed\n')
            with self.assertRaisesRegex(ValueError,'源码变化'):
                worker.read_gate(gate,1,'npu')

    def test_missing_runtime_source_and_dirty_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);gate=self.gate_fixture(root)
            self.mutate_arm(root,gate,'off_functional',loaded_source_sha256={})
            with self.assertRaisesRegex(ValueError,'源码指纹'):
                worker.read_gate(gate,1,'npu')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);gate=self.gate_fixture(root)
            self.mutate_arm(root,gate,'off_functional',pytorch_worktree_status='modified')
            with self.assertRaisesRegex(ValueError,'真实功能臂'):
                worker.read_gate(gate,1,'npu')

    def test_dropout_cannot_silently_choose_zero_dropout_inference(self):
        registration={'scalar_workaround':{'dropout_p':0.113377}}
        candidates=[('_sfdp_pattern_3_half_inference',{'scalar_workaround':{}}),
                    ('_sfdp_pattern_3_half_training',registration)]
        name,chosen,values,training=worker.select_registration(candidates,3)
        self.assertEqual(name,'_sfdp_pattern_3_half_training')
        self.assertIs(chosen,registration)
        self.assertEqual(values['dropout_p'],1e-11)
        self.assertEqual(registration['scalar_workaround']['dropout_p'],0.113377)
        self.assertTrue(training)
        self.assertIn('training-forward',worker.workload_for_pattern(3))

    def test_missing_required_stage_rejected(self):
        with self.assertRaisesRegex(ValueError,'执行阶段'):
            worker.select_registration([('_sfdp_pattern_3_half_inference',{})],3)

    def test_normal_inference_is_not_replaced_by_training(self):
        candidates=[('_sfdp_pattern_13_half_training',{}),('_sfdp_pattern_13_half_inference',{})]
        name,_,values,training=worker.select_registration(candidates,13)
        self.assertTrue(name.endswith('_inference'))
        self.assertFalse(training)
        self.assertEqual(values,{})

    def test_repaired_training_contract_preserves_non_dropout_scalars(self):
        for number in (1,2,5):
            with self.subTest(number=number):
                original={'scalar_workaround':{'inv_scale':2.0}}
                candidates=[(f'_sfdp_pattern_{number}_half_inference',{}),
                            (f'_sfdp_pattern_{number}_half_training',original)]
                name,chosen,values,training=worker.select_registration(candidates,number)
                self.assertTrue(name.endswith('_training'))
                self.assertIs(chosen,original)
                self.assertTrue(training)
                self.assertEqual(values,{'inv_scale':2.0})
                self.assertEqual(worker.workload_for_pattern(number),f'sfdp-pattern-{number}-registered-half-training-forward')
                with self.assertRaisesRegex(ValueError,'执行阶段'):
                    worker.select_registration(candidates[:1],number)

    def test_dropout_parameter_is_required(self):
        with self.assertRaisesRegex(ValueError,'dropout参数'):
            worker.select_registration([('_sfdp_pattern_3_half_training',{})],3)

    def test_binary_mask_domain_not_random_float(self):
        for pattern in (15,18,19,20):
            self.assertTrue(worker.input_initialization_policy(pattern,3).startswith('community-triangular'))
            self.assertTrue(worker.input_initialization_policy(pattern,0).startswith('fixed-seed'))
        for pattern in (14,21,22,24):
            self.assertTrue(worker.input_initialization_policy(pattern,3).startswith('fixed-seed'))
