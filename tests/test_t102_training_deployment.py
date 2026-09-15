"""注册部署门禁的反篡改检查，合成数据不代表设备执行。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'scripts/validate_t102_training_deployment.py'
spec = importlib.util.spec_from_file_location('training_deployment', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrainingDeploymentTests(unittest.TestCase):
    def fixture(self, root, mutate=None):
        def write(name, value):
            path=root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value if isinstance(value,str) else json.dumps(value))
            return {'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        helper=write('after.py','# synthetic helper\n')
        before=write('before-init.py','def _activate():\n    pass\n')
        after=write('after-init.py','def _activate():\n    pass\n    from .sfdp_training import install_sfdp_training_patterns\n    install_sfdp_training_patterns()\n')
        prefix='/synthetic/torch_npu/_inductor/triton_experimental/'
        d=dict(task_id='T-102', backend='triton_experimental', deployed=True,
               pytorch_modified=False, product_disable_bypassed=False, candidate_sha256=helper['sha256'],
               source_files={'__init__.py':dict(target=prefix+'__init__.py',before=before,after=after),
                             'sfdp_training.py':dict(target=prefix+'sfdp_training.py',before=None,after=helper)})
        for index, mode in enumerate(('candidate','installed')):
            d[mode+'_cases']={}
            for n,(count,exact) in module.COUNTS.items():
                r=dict(status='community-contract-passed', tests_ran=1, tests_skipped=0,
                       native_assertions_passed=True, numerical_assertions_modified=False,
                       isolated_registration_candidate=mode=='candidate', isolated_codegen_candidate=False,
                       product_gate_bypassed=False, pytorch_commit=module.COMMIT, backend='triton_experimental',
                       physical_npu='5', test_errors=[], test_failures=[], case_id=f'REF-sfdp-pattern-{n}-native',
                       tensor_assertions=[dict(passed=True,actual_device='npu:0',expected_device='npu:0') for _ in range(count)],
                       exact_target_observations=exact, pid=index*10+n,
                       observations=[dict(graph_changed=True,target=f'_sfdp_pattern_{n}_training') for _ in range(exact)],
                       candidate_snapshot_sha256=helper['sha256'],
                       loaded_source_sha256={x['target']:x['after']['sha256'] for x in d['source_files'].values()})
                if mutate:
                    mutate('case',mode,n,r)
                d[mode+'_cases'][str(n)]=write(f'{mode}-{n}.json',r)
            harness=write(f'{mode}/harness_source.py','# synthetic boundary\n')
            write(f'{mode}/candidate_source.py','# synthetic helper\n')
            boundary=dict(status='passed',candidate=mode=='candidate',backend='triton_experimental',
                          pytorch_commit=module.COMMIT,product_sha256=helper['sha256'],physical_npu='5',
                          pid=index*10+9,harness_sha256=harness['sha256'],
                          scenarios=[dict(name=name,passed=True,target=module.POSITIVE_TARGETS.get(name),
                            actual_targets=[f'_sfdp_pattern_{module.POSITIVE_TARGETS[name]}_training']
                                if name in module.POSITIVE_TARGETS else []) for name in sorted(module.SCENARIOS)])
            if mutate:
                mutate('boundary',mode,0,boundary)
            d[mode+'_boundary']=write(f'{mode}/result.json',boundary)
        return write('deployment.json',d)

    def test_complete_synthetic_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertTrue(module.verify(root,self.fixture(root))['installed_passed'])

    def test_rejects_candidates_false_counts_missing_boundary_and_target_drift(self):
        corruptions=[('case',lambda r:r.update(isolated_registration_candidate=True)),
                     ('case',lambda r:r.update(exact_target_observations=0)),
                     ('case',lambda r:r.update(test_failures=['real failure'])),
                     ('case',lambda r:r.update(loaded_source_sha256={})),
                     ('boundary',lambda r:r.update(scenarios=r['scenarios'][:-1])),
                     ('boundary',lambda r:r['scenarios'][0].update(actual_targets=['_sfdp_pattern_1_training']))]
        for kind, change in corruptions:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                def mutate(actual,mode,n,record):
                    if actual==kind and mode=='installed':
                        change(record)
                root=Path(tmp)
                with self.assertRaises(ValueError):
                    module.verify(root,self.fixture(root,mutate))

    def test_rejects_changed_bound_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); reference=self.fixture(root)
            (root/'after.py').write_text('changed')
            with self.assertRaises(ValueError):
                module.verify(root,reference)
