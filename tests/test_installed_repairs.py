"""安装态证据零设备回归；候选、哈希篡改、跨后端不得混入完成态。"""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('installed_repairs', ROOT/'scripts/validate_installed_repairs.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class InstalledRepairTests(unittest.TestCase):
    def test_archived_chain_without_torch_or_live_install(self):
        self.assertEqual(audit.check()['original_contracts'], 4)

    def test_candidate_skip_backend_and_gate_bypass_rejected(self):
        good = dict(backend='triton_experimental', pytorch_commit=audit.COMMIT,
                    product_candidate=None, product_gate_bypassed=False, tests_ran=1, tests_skipped=0)
        audit.check_contract(good)
        for key, value in [('backend', 'default'), ('pytorch_commit', 'wrong'), ('product_candidate', '/candidate'),
                           ('product_gate_bypassed', True), ('tests_ran', 0), ('tests_skipped', 1)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.check_contract(dict(good, **{key:value}))

    def test_hash_and_path_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'raw.txt'
            path.write_text('原件')
            reference = dict(path='raw.txt', sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            audit.check_refs(reference, root)
            for change in ({'sha256':'0'*64}, {'path':str(path)}, {'path':'../outside'}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    audit.check_refs(dict(reference, **change), root)

    def test_formal_fix_requires_numerical_execution_and_change(self):
        row = audit.load(ROOT/'results/current/T-096/functional/e8m0-rceil-log2.json')
        for key, value in [('numerical_execution', False), ('target_rewrite', 'pending'),
                           ('repair_status', 'isolated-source-candidate-passed-not-deployed')]:
            bad = copy.deepcopy(row)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.check_common(bad)

    def test_timing_recomputed_from_samples(self):
        path = next((ROOT/'issues/REF-e8m0-log2-pattern-native/evidence').glob('performance-*-off1/artifacts/result.json'))
        row = audit.load(path)
        audit.check_timing(row)
        row['timing']['event_ms']['p99'] *= .5
        with self.assertRaises(ValueError):
            audit.check_timing(row)

    def test_nonfinite_samples_rejected(self):
        path = next((ROOT/'issues/REF-e8m0-log2-pattern-native/evidence').glob('performance-*-on1/artifacts/result.json'))
        row = audit.load(path)
        row['samples']['host_ms'][0] = float('nan')
        with self.assertRaises(ValueError):
            audit.check_timing(row)
