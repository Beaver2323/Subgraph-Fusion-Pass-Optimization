"""已签性能gate使用原执行快照；不执行快照或导入torch。"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_t102_t107_attention_performance import frozen_benchmark_worker, WORKER


class FrozenWorkerTests(unittest.TestCase):
    def test_verified_snapshot_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root/'issues/CASE/evidence/run'
            directory.mkdir(parents=True)
            gate = {}
            for key,name in [('worker_snapshot',WORKER.name),
                             ('target_control_snapshot','target_entry_control.py'),
                             ('observer_snapshot','native_contract_observer.py')]:
                path = directory/name
                path.write_text('# never execute\n')
                gate[key] = dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            gate_path = root/'gate.json'
            gate_path.write_text(json.dumps(gate))
            self.assertEqual(frozen_benchmark_worker(gate_path,root), directory/WORKER.name)
            (directory/'native_contract_observer.py').write_text('tampered')
            with self.assertRaises(ValueError):
                frozen_benchmark_worker(gate_path,root)

    def test_missing_snapshot_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'gate.json'
            path.write_text('{}')
            with self.assertRaises(KeyError):
                frozen_benchmark_worker(path,Path(temp))


if __name__ == '__main__':
    unittest.main()
