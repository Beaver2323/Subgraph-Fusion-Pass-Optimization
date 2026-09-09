"""T-086零设备合同测试；不得导入torch。"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


validator = module("t086_validator_test", "scripts/validate_t086.py")
worker = module("t086_worker_test", "runners/t086_performance_worker.py")


class T086PreparationTests(unittest.TestCase):
    def test_complete_static_contract(self):
        counts = validator.validate()
        self.assertEqual(counts["acceptance_units"], 1)
        self.assertEqual(counts["cases"], 2)
        self.assertEqual(counts["variants"], 2)

    def test_worker_gate_rejects_product_disable_and_wrong_backend(self):
        with tempfile.TemporaryDirectory(dir="/home/z50063656/tmp") as temporary:
            root = Path(temporary)
            gpu = {
                "status": "valid-reference-suite",
                "suite_valid": True,
                "cases": [
                    {
                        "acceptance_unit_id": worker.UNIT_ID,
                        "reference_valid": True,
                    }
                ],
            }
            functional = {
                "task_id": worker.TASK_ID,
                "acceptance_unit_id": worker.UNIT_ID,
                "backend": "triton_experimental",
                "pytorch_commit": worker.COMMIT,
                "correctness": "passed",
                "numerical_execution": True,
                "target_rewrite": "confirmed",
                "graph_breaks": 0,
                "fallbacks": 0,
                "fallback_scope": "graph-break-or-CPU-fallback-only",
                "allowed_device_lowering": "registered-IndexPutFallback-extern",
                "product_disabled": False,
                "measurement_workload": worker.WORKLOAD_ID,
                "world_size": 1,
                "input_spec": worker.input_spec(),
                "worker_sha256": hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest(),
                "source_files": {str(Path(worker.__file__)): hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest()},
                "arms": {
                    "off": {"target_rewrite": "disabled-control"},
                    "on": {"target_rewrite": "confirmed"},
                },
            }
            for name, value in (("gpu.json", gpu), ("functional.json", functional)):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            gate = {
                "task_id": worker.TASK_ID,
                "acceptance_unit_id": worker.UNIT_ID,
                "backend": "triton_experimental",
                "pytorch_commit": worker.COMMIT,
                "correctness": "passed",
                "target_rewrite": "confirmed",
                "graph_breaks": 0,
                "fallbacks": 0,
                "fallback_scope": "graph-break-or-CPU-fallback-only",
                "allowed_device_lowering": "registered-IndexPutFallback-extern",
                "product_disabled": False,
                "measurement_workload": worker.WORKLOAD_ID,
                "world_size": 1,
                "input_spec": worker.input_spec(),
                "worker_sha256": hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest(),
                "reviewed_at": "2026-09-09T00:35:07+08:00",
                "reviewer": "test-fixture",
                "gpu_reference": {"path": "gpu.json", "sha256": hashlib.sha256((root / "gpu.json").read_bytes()).hexdigest()},
                "target_functional": {"path": "functional.json", "sha256": hashlib.sha256((root / "functional.json").read_bytes()).hexdigest()},
            }
            gate_path = root / "gate.json"
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            worker.read_gate(gate_path, "npu")
            for key, value in (("product_disabled", True), ("backend", "default")):
                broken = copy.deepcopy(gate)
                broken[key] = value
                gate_path.write_text(json.dumps(broken), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    worker.read_gate(gate_path, "npu")

    def test_cpu_only_candidates_never_enter_reference_cases(self):
        manifest = json.loads((ROOT / "upstream/t086_manifest.yaml").read_text())
        reference = json.loads((ROOT / "upstream/t086_reference_plan.yaml").read_text())
        executed = {case["acceptance_unit_id"] for case in reference["cases"]}
        deferred = {
            item["provisional_unit_id"]
            for item in manifest["deferred_candidates"]
        }
        self.assertEqual(executed, {validator.UNIT})
        self.assertTrue(validator.DEFERRED.issubset(deferred))


if __name__ == "__main__":
    unittest.main()
