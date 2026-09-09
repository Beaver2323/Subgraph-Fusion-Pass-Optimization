"""T-084零设备准备门禁；不导入torch。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PYTORCH = Path("/home/z50063656/Pass/src/pytorch")


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load_module("t084_reference", "runners/reference_runner.py")
worker = load_module("t084_worker", "runners/t084_performance_worker.py")


class T084PreparationTests(unittest.TestCase):
    def test_mapping_freezes_only_legal_off_on_unit(self):
        manifest = json.loads((ROOT / "upstream/t084_manifest.yaml").read_text())
        selected = {
            item["acceptance_unit_id"] for item in manifest["acceptance_units"]
        }
        deferred = {
            item["provisional_unit_id"] for item in manifest["deferred_candidates"]
        }
        self.assertEqual(selected, {"AU-post-grad-dedup-reduce-scatters"})
        self.assertEqual(
            deferred,
            {
                "AU-post-grad-decompose-auto-functionalized",
                "AU-post-grad-decompose-map-to-while-loop",
                "AU-post-grad-decompose-scan-to-while-loop",
                "AU-post-grad-decompose-triton-kernel-wrapper-functional",
            },
        )
        self.assertEqual(len(selected | deferred), 5)

    def test_native_reference_contract_resolves_without_torch(self):
        manifest = json.loads((ROOT / "upstream/t084_manifest.yaml").read_text())
        plan = json.loads((ROOT / "upstream/t084_reference_plan.yaml").read_text())
        counts = reference.validate_contract(manifest, plan, PYTORCH)
        self.assertEqual(counts["acceptance_units"], 1)
        self.assertEqual(counts["community_tests"], 1)
        self.assertEqual(counts["variants"], 1)
        command = reference.case_command(plan["cases"][0], ROOT, PYTORCH)
        self.assertEqual(
            command[-1],
            "TestCollectivesInductor.test_dedup_reduce_scatter",
        )

    def test_performance_gate_binds_originals_and_worker(self):
        with tempfile.TemporaryDirectory(dir="/home/z50063656/tmp") as temporary:
            directory = Path(temporary)
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
            worker_digest = hashlib.sha256(
                (ROOT / "runners/t084_performance_worker.py").read_bytes()
            ).hexdigest()
            functional = {
                "task_id": worker.TASK_ID,
                "acceptance_unit_id": worker.UNIT_ID,
                "backend": "triton_experimental",
                "pytorch_commit": worker.COMMIT,
                "correctness": "passed",
                "target_rewrite": "confirmed",
                "graph_breaks": 0,
                "fallbacks": 0,
                "product_disabled": False,
                "measurement_workload": worker.WORKLOAD,
                "world_size": 2,
                "input_spec": worker.input_spec(),
                "worker_sha256": worker_digest,
                "numerical_execution": True,
                "process_group_backend": "hccl",
                "source_files": {
                    str(ROOT / "runners/t084_performance_worker.py"): worker_digest
                },
            }
            gate = dict(
                functional,
                reviewed_at="2026-09-09T00:34:29+08:00",
                reviewer="test-fixture-only",
            )
            for name, content in (
                ("gpu_reference", gpu),
                ("target_functional", functional),
            ):
                path = directory / f"{name}.json"
                path.write_text(json.dumps(content))
                gate[name] = {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            gate_path = directory / "gate.json"
            gate_path.write_text(json.dumps(gate))
            self.assertEqual(
                worker.read_gate(gate_path, "npu")["backend"],
                "triton_experimental",
            )

    def test_one_click_and_six_arm_contracts_are_wired(self):
        gpu_launcher = (ROOT / "scripts/run_gpu_reference_task.sh").read_text()
        task_launcher = (ROOT / "scripts/run_t084_reference_all.sh").read_text()
        performance = (ROOT / "scripts/run_t084_performance.py").read_text()
        self.assertIn("T-084|T084", gpu_launcher)
        self.assertIn("--task T-084", task_launcher)
        self.assertIn(
            '("off1", "on1", "on2", "off2", "off3", "on3")',
            performance,
        )


if __name__ == "__main__":
    unittest.main()
