"""T-085零设备准备门禁。"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PYTORCH = Path("/home/z50063656/Pass/src/pytorch")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load_module("test_t085_validator", ROOT / "scripts/validate_t085_preparation.py")
worker = load_module("test_t085_worker", ROOT / "runners/t085_performance_worker.py")
aggregator = load_module("test_t085_aggregator", ROOT / "scripts/aggregate_t085_performance.py")
orchestrator = load_module("test_t085_orchestrator", ROOT / "scripts/run_t085_performance.py")


class T085PreparationTests(unittest.TestCase):
    def test_full_contract_validates_without_torch_import(self):
        before = set(sys.modules)
        counts = validator.validate(ROOT, PYTORCH)
        imported = set(sys.modules) - before
        self.assertEqual(counts["units"], 3)
        self.assertEqual(counts["cases"], 5)
        self.assertEqual(counts["deferred"], 2)
        self.assertFalse(any(name == "torch" or name.startswith("torch.") for name in imported))

    def test_partitioned_scatter_is_not_executable_performance_target(self):
        plan = json.loads((ROOT / "upstream/t085_performance_plan.yaml").read_text())
        scatter = next(
            unit
            for unit in plan["acceptance_units"]
            if unit["acceptance_unit_id"] == "AU-post-grad-partitioned-scatter-optimization"
        )
        self.assertIsNone(scatter["worker_unit"])
        self.assertIn("capability-pending", scatter["performance_status"])
        self.assertNotIn("partitioned-scatter", worker.TARGETS)

    def test_backend_is_selected_before_import(self):
        source = (ROOT / "runners/t085_performance_worker.py").read_text()
        select = source.index('os.environ["TORCHINDUCTOR_NPU_BACKEND"]')
        import_torch = source.index("    import torch\n", select)
        self.assertLess(select, import_torch)
        self.assertIn('!= "triton_experimental"', source)

    def test_reference_is_native_only_and_two_gpu_suite(self):
        plan = json.loads((ROOT / "upstream/t085_reference_plan.yaml").read_text())
        self.assertEqual(plan["execution_policy"]["minimum_gpus"], 2)
        self.assertTrue(all(case["tracking_mode"] == "direct" for case in plan["cases"]))
        script = (ROOT / "scripts/run_t085_gpu_all.sh").read_text()
        self.assertIn("--gpus", script)
        self.assertIn("tracker_gpu_acquire", script)
        self.assertIn('export CUDA_VISIBLE_DEVICES="${gpu_a},${gpu_b}"', script)
        shared = (ROOT / "scripts/run_gpu_reference_task.sh").read_text()
        self.assertIn("T-085|T085", shared)
        self.assertIn("run_t085_gpu_all.sh", shared)
        wrapper = (ROOT / "scripts/run_t085_reference_all.sh").read_text()
        self.assertIn('--pytorch-root', wrapper)
        self.assertIn('validator+=(--pytorch-root "${pytorch_root}")', wrapper)

    def test_aggregator_rejects_missing_arms(self):
        with self.assertRaisesRegex(ValueError, "缺少off1"):
            aggregator.aggregate(Path("/home/z50063656/tmp/nonexistent-t085-test"))

    def test_gate_rejects_unsigned_or_wrong_backend(self):
        unsigned = {
            "task_id": "T-085",
            "acceptance_unit_id": "AU-post-grad-pointless-cumsum",
            "backend": "default",
        }
        path = Path("/home/z50063656/tmp/t085-unsigned-test.json")
        path.write_text(json.dumps(unsigned), encoding="utf-8")
        self.addCleanup(path.unlink, missing_ok=True)
        with self.assertRaises(ValueError):
            worker.read_gate(path, "pointless-cumsum", "npu")

    def test_timeout_only_terminates_owned_process_group(self):
        class FakeProcess:
            pid = 123456

            def __init__(self):
                self.calls = 0

            def wait(self, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("owned-t085", timeout)
                return -15

        child = FakeProcess()
        with patch.object(
            orchestrator.subprocess, "Popen", return_value=child
        ) as start, patch.object(orchestrator.os, "killpg") as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                orchestrator.run_arm(
                    ["owned-t085"],
                    Path("/home/z50063656/tmp"),
                    None,
                    None,
                    timeout=1,
                )
            self.assertTrue(start.call_args.kwargs["start_new_session"])
            kill.assert_called_once_with(child.pid, orchestrator.signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
