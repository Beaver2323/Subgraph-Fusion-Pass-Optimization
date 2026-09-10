"""T-087～T-090 零设备准备合同；不得导入torch。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PYTORCH = Path("/home/z50063656/Pass/src/pytorch")


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load("t087_t090_reference", "runners/reference_runner.py")
worker = load("t087_t090_worker", "runners/t087_t090_performance_worker.py")
launcher = load("t087_t090_launcher", "scripts/run_t087_t090_performance.py")


class T087T090PreparationTests(unittest.TestCase):
    def test_all_reference_contracts_resolve_statically(self):
        expected = {87: (2, 3, 3), 88: (2, 3, 3), 89: (1, 1, 1), 90: (1, 1, 1)}
        for number, counts in expected.items():
            manifest = json.loads((ROOT / f"upstream/t0{number}_manifest.yaml").read_text())
            plan = json.loads((ROOT / f"upstream/t0{number}_reference_plan.yaml").read_text())
            actual = reference.validate_contract(manifest, plan, PYTORCH)
            self.assertEqual(
                (actual["acceptance_units"], actual["cases"], actual["variants"]), counts
            )

    def test_generic_device_instantiation_resolves_reorder_cuda_class(self):
        path = PYTORCH / "test/inductor/test_reorder_for_locality_in_training.py"
        self.assertIn(
            "TestReorderForLocalityInTrainingCUDA."
            "test_training_flag_reorders_and_preserves_semantics_cuda",
            reference.python_qualnames(path),
        )

    def test_worker_separates_five_legal_off_on_units_from_functional_only(self):
        self.assertEqual(worker.BENCHMARK_UNITS, {
            "reorder-locality", "split-cat-aten", "select-cat-aten",
            "move-view-after-cat", "normalize-cat-aten",
        })
        self.assertIn("respecialize-current-device", worker.TARGETS)
        self.assertIn(
            "respecialize-current-device", launcher.TASK_FUNCTIONAL_UNITS["T-087"]
        )
        self.assertNotIn("respecialize-current-device", launcher.TASK_UNITS["T-087"])
        self.assertEqual(launcher.ORDER, (
            ("off", 1), ("on", 1), ("on", 2),
            ("off", 2), ("off", 3), ("on", 3),
        ))

    def test_backend_and_tmp_guards_precede_torch_import(self):
        source = (ROOT / "runners/t087_t090_performance_worker.py").read_text()
        self.assertLess(
            source.index('os.environ["TORCHINDUCTOR_NPU_BACKEND"]'),
            source.index("    import torch\n"),
        )
        self.assertIn('"triton_experimental"', source)
        self.assertIn("必须从", source)

    def test_incoming_and_guides_exist(self):
        for number in range(87, 91):
            self.assertTrue((ROOT / f"results/incoming/T-0{number}/README.md").is_file())
            text = (ROOT / f"docs/T0{number}_FUNCTION_PERFORMANCE_GUIDE.md").read_text()
            self.assertIn("更新时间", text)
            self.assertIn("triton_experimental", text)


if __name__ == "__main__":
    unittest.main()
