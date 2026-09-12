"""T-091～T-100 零设备准备合同；不得导入torch。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PYTORCH = Path("/home/z50063656/Pass/src/pytorch")


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load("t091_t100_reference", "runners/reference_runner.py")
worker = load("t091_t100_worker", "runners/t091_t100_performance_worker.py")
launcher = load("t091_t100_launcher", "scripts/run_t091_t100_performance.py")


class T091T100PreparationTests(unittest.TestCase):
    def test_all_reference_contracts_resolve_statically(self):
        expected = {
            91: (1, 1, 1),
            92: (0, 0, 0),
            93: (0, 0, 0),
            94: (0, 0, 0),
            95: (0, 0, 0),
            96: (1, 3, 3),
            97: (0, 0, 0),
            98: (1, 1, 3),
            99: (0, 0, 0),
            100: (1, 1, 2),
        }
        for number, counts in expected.items():
            manifest = json.loads(
                (ROOT / f"upstream/t{number:03d}_manifest.yaml").read_text()
            )
            plan = json.loads(
                (ROOT / f"upstream/t{number:03d}_reference_plan.yaml").read_text()
            )
            actual = reference.validate_contract(manifest, plan, PYTORCH)
            self.assertEqual(
                (actual["acceptance_units"], actual["cases"], actual["variants"]),
                counts,
            )

    def test_review_covers_all_37_original_candidates_without_reassignment(self):
        backlog = json.loads((ROOT / "upstream/task_backlog.json").read_text())
        batches = {
            batch["task_id"]: batch
            for batch in backlog["batches"]
            if 91 <= int(batch["task_id"][2:]) <= 100
        }
        selected = 0
        deferred = 0
        for task, batch in batches.items():
            manifest = json.loads(
                (ROOT / "upstream" / f"{task.lower().replace('-', '')}_manifest.yaml").read_text()
            )
            original = {unit["provisional_unit_id"] for unit in batch["units"]}
            ready = {
                unit["acceptance_unit_id"] for unit in manifest["acceptance_units"]
            }
            held = {
                unit["provisional_unit_id"]
                for unit in manifest["deferred_candidates"]
            }
            self.assertEqual(ready | held, original)
            self.assertFalse(ready & held)
            selected += len(ready)
            deferred += len(held)
        self.assertEqual((selected, deferred), (4, 33))

    def test_copy_tests_cuda_qualnames_and_newly_discovered_binary_case(self):
        efficient = reference.python_qualnames(
            PYTORCH / "test/inductor/test_efficient_conv_bn_eval.py"
        )
        binary = reference.python_qualnames(
            PYTORCH / "test/inductor/test_binary_folding.py"
        )
        self.assertIn("EfficientConvBNEvalGpuTests.test_basic_cuda", efficient)
        self.assertIn("FreezingGpuTests.test_linear_binary_folding_cuda", binary)

    def test_worker_only_contains_legal_prepared_off_on_units(self):
        self.assertEqual(
            set(worker.TARGETS),
            {"stack-normalization", "efficient-conv-bn", "linear-binary-folding"},
        )
        self.assertEqual(
            launcher.TASK_UNITS,
            {
                "T-091": ["stack-normalization"],
                "T-098": ["efficient-conv-bn"],
                "T-100": ["linear-binary-folding"],
            },
        )
        self.assertEqual(
            launcher.ORDER,
            (
                ("off", 1),
                ("on", 1),
                ("on", 2),
                ("off", 2),
                ("off", 3),
                ("on", 3),
            ),
        )
        t096 = json.loads((ROOT / "upstream/t096_performance_plan.yaml").read_text())
        self.assertEqual(t096["implementation"]["status"], "implemented-runtime-validated")
        self.assertEqual(t096["status"], "performance-disposition-complete")
        self.assertEqual(t096["acceptance_units"][0]["performance_status"], "measured")
        self.assertEqual(t096["acceptance_units"][0]["verdict"], "PERF_REGRESSED")

    def test_backend_and_tmp_guards_precede_torch_import(self):
        source = (ROOT / "runners/t091_t100_performance_worker.py").read_text()
        self.assertLess(
            source.index('os.environ["TORCHINDUCTOR_NPU_BACKEND"]'),
            source.index("    import torch\n"),
        )
        self.assertIn('"triton_experimental"', source)
        self.assertIn("必须从", source)

    def test_zero_unit_tasks_are_reviewable_but_not_runnable(self):
        for number in (92, 93, 94, 95, 97, 99):
            task = f"T-{number:03d}"
            manifest = json.loads(
                (ROOT / f"upstream/t{number:03d}_manifest.yaml").read_text()
            )
            self.assertEqual(manifest["status"], "reviewed-no-gpu-ready-units")
            self.assertEqual(manifest["acceptance_units"], [])
            plan = json.loads(
                (ROOT / f"upstream/t{number:03d}_reference_plan.yaml").read_text()
            )
            self.assertEqual(plan["cases"], [], task)
        with tempfile.TemporaryDirectory(dir="/home/z50063656/tmp") as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps({"task_id": "T-092", "cases": []}))
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/check_gpu_task_runnable.py"),
                    "--plan",
                    str(path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 4)
            self.assertIn("gpu_task_runnable=0", result.stdout)

    def test_incoming_guides_and_wrappers_exist(self):
        for number in range(91, 101):
            task = f"T-{number:03d}"
            self.assertTrue((ROOT / "results/incoming" / task / "README.md").is_file())
            self.assertTrue((ROOT / f"scripts/run_t{number:03d}_reference_all.sh").is_file())
            text = (ROOT / f"docs/T{number:03d}_FUNCTION_PERFORMANCE_GUIDE.md").read_text()
            for marker in ("更新时间", "功能测例", "性能测例", "triton_experimental"):
                self.assertIn(marker, text)
            self.assertFalse(text.endswith("\n\n"), task)


if __name__ == "__main__":
    unittest.main()
