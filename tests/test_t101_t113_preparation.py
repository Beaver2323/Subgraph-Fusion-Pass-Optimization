"""T-101～T-113 零设备准备合同；不得导入torch。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PYTORCH = Path("/home/z50063656/Pass/src/pytorch")


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load("t101_t113_reference", "runners/reference_runner.py")
attention_worker = load(
    "t102_t107_attention_worker",
    "runners/t102_t107_attention_performance_worker.py",
)
attention_launcher = load(
    "t102_t107_attention_launcher",
    "scripts/run_t102_t107_attention_performance.py",
)
fsdp_worker = load(
    "t112_fsdp_worker", "runners/t112_dedup_reduce_scatter_worker.py"
)
fsdp_launcher = load(
    "t112_fsdp_launcher", "scripts/run_t112_dedup_reduce_scatter.py"
)


class T101T113PreparationTests(unittest.TestCase):
    def test_all_reference_contracts_resolve_statically(self):
        expected = {
            101: (0, 0, 0),
            102: (5, 5, 5),
            103: (5, 5, 5),
            104: (5, 5, 5),
            105: (5, 5, 5),
            106: (4, 4, 4),
            107: (3, 3, 3),
            108: (0, 0, 0),
            109: (0, 0, 0),
            110: (0, 0, 0),
            111: (0, 0, 0),
            112: (1, 1, 1),
            113: (0, 0, 0),
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

    def test_review_covers_all_52_candidates_without_reassignment(self):
        backlog = json.loads((ROOT / "upstream/task_backlog.json").read_text())
        batches = {
            batch["task_id"]: batch
            for batch in backlog["batches"]
            if 101 <= int(batch["task_id"][2:]) <= 113
        }
        selected = deferred = 0
        for task, batch in batches.items():
            suffix = task.lower().replace("-", "")
            manifest = json.loads((ROOT / "upstream" / f"{suffix}_manifest.yaml").read_text())
            original = {unit["provisional_unit_id"] for unit in batch["units"]}
            ready = {unit["acceptance_unit_id"] for unit in manifest["acceptance_units"]}
            held = {unit["provisional_unit_id"] for unit in manifest["deferred_candidates"]}
            self.assertEqual(ready | held, original)
            self.assertFalse(ready & held)
            selected += len(ready)
            deferred += len(held)
        self.assertEqual((selected, deferred), (28, 24))

    def test_attention_template_assignments_are_resolved_without_import(self):
        qualnames = reference.python_qualnames(
            PYTORCH / "test/inductor/test_fused_attention.py"
        )
        for pattern in (*range(1, 16), *range(17, 25), 28, 29, 30):
            self.assertIn(
                f"SDPAPatternRewriterGpuTests.test_sdpa_rewriter_{pattern}_gpu",
                qualnames,
            )
        self.assertIn(
            "SDPAPatternRewriterGpuTests.test_sdpa_rewriter_16_inference_gpu",
            qualnames,
        )

    def test_static_assignment_resolver_rejects_dynamic_factories(self):
        source = """
import functools
class Template:
    def _test(self): pass
class GPU:
    test_direct = Template._test
    test_partial = functools.partialmethod(Template._test, value=1)
    test_dynamic = make_test(Template._test)
"""
        with tempfile.TemporaryDirectory(dir="/home/z50063656/tmp") as directory:
            path = Path(directory) / "test_sample.py"
            path.write_text(source)
            names = reference.python_qualnames(path)
        self.assertIn("GPU.test_direct", names)
        self.assertIn("GPU.test_partial", names)
        self.assertNotIn("GPU.test_dynamic", names)

    def test_attention_and_fsdp_workers_have_fresh_process_contracts(self):
        self.assertEqual(
            attention_launcher.ORDER,
            (("off", 1), ("on", 1), ("on", 2), ("off", 2), ("off", 3), ("on", 3)),
        )
        self.assertEqual(
            set(attention_worker.SUPPORTED_PATTERNS),
            {*range(1, 25), 28, 29, 30},
        )
        self.assertNotIn(25, attention_worker.SUPPORTED_PATTERNS)
        self.assertEqual(fsdp_worker.WORKLOAD, "dedup-reduce-scatter-two-rank-fp32")
        self.assertEqual(fsdp_launcher.ORDER, attention_launcher.ORDER)
        for relative in (
            "runners/t102_t107_attention_performance_worker.py",
            "runners/t112_dedup_reduce_scatter_worker.py",
        ):
            source = (ROOT / relative).read_text()
            self.assertLess(
                source.index('os.environ["TORCHINDUCTOR_NPU_BACKEND"]'),
                source.index("    import torch\n"),
            )
            self.assertIn('"triton_experimental"', source)

    def test_explicit_cuda_disabled_and_zero_unit_batches_are_not_runnable(self):
        for pattern in (25, 26, 27):
            task = 106 if pattern == 25 else 107
            manifest = json.loads((ROOT / f"upstream/t{task}_manifest.yaml").read_text())
            record = next(
                item for item in manifest["deferred_candidates"]
                if item["provisional_unit_id"].endswith(f"pattern-{pattern}")
            )
            self.assertEqual(record["status"], "deferred-explicit-cuda-disabled-xpu-only")
        for number in (101, 108, 109, 110, 111, 113):
            manifest = json.loads((ROOT / f"upstream/t{number}_manifest.yaml").read_text())
            performance = json.loads((ROOT / f"upstream/t{number}_performance_plan.yaml").read_text())
            self.assertEqual(manifest["status"], "reviewed-no-gpu-ready-units")
            self.assertEqual(manifest["acceptance_units"], [])
            self.assertEqual(performance["implementation"]["status"], "not-implemented")

    def test_guides_wrappers_and_incoming_directories_exist(self):
        for number in range(101, 114):
            task = f"T-{number:03d}"
            self.assertTrue((ROOT / "results/incoming" / task / "README.md").is_file())
            self.assertTrue((ROOT / f"scripts/run_t{number:03d}_reference_all.sh").is_file())
            text = (ROOT / f"docs/T{number:03d}_FUNCTION_PERFORMANCE_GUIDE.md").read_text()
            for marker in ("更新时间", "功能测例", "性能测例", "triton_experimental"):
                self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
