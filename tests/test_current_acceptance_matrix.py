"""当前 acceptance-unit 矩阵的零设备一致性回归。"""

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_generator():
    path = ROOT / "scripts/generate_current_acceptance_matrix.py"
    spec = importlib.util.spec_from_file_location("current_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = load_generator()


class CurrentAcceptanceMatrixTests(unittest.TestCase):
    def test_current_units_and_task_boundaries(self):
        rows = matrix.build_rows("2026-09-06T00:00:00+08:00")
        self.assertEqual(len(rows), 28)
        self.assertEqual(
            Counter(row["task_id"] for row in rows),
            Counter({"T-076": 5, "T-077": 5, "T-078": 4, "T-079": 4, "T-080": 3,
                     "T-081": 2, "T-082": 2, "T-083": 3}),
        )
        self.assertEqual(len({row["acceptance_unit_id"] for row in rows}), 28)

    def test_npu_backend_never_inherits_reference_backend(self):
        rows = matrix.build_rows("2026-09-06T00:00:00+08:00")
        self.assertEqual({row["reference_backend"] for row in rows}, {"inductor-default"})
        self.assertEqual(
            {row["required_npu_backend"] for row in rows}, {"triton_experimental"}
        )
        observed = {
            row["observed_npu_backend"]
            for row in rows
            if row["observed_npu_backend"]
        }
        self.assertEqual(observed, {"triton_experimental"})

    def test_dynamic_and_pending_evidence_are_not_conflated(self):
        rows = matrix.build_rows("2026-09-06T00:00:00+08:00")
        self.assertEqual(sum(bool(row["comparison_result_path"]) for row in rows), 21)
        self.assertEqual(
            sum(row["denominator_eligible"] == "yes-frozen" for row in rows), 21
        )
        self.assertEqual(
            sum(row["current_phase"] == "awaiting-gpu-reference" for row in rows), 7
        )
        self.assertEqual(
            sum(row["current_phase"] == "awaiting-npu" for row in rows), 0
        )
        self.assertEqual(
            sum(
                row["performance_evidence_path"].startswith("results/current/")
                for row in rows
            ),
            21,
        )

    def test_t080_results_include_learning_evidence(self):
        rows = matrix.build_rows("2026-09-07T20:50:00+08:00")
        t080 = [row for row in rows if row["task_id"] == "T-080"]
        self.assertEqual(len(t080), 3)
        for row in t080:
            comparison = matrix.read_json(ROOT / row["comparison_result_path"])
            explanation = comparison.get("pattern_explanation", {})
            self.assertTrue(explanation.get("name"), row["acceptance_unit_id"])
            self.assertTrue(explanation.get("intent"), row["acceptance_unit_id"])
            self.assertTrue(
                explanation.get("source_excerpt"), row["acceptance_unit_id"]
            )
            self.assertTrue(
                comparison.get("variant_comparisons"), row["acceptance_unit_id"]
            )

    def test_committed_outputs_are_current(self):
        matrix.check_outputs()


if __name__ == "__main__":
    unittest.main()
