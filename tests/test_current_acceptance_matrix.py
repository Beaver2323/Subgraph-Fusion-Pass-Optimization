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
        self.assertEqual(len(rows), 21)
        self.assertEqual(
            Counter(row["task_id"] for row in rows),
            Counter({"T-076": 5, "T-077": 5, "T-078": 4, "T-079": 4, "T-080": 3}),
        )
        self.assertEqual(len({row["acceptance_unit_id"] for row in rows}), 21)

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
        self.assertEqual(sum(bool(row["comparison_result_path"]) for row in rows), 10)
        self.assertEqual(
            sum(row["denominator_eligible"] == "yes-frozen" for row in rows), 10
        )
        self.assertEqual(
            sum(row["current_phase"] == "awaiting-gpu-reference" for row in rows), 11
        )
        self.assertEqual(
            sum(
                row["performance_evidence_path"].startswith("results/current/")
                for row in rows
            ),
            10,
        )

    def test_committed_outputs_are_current(self):
        matrix.check_outputs()


if __name__ == "__main__":
    unittest.main()
