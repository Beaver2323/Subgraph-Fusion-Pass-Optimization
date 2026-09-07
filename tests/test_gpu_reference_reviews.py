"""T-081～T-083 上传 GPU reference 与冻结记录的一致性校验。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from import_reference_text import load_input, validate_payload  # noqa: E402


TASKS = {
    "T-081": ("manifest.json", 2, 3, 10, 7, 2, 1),
    "T-082": ("text-handoff.json", 2, 4, 8, 4, 1, 3),
    "T-083": ("text-handoff.json", 3, 4, 6, 5, 2, 2),
}


class GPUReferenceReviewTests(unittest.TestCase):
    def test_handoff_plan_manifest_and_review_are_bound(self):
        for task_id, expected in TASKS.items():
            input_name, units, cases, variants, tests_ran, numerical, structural = expected
            suffix = task_id.lower().replace("-", "")
            input_path = ROOT / "results/incoming" / task_id / input_name
            payload = load_input(input_path)
            validate_payload(payload)
            summary = payload["reference_summary"]
            audit = payload["case_audit"]
            plan = json.loads(
                (ROOT / "upstream" / f"{suffix}_reference_plan.yaml").read_text()
            )
            manifest = json.loads(
                (ROOT / "upstream" / f"{suffix}_manifest.yaml").read_text()
            )
            review = json.loads(
                (ROOT / "results/current" / task_id / "gpu_reference_review.json").read_text()
            )

            with self.subTest(task=task_id):
                self.assertEqual(summary["status"], "valid-reference-suite")
                self.assertTrue(summary["suite_complete"])
                self.assertTrue(summary["suite_valid"])
                self.assertEqual(summary["case_counts"]["reference_valid"], cases)
                self.assertEqual(len(summary["variants"]), variants)
                self.assertEqual(sum(item["tests_ran"] for item in audit), tests_ran)
                self.assertTrue(all(item["tests_skipped"] == 0 for item in audit))
                self.assertTrue(all(item["reference_valid"] for item in audit))
                self.assertEqual(
                    {item["adapter_decision"] for item in audit},
                    {"not-needed-direct-valid"},
                )
                self.assertEqual(
                    {item["case_id"] for item in audit},
                    {item["case_id"] for item in plan["cases"]},
                )
                self.assertEqual(
                    manifest["counting_policy"]["current_frozen_denominator_units"],
                    units,
                )
                self.assertTrue(
                    all(
                        item["denominator_eligible"] == "yes-frozen"
                        for item in manifest["acceptance_units"]
                    )
                )
                self.assertEqual(review["payload_sha256"], payload["payload_sha256"])
                self.assertEqual(review["run_id"], summary["run_id"])
                self.assertEqual(
                    review["environment_fingerprint"],
                    payload["environment"]["fingerprint_sha256"],
                )
                scope = review["evidence_scope"]
                observed_numerical = scope.get(
                    "compiled_numerical_cases", scope.get("single_rank_numerical_cases")
                )
                self.assertEqual(observed_numerical, numerical)
                self.assertEqual(scope["structure_only_cases"], structural)

    def test_t083_does_not_unlock_cross_rank_performance(self):
        review = json.loads(
            (ROOT / "results/current/T-083/gpu_reference_review.json").read_text()
        )
        self.assertEqual(review["distributed_world_size"], 1)
        self.assertFalse(review["evidence_scope"]["cross_rank_numerical_proven"])
        self.assertFalse(review["evidence_scope"]["cross_rank_performance_allowed"])


if __name__ == "__main__":
    unittest.main()
