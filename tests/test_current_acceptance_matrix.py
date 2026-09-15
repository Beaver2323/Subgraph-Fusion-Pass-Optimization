"""当前 acceptance-unit 矩阵的零设备一致性回归。"""

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_generator():
    path = ROOT / "scripts/generate_current_acceptance_matrix.py"
    spec = importlib.util.spec_from_file_location("current_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = load_generator()


class CurrentAcceptanceMatrixTests(unittest.TestCase):
    def test_generator_import_works_without_other_tests_sys_path(self):
        code = ("import importlib.util,sys; "
                f"s=importlib.util.spec_from_file_location('standalone_matrix',{str(ROOT / 'scripts/generate_current_acceptance_matrix.py')!r}); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "assert len(m.build_rows('2026-09-14T22:25:00+08:00'))==71; "
                "assert 'torch' not in sys.modules")
        subprocess.run([sys.executable,'-I','-c',code],check=True,cwd='/home/z50063656/tmp')

    def test_progress_rejects_foreign_backend_and_fake_completion(self):
        original = matrix.read_json
        for field, value in (("backend", "default"), ("contract_complete", False)):
            def altered(path):
                data = original(path)
                if path.name == "npu_contract_progress.json" and data.get("task_id") == "T-088":
                    data["units"]["AU-split-cat-merge-select-cat-aten"][field] = value
                return data
            with self.subTest(field=field), patch.object(matrix, "read_json", side_effect=altered):
                with self.assertRaises(ValueError):
                    matrix.npu_contract_progress("T-088", "AU-split-cat-merge-select-cat-aten")

    def test_progress_rejects_wrong_digest_and_outside_path(self):
        with self.assertRaises(ValueError):
            matrix.verified_file("../activate_pass.sh", "0" * 64)
        with self.assertRaises(ValueError):
            matrix.verified_file("results/current/T-087/npu_training_review.json", "0" * 64)

    def test_current_units_and_task_boundaries(self):
        rows = matrix.build_rows("2026-09-06T00:00:00+08:00")
        self.assertEqual(len(rows), 71)
        self.assertEqual(
            Counter(row["task_id"] for row in rows),
            Counter({
                "T-076": 5,
                "T-077": 5,
                "T-078": 4,
                "T-079": 4,
                "T-080": 3,
                "T-081": 2,
                "T-082": 2,
                "T-083": 3,
                "T-084": 1,
                "T-085": 3,
                "T-086": 1,
                "T-087": 2,
                "T-088": 2,
                "T-089": 1,
                "T-090": 1,
                "T-091": 1,
                "T-096": 1,
                "T-098": 1,
                "T-100": 1,
                "T-102": 5,
                "T-103": 5,
                "T-104": 5,
                "T-105": 5,
                "T-106": 4,
                "T-107": 3,
                "T-112": 1,
            }),
        )
        self.assertEqual(len({row["acceptance_unit_id"] for row in rows}), 71)
        self.assertEqual(sum(row["independent_unit_contribution"] for row in rows), 69)
        alias = next(row for row in rows if row["task_id"] == "T-112")
        self.assertEqual(alias["canonical_acceptance_unit_id"], "AU-post-grad-dedup-reduce-scatters")
        self.assertEqual(alias["independent_unit_contribution"], 0)

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
        self.assertEqual(sum(bool(row["comparison_result_path"]) for row in rows), 54)
        self.assertEqual(
            sum(row["denominator_eligible"] == "yes-frozen" for row in rows), 54
        )
        self.assertEqual(
            sum(row["current_phase"] == "awaiting-gpu-reference" for row in rows), 0
        )
        # 安装态基线逐例推进，未完成单元只能在这些真实阶段之间移动；
        # 不能因为原例执行完就自动增加comparison/性能计数。
        pending_phases = {'awaiting-npu','npu-regression-open',
                          'npu-candidate-verified-awaiting-product-review','npu-contract-review','npu-adapter-review',
                          'npu-installed-repair-verification-running',
                          'npu-installed-repair-verified-awaiting-performance'}
        self.assertEqual(sum(row['current_phase'] in pending_phases for row in rows), 13)
        progress = [row for row in rows if row["npu_progress_path"]]
        progress_tasks = {r['task_id'] for r in progress}
        self.assertTrue({'T-098','T-103'} <= progress_tasks)
        self.assertTrue(progress_tasks <= {'T-098','T-102','T-103','T-104','T-105','T-106','T-107'})
        pending = next(r for r in progress if r['acceptance_unit_id']=='AU-fuse-attention-sfdp-pattern-6')
        self.assertEqual(pending['repair_status'],'baseline-failed-diagnosis-pending')
        self.assertEqual(pending['npu_execution_status'],'failed')
        for number in range(1,6):
            repaired_training=next(r for r in rows if r['acceptance_unit_id']==f'AU-fuse-attention-sfdp-pattern-{number}')
            self.assertEqual(repaired_training['repair_status'],'installed-original-neighbors-boundary-verified')
            self.assertEqual(repaired_training['comparison_verdict'],'NEWLY_SUPPORTED')
            self.assertTrue(repaired_training['comparison_result_path'])
        old=matrix.read_json(ROOT/'issues/REF-sfdp-pattern-1-native/evidence/t102-native-oen9jamj/adapter/result.json')
        self.assertEqual(old['status'],'failed-or-target-missing')
        self.assertFalse(old['native_assertions_passed'])
        self.assertTrue(all(not r['comparison_result_path'] for r in progress))
        installed = [row for row in rows if row["repair_status"] == "installed-fix-verified-not-upstream-merged"]
        self.assertEqual(len(installed), 2)
        self.assertTrue(all(row["comparison_verdict"] == "NEWLY_SUPPORTED" for row in installed))
        self.assertTrue(all(row["comparison_result_path"] for row in installed))
        self.assertEqual(sum(row["current_phase"] == "gpu-target-attribution-pending" for row in rows), 2)
        self.assertEqual(
            sum(row["current_phase"] == "functional-comparison-closed" for row in rows),
            28,
        )
        self.assertEqual(
            sum(row["current_phase"] == "formally-closed" for row in rows), 26
        )
        self.assertEqual(
            sum(
                row["current_phase"]
                == "coverage-extension-gpu-reference-pending"
                for row in rows
            ),
            0,
        )
        self.assertEqual(
            sum(
                row["current_phase"]
                == "coverage-extension-fp16-precision-blocked"
                for row in rows
            ),
            0,
        )
        self.assertEqual(
            sum(
                row["performance_evidence_path"].startswith("results/current/")
                for row in rows
            ),
            54,
        )

        repaired = next(r for r in rows if r['acceptance_unit_id']=='AU-fuse-attention-sfdp-pattern-22')
        self.assertEqual(repaired['current_phase'], 'formally-closed')
        self.assertEqual(repaired['repair_status'], 'installed-original-neighbors-boundary-verified')
        self.assertTrue(repaired['comparison_result_path'])
        limited = next(r for r in rows if r['acceptance_unit_id']=='AU-fuse-attention-sfdp-pattern-21')
        self.assertEqual(limited['performance_verdict'], 'PERF_NOT_INDEPENDENTLY_ATTRIBUTABLE')
        self.assertEqual(limited['performance_status'], 'accepted-not-independently-attributable')
        self.assertEqual(limited['current_phase'], 'formally-closed')

    def test_mixed_task_keeps_per_pattern_gpu_verdict(self):
        rows = matrix.build_rows("2026-09-14T20:04:00+08:00")
        by_unit = {r['acceptance_unit_id']:r for r in rows}
        for number in (16,29):
            row = by_unit[f'AU-fuse-attention-sfdp-pattern-{number}']
            self.assertEqual(row['reference_status'], 'native-passed-different-target-observed')
            self.assertEqual(row['current_phase'], 'gpu-target-attribution-pending')
        alias = by_unit['AU-fuse-attention-sfdp-pattern-17']
        self.assertEqual(alias['canonical_acceptance_unit_id'], 'AU-fuse-attention-sfdp-pattern-15')
        self.assertEqual(alias['independent_unit_contribution'], 0)
        self.assertEqual(alias['current_phase'], 'duplicate-evidence-retained-not-counting')
        self.assertFalse(alias['comparison_result_path'])
        for number in (18,19,20,28,30):
            row = by_unit[f'AU-fuse-attention-sfdp-pattern-{number}']
            expected = ('valid-reference-frozen' if row['denominator_eligible'] == 'yes-frozen'
                        else 'gpu-contract-reviewed-awaiting-npu')
            self.assertEqual(row['reference_status'], expected)
            self.assertIn(row['current_phase'], {
                'awaiting-npu', 'npu-contract-review', 'npu-regression-open', 'npu-adapter-review',
                'npu-candidate-verified-awaiting-product-review', 'functional-comparison-closed', 'formally-closed',
            })

    def test_unclosed_base_contract_is_not_fully_covered(self):
        rows = matrix.build_rows("2026-09-14T23:52:00+08:00")
        for row in rows:
            if not row['comparison_result_path']:
                self.assertNotEqual(row['coverage_status'], 'fully-covered')
        alias = next(row for row in rows if row['task_id'] == 'T-112')
        self.assertEqual(alias['coverage_status'], 'duplicate-noncounting')

    def test_t084_t086_bind_explicit_alignment_and_performance(self):
        rows = matrix.build_rows("2026-09-10T06:55:00+08:00")
        recent = [
            row for row in rows if row["task_id"] in {"T-084", "T-085", "T-086"}
        ]
        self.assertEqual(len(recent), 5)
        self.assertTrue(all(row["current_phase"] == "formally-closed" for row in recent))
        self.assertTrue(all(row["community_alignment_source"] == "explicit" for row in recent))
        self.assertEqual(
            {row["performance_verdict"] for row in recent},
            {"PERF_IMPROVED", "PERF_MIXED", "PERF_REGRESSED"},
        )

    def test_t078_value_one_extension_closed_but_rounding_divergence_visible(self):
        rows = matrix.build_rows("2026-09-08T06:07:40+08:00")
        row = next(
            item
            for item in rows
            if item["acceptance_unit_id"]
            == "AU-post-grad-fuse-addcdiv-to-fma"
        )
        self.assertEqual(row["variant_count"], 6)
        self.assertEqual(row["verified_variant_count"], 6)
        self.assertEqual(row["pending_variant_count"], 0)
        self.assertEqual(
            row["current_phase"],
            "functional-comparison-closed",
        )
        self.assertEqual(row["community_alignment_status"], "PARTIAL_ALIGNED")
        self.assertEqual(row["community_alignment_source"], "explicit")
        self.assertIn("NPU FP16", row["community_divergent_scope"])
        self.assertEqual(row["community_open_scope"], "")
        self.assertIn("FP16 value=1", row["community_aligned_scope"])

    def test_legacy_results_are_not_inferred_as_fully_aligned(self):
        rows = matrix.build_rows("2026-09-08T21:10:00+08:00")
        legacy = next(
            item
            for item in rows
            if item["acceptance_unit_id"] == "AU-apply-gumbel-max-trick"
        )
        self.assertEqual(legacy["community_alignment_status"], "PENDING_REVIEW")
        self.assertEqual(
            legacy["community_alignment_source"], "legacy-missing-explicit"
        )

    def test_t101_t113_preparation_alignment_is_visible_without_fake_result(self):
        rows = matrix.build_rows("2026-09-10T22:55:21+08:00")
        pattern_1 = next(
            row
            for row in rows
            if row["acceptance_unit_id"] == "AU-fuse-attention-sfdp-pattern-1"
        )
        pattern_16 = next(
            row
            for row in rows
            if row["acceptance_unit_id"] == "AU-fuse-attention-sfdp-pattern-16"
        )
        self.assertEqual(
            pattern_1["community_alignment_status"], "FULL_ALIGNED"
        )
        self.assertEqual(
            pattern_16["community_alignment_status"],
            "backend-specific-partial-alignment",
        )
        self.assertEqual(pattern_1["community_alignment_source"], "explicit")
        self.assertEqual(pattern_16["community_alignment_source"], "manifest-preparation-contract")
        self.assertTrue(pattern_1["comparison_result_path"])
        self.assertFalse(pattern_16["comparison_result_path"])
        self.assertIn("设备行为尚未形成结论", pattern_16["community_alignment_disposition"])

    def test_t102_installed_training_keeps_partial_alignment_visible(self):
        rows={row['acceptance_unit_id']:row for row in matrix.build_rows('2026-09-15T20:00:00+08:00')}
        for number in range(1,6):
            with self.subTest(pattern=number):
                row=rows[f'AU-fuse-attention-sfdp-pattern-{number}']
                self.assertEqual(row['community_alignment_status'],
                                 'PARTIAL_ALIGNED' if number in (3,4,5) else 'FULL_ALIGNED')
                raw=matrix.read_json(ROOT/f'results/current/T-102/functional/pattern-{number}.json')
                self.assertEqual(raw['measurement_stage'],'training-forward-only')
                self.assertTrue(any(x.get('requires_grad') for x in raw['input_spec']))
                self.assertIn('installed_deployment',raw)
                if number in (3,4):
                    self.assertIn('原社区无数值oracle',row['community_open_scope'])
                if number==5:
                    self.assertIn('数学展开',row['community_divergent_scope'])

    def test_t081_t083_results_bind_backend_and_learning_evidence(self):
        rows = matrix.build_rows("2026-09-08T03:17:00+08:00")
        recent = [row for row in rows if row["task_id"] in {"T-081", "T-082", "T-083"}]
        self.assertEqual(len(recent), 7)
        for row in recent:
            result = matrix.read_json(ROOT / row["comparison_result_path"])
            self.assertEqual(result["backend"], "triton_experimental")
            summary = matrix.read_json(ROOT / row["performance_evidence_path"])
            unit = next(
                item
                for item in summary["acceptance_units"]
                if item["acceptance_unit_id"] == row["acceptance_unit_id"]
            )
            explanation = unit["pattern_explanation"]
            self.assertTrue(explanation["intent"])
            self.assertTrue(explanation["source"])
            self.assertTrue(explanation["gpu_behavior"])
            self.assertTrue(explanation["npu_behavior"])

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
