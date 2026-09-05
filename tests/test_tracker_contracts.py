"""验收工具的零设备回归；从 /home/z50063656/tmp 用 unittest discover 执行。"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


reference = module("reference_runner", "runners/reference_runner.py")
comparison = module("comparison_validator", "scripts/validate_comparison_data.py")
preparation = module("preparation_validator", "scripts/validate_prepared_tasks.py")
publication = module("publication", "scripts/publish_reference_latest.py")
backlog = module("backlog", "scripts/build_task_backlog.py")


class ReferenceTests(unittest.TestCase):
    def test_functional_worker_does_not_inherit_performance_flags(self):
        from unittest.mock import Mock

        plan = json.loads((ROOT / "upstream/t078_reference_plan.yaml").read_text())
        case = plan["cases"][0]
        case["correctness_evidence"] = "codegen-only"
        observed = {}

        def start(*args, **kwargs):
            observed.update(kwargs["env"])
            kwargs["stderr"].write("Ran 1 test in 1s\nOK\n")
            return Mock(wait=Mock(return_value=0))

        with tempfile.TemporaryDirectory(dir="/home/z50063656/tmp") as directory:
            run_dir = Path(directory)
            with patch.dict(reference.os.environ, {"DO_PERF_TEST": "1", "USE_LARGE_INPUT": "1"}), patch.object(reference.subprocess, "Popen", side_effect=start), patch.object(reference, "git_output", return_value="0" * 40), patch.object(reference, "collect_fx", return_value={"before": {"captured": True}, "after": {"captured": True}}):
                result = reference.run_case(case, ROOT, ROOT, run_dir, run_dir, "test", "0" * 40, 60, None)
            self.assertTrue(result["reference_valid"])
            self.assertEqual(result["correctness"]["status"], "not-asserted-codegen-only")
            self.assertEqual(result["benchmark"]["functional_gate"], "not-passed")
        self.assertNotIn("DO_PERF_TEST", observed)
        self.assertNotIn("USE_LARGE_INPUT", observed)

    def test_complete_run_passes(self):
        result = reference.parse_unittest_output("", "Ran 3 tests in 1s\nOK", 0, 3)
        self.assertTrue(result["success"])

    def test_incomplete_or_failed_runs_never_pass(self):
        for text, code, expected in (
            ("Ran 3 tests in 1s\nOK (skipped=1)", 0, 3),
            ("Ran 3 tests in 1s\nOK (skipped=3)", 0, 3),
            ("Ran 3 tests in 1s\nOK (expected failures=1)", 0, 3),
            ("Ran 3 tests in 1s\nOK (unexpected successes=1)", 0, 3),
            ("Ran 2 tests in 1s\nOK", 0, 3),
            ("Ran 4 tests in 1s\nOK", 0, 3),
            ("Ran 3 tests in 1s\nFAILED (failures=1)", 0, 3),
            ("Ran 0 tests in 1s\nOK", 0, 1),
            ("", 0, 1),
            ("Ran 1 test in 1s\nOK", 1, 1),
            ("Ran 1 test in 1s\nOK", None, 1),
        ):
            with self.subTest(text=text, code=code, expected=expected):
                result = reference.parse_unittest_output("", text, code, expected)
                self.assertFalse(result["success"])

    def test_partial_skip_invalidates_all_linked_variants(self):
        case = {"case_id": "case", "acceptance_unit_id": "unit", "variant_ids": ["a", "b"]}
        plan = {"cases": [case], "non_executed_variants": []}
        manifest = {"acceptance_units": [{"acceptance_unit_id": "unit", "variants": [
            {"variant_id": "a"}, {"variant_id": "b"}
        ]}]}
        parsed = reference.parse_unittest_output("", "Ran 2 tests in 1s\nOK (skipped=1)", 0, 2)
        summaries = reference.build_variant_summary(manifest, plan, [
            {"case": case, "reference_valid": parsed["success"]}
        ])
        self.assertTrue(all(item["status"] == "direct-blocked-or-invalid" for item in summaries))


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.result_path = ROOT / "results/current/AU-apply-gumbel-max-trick/npu_result.json"
        self.comparison_path = self.result_path.with_name("comparison_result.json")
        self.result = json.loads(self.result_path.read_text())
        self.record = json.loads(self.comparison_path.read_text())
        manifest = json.loads((ROOT / "upstream/t077_manifest.yaml").read_text())
        self.unit = next(item for item in manifest["acceptance_units"] if item["acceptance_unit_id"] == self.result["acceptance_unit_id"])

    def validate_result(self):
        comparison.validate_npu_result(self.result, self.result_path, ROOT, self.unit)

    def validate_record(self):
        # 测试内存变体；模拟其实际序列化后的文件哈希，不修改现有证据。
        with patch.object(comparison, "sha256", return_value=self.record["npu"]["result_sha256"]):
            comparison.validate_comparison(self.record, self.comparison_path, self.result, self.result_path, self.unit)

    def fail_correctness(self):
        self.result["selected_execution"].update(status="failed", execution_success=False)
        variant = self.result["variants"][0]
        variant["correctness"]["status"] = "failed"
        variant["support_status"] = "regressed"
        variant["performance"]["status"] = "not-run"
        variant["first_divergence"] = "correctness"
        self.record["npu"].update(correctness_status="failed", execution_success=False)
        self.record.update(final_verdict="NPU_REGRESSION", repair_status="queued")
        self.record["variant_comparisons"][0].update(correctness_status="failed", verdict="NPU_REGRESSION", performance_status="not-run")

    def test_original_records_pass(self):
        self.validate_result()
        self.validate_record()
        self.assertTrue(comparison.is_formally_closed(self.record))

    def test_other_backend_rejected_even_with_consistent_fingerprint(self):
        for backend in ("default", "dvm", "mlir", ""):
            with self.subTest(backend=backend):
                env = self.result["environment"]
                env["backend"] = backend
                env["fingerprint_sha256"] = comparison.environment_fingerprint(env)
                with self.assertRaisesRegex(ValueError, "backend 必须"):
                    self.validate_result()

    def test_open_failure_is_valid_evidence_but_not_closed(self):
        self.fail_correctness()
        self.validate_result()
        self.validate_record()
        self.assertFalse(comparison.is_formally_closed(self.record))

    def test_failure_cannot_claim_verified_repair(self):
        self.fail_correctness()
        self.record["repair_status"] = "verified"
        with self.assertRaisesRegex(ValueError, "未修复 NPU_REGRESSION"):
            self.validate_record()

    def test_failure_cannot_claim_success_or_performance(self):
        self.fail_correctness()
        for field, value in (("selected_execution", True), ("performance", "passed")):
            with self.subTest(field=field):
                original = copy.deepcopy(self.result)
                if field == "selected_execution":
                    self.result[field].update(status="passed", execution_success=value)
                else:
                    self.result["variants"][0][field]["status"] = value
                with self.assertRaises(ValueError):
                    self.validate_result()
                self.result = original

    def test_failed_variant_cannot_claim_benefit(self):
        self.fail_correctness()
        self.record["variant_comparisons"][0]["verdict"] = "BEHAVIOR_UNCHANGED"
        with self.assertRaisesRegex(ValueError, "不得标记支持或收益"):
            self.validate_record()

    def test_failed_variant_cannot_claim_supported(self):
        self.fail_correctness()
        self.result["variants"][0]["support_status"] = "supported"
        with self.assertRaisesRegex(ValueError, "不得标记 supported"):
            self.validate_result()

    def test_comparison_cannot_hide_variant_failure(self):
        self.fail_correctness()
        self.record["npu"]["correctness_status"] = "passed"
        with self.assertRaisesRegex(ValueError, "correctness 与 variants"):
            self.validate_record()


class PreparationTests(unittest.TestCase):
    def validate_changed(self, mutate):
        real_loader = preparation.load_json

        def loader(path):
            value = real_loader(path)
            if path.name == "t078_performance_plan.yaml":
                mutate(value)
            return value

        with patch.object(preparation, "load_json", side_effect=loader):
            return preparation.validate_task(ROOT, "T-078")

    def test_current_plans_pass(self):
        for task in preparation.TASKS:
            with self.subTest(task=task):
                preparation.validate_task(ROOT, task)

    def test_invalid_iterations_rejected(self):
        for key in ("warmup", "runs"):
            for value in (-1, 0, True, 1.5, "100"):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "正整数"):
                    self.validate_changed(lambda data: data["measurement_contract"].__setitem__(key, value))

    def test_missing_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "nodeids"):
            self.validate_changed(lambda data: data["acceptance_units"][0]["case_source"].update(nodeids=[]))

    def test_source_from_other_unit_rejected(self):
        def mutate(data):
            data["acceptance_units"][0]["case_source"]["nodeids"] = data["acceptance_units"][1]["case_source"]["nodeids"]
        with self.assertRaisesRegex(ValueError, "性能来源未登记"):
            self.validate_changed(mutate)

    def test_duplicate_unit_rejected(self):
        with self.assertRaisesRegex(ValueError, "重复 acceptance_unit_id"):
            self.validate_changed(lambda data: data["acceptance_units"].append(copy.deepcopy(data["acceptance_units"][0])))

    def test_duplicate_workload_rejected(self):
        def mutate(data):
            values = data["acceptance_units"][0]["workloads"]
            values.append(copy.deepcopy(values[0]))
        with self.assertRaisesRegex(ValueError, "重复 workload_id"):
            self.validate_changed(mutate)

    def test_missing_timezone_rejected(self):
        with self.assertRaisesRegex(ValueError, "时区"):
            self.validate_changed(lambda data: data.update(generated_at="2026-09-06T02:00:00"))

    def test_nonexistent_worker_cannot_claim_implemented(self):
        def mutate(data):
            data["implementation"] = {"status": "implemented-awaiting-runtime-validation", "entrypoint": "runners/missing_worker.py"}
        with self.assertRaisesRegex(ValueError, "实际文件"):
            self.validate_changed(mutate)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tracker-contract-", dir="/home/z50063656/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_dir(self, name, complete=True):
        directory = self.root / name
        directory.mkdir()
        if complete:
            (directory / "text-handoff.json").write_text(json.dumps({"reference_summary": {"run_id": name}}))
        return directory

    def test_new_failure_never_returns_old_success(self):
        old = self.run_dir("old")
        publication.publish(self.root, old, 0, 0)
        new = self.run_dir("new", complete=False)
        publication.publish(self.root, new, 1, 1)
        alias = self.root / "latest-text-handoff.json"
        data = json.loads(alias.read_text())
        self.assertEqual(data["handoff_status"], "export-failed")
        self.assertEqual(data["run_id"], "new")
        self.assertEqual(alias.resolve().parent, (self.root / "latest").resolve())
        self.assertTrue((old / "text-handoff.json").exists())

    def test_launch_without_artifacts_publishes_failure(self):
        directory = publication.publish(self.root, None, 0, 1)
        data = json.loads((directory / "text-handoff.json").read_text())
        self.assertFalse(data["reference_valid"])
        self.assertNotEqual(data["runner_status"], 0)

    def test_mismatched_run_id_rejected(self):
        directory = self.run_dir("new")
        (directory / "text-handoff.json").write_text('{"reference_summary":{"run_id":"old"}}')
        with self.assertRaisesRegex(ValueError, "run_id"):
            publication.publish(self.root, directory, 0, 0)

    def test_real_latest_directory_preserved(self):
        directory = self.run_dir("new")
        (self.root / "latest").mkdir()
        with self.assertRaisesRegex(ValueError, "非软链接"):
            publication.publish(self.root, directory, 0, 0)
        self.assertTrue((self.root / "latest").is_dir())

    def test_incomplete_handoff_is_preserved(self):
        directory = self.run_dir("new")
        publication.publish(self.root, directory, 1, 1)
        self.assertEqual(len(list(directory.glob("text-handoff.incomplete-*.json"))), 1)

    def test_run_outside_root_rejected(self):
        with self.assertRaisesRegex(ValueError, "直接子目录"):
            publication.publish(self.root, self.root.parent, 0, 0)


class BacklogTests(unittest.TestCase):
    def test_inventory_partition_is_complete_without_duplicates(self):
        data = backlog.build(ROOT, "2026-09-06T02:15:00+08:00")
        units = [unit["provisional_unit_id"] for batch in data["batches"] for unit in batch["units"]]
        controls = [unit["provisional_unit_id"] for unit in data["non_counting_review"]]
        self.assertEqual(len(units), len(set(units)))
        self.assertFalse(set(units) & set(controls))
        self.assertEqual(len(units) + len(controls) + data["counts"]["selected_manifest_units"], data["counts"]["inventory_units"])
        self.assertNotIn("AU-post-grad-move-constructors-to-cuda", units)
        self.assertTrue(all(1 <= len(batch["units"]) <= 5 for batch in data["batches"]))

    def test_draft_batches_never_claim_runnable(self):
        data = backlog.build(ROOT, "2026-09-06T02:15:00+08:00")
        self.assertTrue(all(not batch["reference_ready"] for batch in data["batches"]))
        self.assertTrue(all(batch["performance_readiness"] == "needs-community-benchmark-search-and-worker" for batch in data["batches"]))


if __name__ == "__main__":
    unittest.main()
