"""历史复核与完整文本发布链路的零设备回归；不修改正式证据。"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = module("audit_history", "scripts/audit_history.py")
publisher = module("publish_reference_latest", "scripts/publish_reference_latest.py")
with patch.dict(sys.modules, {"audit_history": audit, "publish_reference_latest": publisher}):
    gate = module("validate_all", "scripts/validate_all.py")


class HistoryFixtures(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir="/home/z50063656/tmp")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))


class HistoryTests(HistoryFixtures):
    def test_missing_file_is_pending_but_bad_hash_is_failed(self):
        path = self.root / "evidence.json"
        evidence = {}
        self.assertEqual(audit.observe(path, evidence)["status"], "pending")
        self.write(path, {"value": 1})
        self.assertEqual(audit.observe(path, evidence, "0" * 64)["status"], "failed")
        self.assertEqual(evidence[str(path)]["sha256"], audit.digest(path))

    def test_archive_is_append_only(self):
        path = self.root / "audit.json"
        audit.write_new(path, {"value": "original"})
        with self.assertRaises(FileExistsError):
            audit.write_new(path, {"value": "new"})
        self.assertEqual(audit.read_json(path), {"value": "original"})

    def test_escape_and_symlink_outside_evidence_are_rejected(self):
        folder = self.root / "case"
        folder.mkdir()
        (folder / "outside").symlink_to(self.root)
        for path in ("../evidence.json", "outside/evidence.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                audit.inside(folder, path)

    def test_strict_gate_does_not_greenwash_pending(self):
        self.assertEqual(gate.history_exit("passed", True), 0)
        self.assertEqual(gate.history_exit("pending", True), 3)
        self.assertEqual(gate.history_exit("pending", False), 0)
        self.assertEqual(gate.history_exit("failed", False), 2)
        with self.assertRaises(ValueError):
            gate.history_exit("unknown", False)
        with self.assertRaises(ValueError):
            audit.combined([])

    def test_current_records_get_sidecar_without_rewriting_evidence(self):
        originals = {path: audit.digest(path) for path in (ROOT / "results/current").rglob("*.json")}
        payload = audit.build()
        self.assertEqual(payload["npu_record_checks_passed"], 10)
        self.assertEqual(sum(len(task["gpu_cases"]) for task in payload["tasks"]), 24)
        self.assertEqual(sum(len(task["performance_units"]) for task in payload["tasks"]), 10)
        self.assertEqual(payload["status"], "pending")
        self.assertEqual(payload["validator_files"]["scripts/audit_history.py"], audit.digest(ROOT / "scripts/audit_history.py"))
        self.assertEqual({path: audit.digest(path) for path in originals}, originals)

    def test_gpu_run_selection_rejects_unknown_and_duplicates(self):
        for values in (["T-067=/tmp/run"], ["T-076=/a", "T-076=/b"], ["T-077="]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                audit.parse_gpu_runs(values)

    def gpu_case(self, log):
        case = {"case_id": "case", "source_test": "test/example.py::Test.test_example", "direct_args": ["Test.test_a", "Test.test_b"]}
        case_dir = self.root / "cases/case"
        case_dir.mkdir(parents=True)
        result = {"case": {"case_id": "case"}, "source": {"actual_commit": "a" * 40}, "execution": {"return_code": 0}}
        self.write(case_dir / "reference_result.json", result)
        for name, content in (("stdout.log", ""), ("stderr.log", log), ("fx_before.txt", "before"), ("fx_after.txt", "after")):
            (case_dir / name).write_text(content)
        inventory = [{"path": name, "sha256": audit.digest(case_dir / name)} for name in ("stdout.log", "stderr.log", "fx_before.txt", "fx_after.txt")]
        self.write(case_dir / "artifact_inventory.json", inventory)
        recorded = {"reference_result_sha256": audit.digest(case_dir / "reference_result.json"), "artifact_inventory_sha256": audit.digest(case_dir / "artifact_inventory.json")}
        return audit.audit_gpu_case(self.root, case, recorded, "a" * 40, {})

    def test_old_pass_with_partial_skip_fails_new_reparse(self):
        result = self.gpu_case("Ran 2 tests in 1s\nOK (skipped=1)\n")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["checks"]["unittest_reparse"]["parsed"]["tests_skipped"], 1)

    def test_clean_log_is_not_automatic_numeric_certification(self):
        result = self.gpu_case("Ran 2 tests in 1s\nOK\n")
        self.assertEqual(result["checks"]["unittest_reparse"]["status"], "passed")
        self.assertEqual(result["status"], "pending")

    def workers(self, mutate=None):
        self.write(self.root / "performance_summary.json", {})
        policy = audit.read_json(ROOT / "schemas/audit_policy.json")
        raw = {"root": str(self.root), "summary_sha256": audit.digest(self.root / "performance_summary.json"), "worker_sha256": {}}
        for name in policy["performance_round_order"]:
            worker = {"acceptance_unit_id": "unit", "mode": name[:-1], "round": int(name[-1]),
                      "warmup": 10, "runs": 100, "input": {"shape": [10, 10]},
                      "correctness": "passed", "pattern": {"actual_count": int(name.startswith("on")), "expected_count": int(name.startswith("on"))},
                      "environment": {"backend": "triton_experimental"}}
            if mutate:
                mutate(worker)
            path = self.root / "workers" / name / "result.json"
            self.write(path, worker)
            raw["worker_sha256"][name] = audit.digest(path)
        return audit.audit_workers(raw, "unit", policy, {})

    def test_performance_wrong_backend_is_failed_not_pending(self):
        checks = self.workers(lambda worker: worker["environment"].update(backend="default"))
        self.assertEqual(checks["off1:backend"]["status"], "failed")
        self.assertEqual(audit.combined(checks.values()), "failed")

    def test_missing_backend_and_lifecycle_are_pending(self):
        checks = self.workers(lambda worker: worker.update(environment={}))
        self.assertEqual(checks["off1:functional_round"]["status"], "passed")
        self.assertEqual(checks["off1:backend"]["status"], "pending")
        self.assertEqual(audit.combined(checks.values()), "pending")

    def test_duplicate_round_failed_even_with_six_files(self):
        checks = self.workers(lambda worker: worker.update(round=1))
        self.assertEqual(checks["off2:functional_round"]["status"], "failed")

    def test_failed_correctness_blocks_performance(self):
        checks = self.workers(lambda worker: worker.update(correctness="failed"))
        self.assertEqual(checks["on1:functional_round"]["status"], "failed")


class ExportIntegrationTests(HistoryFixtures):
    # 真实 CLI 导出与发布，覆盖过去只 mock publisher 没有触达的路径约束冲突。
    def fixture(self):
        run = self.root / "reference-run"
        run.mkdir()
        self.write(run / "environment.json", {"source": {"actual_commit": "a" * 40}})
        self.write(run / "reference_summary.json", {"run_id": run.name, "cases": []})
        self.write(run / "manifest_snapshot.json", {})
        self.write(run / "reference_plan_snapshot.json", {})
        return run

    def export(self, run, path, allow=True):
        args = [sys.executable, str(ROOT / "scripts/export_reference_text.py"), "--run-dir", str(run), "--output", str(path)]
        if allow:
            args.append("--allow-derived-output")
        return subprocess.run(args, cwd="/home/z50063656/tmp", capture_output=True, text=True)

    def test_real_export_then_latest_publication(self):
        run = self.fixture()
        originals = {str(path): audit.digest(path) for path in run.iterdir()}
        result = self.export(run, run / "text-handoff.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        publisher.publish(self.root, run, 0, 0)
        payload = audit.read_json(self.root / "latest-text-handoff.json")
        self.assertEqual(payload["reference_summary"]["run_id"], run.name)
        for path, expected in originals.items():
            self.assertEqual(audit.digest(Path(path)), expected)
        self.assertIn("--allow-derived-output", (ROOT / "scripts/run_gpu_reference_task.sh").read_text())
        self.assertIn("--include-raw-text", (ROOT / "scripts/run_gpu_reference_task.sh").read_text())

    def test_output_cannot_replace_source_even_with_flag(self):
        run = self.fixture()
        path = run / "environment.json"
        before = path.read_bytes()
        result = self.export(run, path)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(path.read_bytes(), before)

    def test_reserved_file_requires_flag_and_no_overwrite(self):
        run = self.fixture()
        path = run / "text-handoff.json"
        self.assertEqual(self.export(run, path, allow=False).returncode, 2)
        self.assertEqual(self.export(run, path).returncode, 0)
        before = path.read_bytes()
        self.assertEqual(self.export(run, path).returncode, 2)
        self.assertEqual(path.read_bytes(), before)

    def test_successful_publication_rejects_handoff_symlink(self):
        run = self.fixture()
        (run / "text-handoff.json").symlink_to(run / "reference_summary.json")
        with self.assertRaisesRegex(ValueError, "软链接"):
            publisher.publish(self.root, run, 0, 0)


if __name__ == "__main__":
    unittest.main()
