"""未来批次准备与失败关闭门禁；标准库测试，不导入torch。"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PYTORCH = Path("/home/z50063656/Pass/src/pytorch")


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


reference = module("future_reference", "runners/reference_runner.py")
worker = module("future_worker", "runners/t081_t083_performance_worker.py")
launcher = module("future_launcher", "scripts/run_prepared_performance.py")


class FutureReferenceTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / "upstream/t081_manifest.yaml").read_text())
        self.plan = json.loads((ROOT / "upstream/t081_reference_plan.yaml").read_text())

    def test_all_native_entrypoints_resolve_without_torch(self):
        frozen_counts = {81: 2, 82: 2, 83: 3}
        for task in (81, 82, 83):
            manifest = json.loads((ROOT / f"upstream/t0{task}_manifest.yaml").read_text())
            plan = json.loads((ROOT / f"upstream/t0{task}_reference_plan.yaml").read_text())
            reference.validate_contract(manifest, plan, PYTORCH)
            self.assertEqual(
                manifest["counting_policy"]["current_frozen_denominator_units"],
                frozen_counts[task],
            )

    def test_observer_rejects_non_boolean_or_unreviewed_task(self):
        for value in (False, 1, "true"):
            plan = copy.deepcopy(self.plan)
            plan["cases"][0]["native_observer"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "native_observer"):
                reference.validate_contract(self.manifest, plan, PYTORCH)
        self.plan["task_id"] = "T-080"
        with self.assertRaisesRegex(ValueError, "未审核该任务"):
            reference.validate_contract(self.manifest, self.plan, PYTORCH)

    def test_observer_preserves_exact_native_selectors(self):
        case = self.plan["cases"][-1]
        command = reference.case_command(case, ROOT, PYTORCH)
        self.assertEqual(command[1], str(ROOT / "runners/native_fx_observer.py"))
        self.assertEqual(command[command.index("--") + 1:], case["direct_args"])
        self.assertNotIn("--adapter", command)

    def test_observer_renders_current_graph_without_recompile_side_effect(self):
        source = (ROOT / "runners/native_fx_observer.py").read_text()
        self.assertIn('gm.graph.python_code(root_module="self").src', source)
        self.assertNotIn("output.recompile()", source)

    def test_parameter_name_typo_or_missing_variant_fails(self):
        for mutation in (lambda names: names.pop(), lambda names: names.__setitem__(0, names[0] + "_typo")):
            plan = copy.deepcopy(self.plan)
            mutation(plan["cases"][-1]["direct_args"])
            with self.assertRaisesRegex(ValueError, "参数化入口"):
                reference.validate_contract(self.manifest, plan, PYTORCH)

    def test_device_instantiation_does_not_create_cuda_for_cpu_only(self):
        with tempfile.TemporaryDirectory(dir="/home/z50063656/tmp") as temporary:
            source = Path(temporary) / "test_torchinductor_dynamic_shapes.py"
            template = "class TestInductorDynamic:\n    def test_a(self, device): pass\ninstantiate_device_type_tests(TestInductorDynamic, globals(), {} )\n"
            for restriction in ("only_for=['cpu']", "except_for=['cuda']"):
                source.write_text(template.format(restriction))
                self.assertNotIn("TestInductorDynamicCUDA.test_a_cuda", reference.python_qualnames(source))
            source.write_text(template.format("allow_xpu=True"))
            self.assertIn("TestInductorDynamicCUDA.test_a_cuda", reference.python_qualnames(source))


class FuturePerformanceGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir="/home/z50063656/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.unit = "all-gather"
        task, unit_id, _ = worker.TARGETS[self.unit]
        self.gate = {
            "task_id": task, "acceptance_unit_id": unit_id, "backend": "triton_experimental",
            "pytorch_commit": worker.COMMIT, "correctness": "passed", "target_rewrite": "confirmed",
            "graph_breaks": 0, "fallbacks": 0, "product_disabled": False,
            "measurement_workload": self.unit + "-community-shape", "world_size": 2,
            "reviewed_at": "2026-09-07T22:36:49+08:00", "reviewer": "test-fixture-only",
            "input_spec": worker.input_spec(self.unit),
            "worker_sha256": hashlib.sha256((ROOT / "runners/t081_t083_performance_worker.py").read_bytes()).hexdigest(),
        }
        self.functional = dict(self.gate, numerical_execution=True, process_group_backend="hccl",
                               source_files={str(ROOT / "runners/t081_t083_performance_worker.py"): self.gate["worker_sha256"]})
        self.gpu = {"status": "valid-reference-suite", "suite_valid": True,
                    "cases": [{"acceptance_unit_id": unit_id, "reference_valid": True}]}

    def write_gate(self):
        for key, data in (("gpu_reference", self.gpu), ("target_functional", self.functional)):
            path = self.directory / f"{key}.json"
            path.write_text(json.dumps(data))
            self.gate[key] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        path = self.directory / "gate.json"
        path.write_text(json.dumps(self.gate))
        return path

    def test_matching_review_and_originals_are_accepted(self):
        self.assertEqual(worker.read_gate(self.write_gate(), self.unit, "npu")["world_size"], 2)

    def test_false_sidecar_cannot_override_original_failure(self):
        for field, value in (("fallbacks", 1), ("graph_breaks", 1), ("product_disabled", True),
                             ("numerical_execution", False), ("process_group_backend", "fake"),
                             ("backend", "default"), ("world_size", 1)):
            original = self.functional[field]
            self.functional[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                worker.read_gate(self.write_gate(), self.unit, "npu")
            self.functional[field] = original

    def test_changed_worker_or_input_invalidates_gate(self):
        for field, value in (("worker_sha256", "0" * 64), ("input_spec", [])):
            original = self.gate[field]
            self.gate[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                worker.read_gate(self.write_gate(), self.unit, "npu")
            self.gate[field] = original

    def test_missing_changed_source_is_rejected(self):
        self.functional["source_files"] = {str(self.directory / "absent.py"): "0" * 64}
        with self.assertRaisesRegex(ValueError, "当前源码"):
            worker.read_gate(self.write_gate(), self.unit, "npu")

    def test_distributed_timeout_terminates_only_created_process_group(self):
        class FakeProcess:
            pid = 123456
            def __init__(self): self.calls = 0
            def wait(self, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("owned-test", timeout)
                return -15
        child = FakeProcess()
        with patch.object(launcher.subprocess, "Popen", return_value=child) as start, patch.object(launcher.os, "killpg") as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                launcher.run_arm(["owned-test"], self.directory, None, None, timeout=1)
            self.assertTrue(start.call_args.kwargs["start_new_session"])
            kill.assert_called_once_with(child.pid, launcher.signal.SIGTERM)

    def test_formal_launcher_serializes_tracker_performance(self):
        source = (ROOT / "scripts/run_t081_t083_performance.sh").read_text()
        self.assertIn("pass-tracker-npu-performance.lock", source)
        self.assertIn("flock -n 9", source)

    def test_committed_functional_and_performance_summaries_are_complete(self):
        expected = {"T-081": 2, "T-082": 2, "T-083": 3}
        for task, count in expected.items():
            root = ROOT / "results/current" / task
            functional = json.loads((root / "npu_functional_summary.json").read_text())
            performance = json.loads((root / "performance_summary.json").read_text())
            self.assertEqual(functional["backend"], "triton_experimental")
            self.assertEqual(performance["backend"], "triton_experimental")
            self.assertEqual(len(functional["units"]), count)
            self.assertTrue(
                all(
                    unit["status"] == "functional-passed-performance-gate-signed"
                    for unit in functional["units"]
                )
            )
            self.assertEqual(performance["completion"]["acceptance_units"], count)
            self.assertEqual(performance["completion"]["pending_units"], 0)


if __name__ == "__main__":
    unittest.main()
