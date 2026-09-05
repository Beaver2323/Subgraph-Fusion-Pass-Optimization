"""原文文本回传的零设备测试；不执行回传的 Python/内核代码。"""

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
WORK = Path("/home/z50063656/tmp")


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


exporter = module("export_reference_text", "scripts/export_reference_text.py")
with patch.dict(sys.modules, {"export_reference_text": exporter}):
    importer = module("import_reference_text", "scripts/import_reference_text.py")


class RawTextTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=WORK)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run = self.root / "reference-fixture"
        self.run.mkdir()
        self.write("environment.json", {"source": {"actual_commit": "a" * 40}})
        self.write(
            "reference_summary.json",
            {"run_id": self.run.name, "cases": [{"case_id": "case"}]},
        )
        self.write("manifest_snapshot.json", {})
        self.write("reference_plan_snapshot.json", {})
        case = self.run / "cases/case"
        (case / "cache").mkdir(parents=True)
        (case / "fx_before.txt").write_bytes("图 before\r\n".encode())
        (case / "fx_after.txt").write_bytes("图 after\n".encode())
        (case / "stdout.log").write_text("")
        (case / "stderr.log").write_text("Ran 1 test in 1s\nOK\n")
        (case / "cache/output_code.py").write_text(
            "raise RuntimeError('不得执行回传代码')\n"
        )
        (case / "cache/kernel.cubin").write_bytes(b"\x00\xffbinary")
        inventory = [
            exporter.file_record(path, case)
            for path in sorted(case.rglob("*"))
            if path.is_file()
        ]
        self.write("cases/case/artifact_inventory.json", inventory)
        self.write(
            "cases/case/reference_result.json",
            {
                "case": {
                    "case_id": "case",
                    "acceptance_unit_id": "AU",
                    "variant_ids": [],
                },
                "source": {"actual_commit": "a" * 40},
                "execution": {
                    "status": "passed",
                    "return_code": 0,
                    "tests_ran": 1,
                    "tests_skipped": 0,
                },
                "fx": {
                    "before": {"captured": True},
                    "after": {"captured": True},
                },
                "reference_valid": True,
            },
        )

    def write(self, relative, data):
        path = self.run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False) + "\n")

    def payload(self):
        payload = exporter.build_payload(self.run)
        exporter.include_raw_text(payload, self.run)
        return payload

    def test_round_trip_restores_exact_text_not_binary_and_never_executes(self):
        payload = self.payload()
        restored = importer.restore(payload, self.root / "imports")
        for item in payload["raw_text_files"]:
            self.assertEqual(
                (restored / item["path"]).read_bytes(),
                (self.run / item["path"]).read_bytes(),
            )
        self.assertFalse((restored / "cases/case/cache/kernel.cubin").exists())
        self.assertFalse(
            payload["raw_text_transfer"]["all_registered_artifacts_embedded"]
        )
        again = importer.restore(payload, self.root / "imports")
        self.assertNotEqual(again, restored)
        self.assertTrue((restored / "cases/case/fx_before.txt").exists())

    def test_compact_handoff_does_not_pretend_to_restore_fx(self):
        with self.assertRaisesRegex(ValueError, "1.1"):
            importer.validate_payload(exporter.build_payload(self.run))

    def test_packet_and_individual_text_tampering_rejected(self):
        payload = self.payload()
        payload["raw_text_files"][0]["text"] += "tampered"
        with self.assertRaisesRegex(ValueError, "整包"):
            importer.validate_payload(payload)
        exporter.seal_payload(payload)
        with self.assertRaisesRegex(ValueError, "大小/哈希"):
            importer.validate_payload(payload)

    def test_inventory_binding_cannot_be_replaced_by_self_reported_hash(self):
        payload = self.payload()
        item = next(
            item
            for item in payload["raw_text_files"]
            if item["path"].endswith("output_code.py")
        )
        before = item["bytes"]
        item["text"] = "forged text\n"
        item["bytes"] = len(item["text"].encode())
        item["sha256"] = hashlib.sha256(item["text"].encode()).hexdigest()
        payload["raw_text_transfer"]["embedded_bytes"] += item["bytes"] - before
        exporter.seal_payload(payload)
        with self.assertRaisesRegex(ValueError, "inventory"):
            importer.validate_payload(payload)

    def test_duplicate_and_unsafe_paths_rejected_before_writing(self):
        payload = self.payload()
        variants = []
        duplicate = copy.deepcopy(payload)
        duplicate["raw_text_files"].append(duplicate["raw_text_files"][0])
        variants.append(duplicate)
        for relative in (
            "../escape",
            "/absolute",
            ".git/config",
            "cases/../escape",
            "cases\\escape",
            "cases/newline\nname",
        ):
            altered = copy.deepcopy(payload)
            altered["raw_text_files"][0]["path"] = relative
            variants.append(altered)
        output = self.root / "never-created"
        for altered in variants:
            exporter.seal_payload(altered)
            with self.assertRaises(ValueError):
                importer.restore(altered, output)
        self.assertFalse(output.exists())

    def test_malformed_top_level_and_file_record_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            importer.validate_payload([])
        payload = self.payload()
        payload["raw_text_files"][0]["bytes"] = True
        exporter.seal_payload(payload)
        with self.assertRaisesRegex(ValueError, "非负整数"):
            importer.validate_payload(payload)

    def test_changed_original_inventory_is_not_silently_rehashed(self):
        (self.run / "cases/case/cache/output_code.py").write_text(
            "changed after inventory\n"
        )
        with self.assertRaisesRegex(ValueError, "哈希"):
            self.payload()

    def test_symlink_source_is_rejected(self):
        path = self.run / "cases/case/cache/output_code.py"
        saved = self.root / "saved.py"
        path.rename(saved)
        path.symlink_to(saved)
        with self.assertRaisesRegex(ValueError, "软链接"):
            self.payload()

    def test_real_cli_export_and_import(self):
        output = self.root / "raw-handoff.json"
        export = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_reference_text.py"),
                "--run-dir",
                str(self.run),
                "--include-raw-text",
                "--output",
                str(output),
            ],
            cwd=WORK,
            capture_output=True,
            text=True,
        )
        self.assertEqual(export.returncode, 0, export.stderr)
        validation = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/import_reference_text.py"),
                "--input",
                str(output),
                "--validate-only",
            ],
            cwd=WORK,
            capture_output=True,
            text=True,
        )
        self.assertEqual(validation.returncode, 0, validation.stdout)
        self.assertIn("handoff_validation=OK", validation.stdout)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/import_reference_text.py"),
                "--input",
                str(output),
                "--output-root",
                str(self.root / "imports"),
            ],
            cwd=WORK,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("code_executed=false", result.stdout)


if __name__ == "__main__":
    unittest.main()
