"""原文文本回传的零设备测试；不执行回传的 Python/内核代码。"""

import copy
import base64
import hashlib
import importlib.util
import json
import lzma
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
        (case / "benchmark.json").write_text("{}\n")
        (case / "metadata.json").write_text("{}\n")
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

    def payload(self, *, compress=False, profile="archive"):
        payload = exporter.build_payload(self.run)
        exporter.include_raw_text(
            payload, self.run, compress=compress, profile=profile
        )
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

    def test_summary_handoff_does_not_pretend_to_restore_fx(self):
        with self.assertRaisesRegex(ValueError, "1.1"):
            importer.validate_payload(exporter.build_payload(self.run))

    def test_compressed_round_trip_restores_exact_text(self):
        payload = self.payload(compress=True)
        self.assertEqual(payload["handoff_format_version"], "1.2")
        self.assertTrue(
            all(item["encoding"] == "zlib+base64" for item in payload["raw_text_files"])
        )
        restored = importer.restore(payload, self.root / "compressed-imports")
        for item in payload["raw_text_files"]:
            self.assertEqual(
                (restored / item["path"]).read_bytes(),
                (self.run / item["path"]).read_bytes(),
            )

    def test_review_profile_keeps_audit_text_and_hashes_large_artifacts(self):
        payload = self.payload(compress=True, profile="review")
        self.assertEqual(payload["handoff_format_version"], "1.3")
        self.assertEqual(payload["handoff_profile"], "review")
        embedded = {item["path"] for item in payload["raw_text_files"]}
        self.assertIn("cases/case/fx_before.txt", embedded)
        self.assertIn("cases/case/fx_after.txt", embedded)
        self.assertIn("cases/case/stderr.log", embedded)
        self.assertIn("cases/case/stdout.log", embedded)
        self.assertIn("cases/case/cache/output_code.py", embedded)
        omitted = {
            item["path"]: item["reason"]
            for item in payload["raw_text_transfer"]["omitted_files"]
        }
        self.assertEqual(
            omitted["cases/case/cache/kernel.cubin"],
            "review-profile-hash-only",
        )
        restored = importer.restore(payload, self.root / "review-imports")
        self.assertTrue((restored / "cases/case/fx_before.txt").is_file())
        self.assertTrue((restored / "cases/case/stderr.log").is_file())
        self.assertTrue((restored / "cases/case/cache/output_code.py").is_file())

    def test_bundle_round_trip_deduplicates_and_preserves_every_file(self):
        for profile in ("review", "archive"):
            with self.subTest(profile=profile):
                old = self.payload(compress=True, profile=profile)
                expected_id, expected = importer.validate_payload(old)
                payload = copy.deepcopy(old)
                exporter.bundle_raw_text(payload)
                self.assertEqual(payload["handoff_format_version"], "1.4")
                run_id, data = importer.validate_payload(payload)
                self.assertEqual((run_id, data), (expected_id, expected))
                self.assertEqual(payload["evidence_files"], old["evidence_files"])
                self.assertLess(payload["raw_text_bundle"]["bytes"], old["raw_text_transfer"]["embedded_bytes"])
                restored = importer.restore(payload, self.root / profile)
                for path, content in expected.items():
                    self.assertEqual((restored / path).read_bytes(), content)
                parts = self.root / (profile + "-parts")
                exporter.write_split_payload(
                    json.dumps(payload).encode(), parts,
                    payload_sha256=payload["payload_sha256"], part_bytes=16384,
                )
                self.assertEqual(importer.validate_payload(importer.load_input(parts / "manifest.json")), (run_id, data))

    def test_bundle_rejects_corruption_and_invalid_ranges(self):
        original = self.payload(compress=True, profile="review")
        exporter.bundle_raw_text(original)
        for mode in ("base64", "truncated", "trailing", "size", "limit", "hash", "offset", "file-hash", "statistics"):
            with self.subTest(mode=mode):
                payload = copy.deepcopy(original)
                bundle = payload["raw_text_bundle"]
                if mode == "base64":
                    bundle["data"] = "!" + bundle["data"][1:]
                elif mode in {"truncated", "trailing"}:
                    compressed = base64.b64decode(bundle["data"])
                    compressed = compressed[:-1] if mode == "truncated" else compressed + lzma.compress(b"extra")
                    bundle.update(data=base64.b64encode(compressed).decode(), compressed_bytes=len(compressed))
                elif mode == "size":
                    bundle["bytes"] -= 1
                elif mode == "limit":
                    bundle["bytes"] = exporter.MAX_TEXT_BYTES + 1
                elif mode == "hash":
                    bundle["sha256"] = "0" * 64
                elif mode == "offset":
                    payload["raw_text_files"][0]["offset"] = 1
                elif mode == "file-hash":
                    payload["raw_text_files"][0]["sha256"] = "0" * 64
                else:
                    payload["raw_text_transfer"]["unique_bytes"] += 1
                exporter.seal_payload(payload)
                with self.assertRaises(ValueError):
                    importer.validate_payload(payload)

    def test_bundle_cli_validates_without_gpu(self):
        output = self.root / "bundle.json"
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/export_reference_text.py"),
             "--run-dir", str(self.run), "--profile", "review", "--bundle-raw-text",
             "--compact", "--output", str(output)],
            cwd=WORK, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = importer.load_input(output)
        self.assertEqual(payload["handoff_format_version"], "1.4")
        importer.validate_payload(payload)

    def test_reexport_task_preserves_source_and_uses_latest(self):
        (self.root / "latest").symlink_to(self.run.name)
        before = {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        command = [sys.executable, str(ROOT / "scripts/reexport_reference_text.py"),
                   "--task", "T-098", "--result-root", str(self.root)]
        outputs = []
        for _ in range(2):
            result = subprocess.run(command, cwd=WORK, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            outputs.append(next(line.split("=", 1)[1] for line in result.stdout.splitlines()
                                if line.startswith("handoff_upload_input=")))
            importer.validate_payload(importer.load_input(Path(outputs[-1])))
        self.assertNotEqual(*outputs)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()})
        self.assertEqual((self.root / "latest").resolve(), self.run)

    def test_legacy_split_still_imports(self):
        payload = self.payload(compress=True, profile="review")
        parts = self.root / "legacy-parts"
        manifest = exporter.write_split_payload(
            json.dumps(payload).encode(), parts, payload_sha256=payload["payload_sha256"],
            part_bytes=16384, compress_transport=False,
        )
        self.assertEqual(manifest["split_format_version"], "1.0")
        self.assertEqual(importer.validate_payload(importer.load_input(parts / "manifest.json")),
                         importer.validate_payload(payload))

    def test_compressed_split_rejects_invalid_stream_even_with_resealed_transport(self):
        payload = self.payload(compress=True, profile="review")
        content = json.dumps(payload).encode()
        for mode in ("truncated", "trailing", "size", "limit"):
            with self.subTest(mode=mode):
                parts = self.root / mode
                manifest = exporter.write_split_payload(content, parts,
                    payload_sha256=payload["payload_sha256"], part_bytes=49152)
                self.assertEqual(manifest["part_count"], 1)
                part_path = parts / "part-0001.json"
                part = json.loads(part_path.read_text())
                raw = base64.b64decode("".join(part["data"]))
                if mode in {"truncated", "trailing"}:
                    raw = raw[:-1] if mode == "truncated" else raw + lzma.compress(b"extra")
                    digest = hashlib.sha256(raw).hexdigest()
                    part.update(data=[base64.b64encode(raw).decode()], payload_bytes=len(raw), payload_sha256=digest)
                    manifest["parts"][0].update(payload_bytes=len(raw), payload_sha256=digest)
                    manifest.update(transport_bytes=len(raw), transport_sha256=digest)
                elif mode == "size":
                    manifest["source_bytes"] -= 1
                else:
                    manifest["source_bytes"] = importer.MAX_HANDOFF_TRANSPORT_BYTES + 1
                part_path.write_text(json.dumps(part))
                (parts / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    importer.load_input(parts / "manifest.json")

    def test_review_profile_includes_failure_logs(self):
        result_path = self.run / "cases/case/reference_result.json"
        result = json.loads(result_path.read_text())
        result["execution"]["status"] = "failed"
        result["execution"]["return_code"] = 1
        result["reference_valid"] = False
        result_path.write_text(json.dumps(result) + "\n")
        payload = self.payload(compress=True, profile="review")
        embedded = {item["path"] for item in payload["raw_text_files"]}
        self.assertIn("cases/case/stderr.log", embedded)
        self.assertIn("cases/case/stdout.log", embedded)
        importer.validate_payload(payload)

    def test_compressed_payload_tampering_is_rejected(self):
        payload = self.payload(compress=True)
        payload["raw_text_files"][0]["data"] += "!"
        exporter.seal_payload(payload)
        with self.assertRaisesRegex(ValueError, "Base64"):
            importer.validate_payload(payload)

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
        parts = self.root / "unused-parts"
        export = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_reference_text.py"),
                "--run-dir",
                str(self.run),
                "--profile",
                "review",
                "--compact",
                "--output",
                str(output),
                "--split-output-dir",
                str(parts),
                "--auto-split-over-bytes",
                str(1024 * 1024),
            ],
            cwd=WORK,
            capture_output=True,
            text=True,
        )
        self.assertEqual(export.returncode, 0, export.stderr)
        output_text = output.read_text()
        self.assertEqual(output_text.count("\n"), 1)
        self.assertFalse(parts.exists())
        self.assertIn("handoff_upload_mode=single-file", export.stdout)
        self.assertEqual(json.loads(output_text)["handoff_format_version"], "1.3")
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

    def test_split_cli_round_trip_and_tampering_rejected(self):
        case = self.run / "cases/case"
        (case / "cache/large.log").write_text(
            "".join(
                hashlib.sha256(str(index).encode()).hexdigest() + "\n"
                for index in range(5000)
            )
        )
        excluded = {"artifact_inventory.json", "reference_result.json"}
        inventory = [
            exporter.file_record(path, case)
            for path in sorted(case.rglob("*"))
            if path.is_file() and path.name not in excluded
        ]
        self.write("cases/case/artifact_inventory.json", inventory)
        parts = self.root / "handoff-parts"
        export = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_reference_text.py"),
                "--run-dir",
                str(self.run),
                "--include-raw-text",
                "--compress-raw-text",
                "--split-output-dir",
                str(parts),
                "--auto-split-over-bytes",
                "1",
            ],
            cwd=WORK,
            capture_output=True,
            text=True,
        )
        self.assertEqual(export.returncode, 0, export.stderr)
        self.assertIn("handoff_upload_mode=split", export.stdout)
        manifest = json.loads((parts / "manifest.json").read_text())
        self.assertGreater(manifest["part_count"], 1)
        self.assertLessEqual(
            max(item["transport_file_bytes"] for item in manifest["parts"]),
            70 * 1024,
        )
        validation = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/import_reference_text.py"),
                "--input",
                str(parts / "manifest.json"),
                "--validate-only",
            ],
            cwd=WORK,
            capture_output=True,
            text=True,
        )
        self.assertEqual(validation.returncode, 0, validation.stdout)
        restored = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/import_reference_text.py"),
                "--input",
                str(parts / "manifest.json"),
                "--output-root",
                str(self.root / "split-imports"),
            ],
            cwd=WORK,
            capture_output=True,
            text=True,
        )
        self.assertEqual(restored.returncode, 0, restored.stdout)

        first_path = parts / manifest["parts"][0]["path"]
        first = json.loads(first_path.read_text())
        first["data"][0] = "A" + first["data"][0][1:]
        first_path.write_text(json.dumps(first))
        rejected = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/import_reference_text.py"),
                "--input",
                str(parts / "manifest.json"),
                "--validate-only",
            ],
            cwd=WORK,
            capture_output=True,
            text=True,
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("不一致", rejected.stdout)


if __name__ == "__main__":
    unittest.main()
