"""仅日志补证的真实编码/哈希/目录边界回归。"""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_history_logs as logs


class HistoryLogSupplementTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir="/home/z50063656/tmp")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        case = self.root / "cases/case"
        case.mkdir(parents=True)
        (self.root / "reference_summary.json").write_text(json.dumps({"run_id": logs.RUNS["T-076"], "cases": [{"case_id": "case"}]}))
        inventory = []
        for name, text in (("stdout.log", ""), ("stderr.log", "Ran 1 test in 1s\nOK\n")):
            data = text.encode()
            (case / name).write_bytes(data)
            inventory.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        (case / "artifact_inventory.json").write_text(json.dumps(inventory))

    def test_roundtrip_omits_empty_log_and_keeps_exact_newlines(self):
        payload = logs.build_logs("T-076", self.root)
        data = logs.validate_logs(payload, "T-076", logs.RUNS["T-076"])
        self.assertEqual(data, {"cases/case/stderr.log": b"Ran 1 test in 1s\nOK\n"})
        self.assertEqual(payload["empty_logs"], ["cases/case/stdout.log"])

    def test_historical_file_mutation_rejected_before_export(self):
        (self.root / "cases/case/stderr.log").write_text("new result")
        with self.assertRaisesRegex(ValueError, "inventory"):
            logs.build_logs("T-076", self.root)

    def test_other_run_cannot_be_used_as_historical_evidence(self):
        (self.root / "reference_summary.json").write_text(json.dumps({"run_id": "new-run", "cases": []}))
        with self.assertRaisesRegex(ValueError, "原始 run"):
            logs.build_logs("T-076", self.root)

    def test_traversal_even_with_valid_seal_rejected(self):
        payload = logs.build_logs("T-076", self.root)
        payload["files"][0]["path"] = "cases/../stderr.log"
        logs.seal_payload(payload)
        with self.assertRaisesRegex(ValueError, "只允许"):
            logs.validate_logs(payload, "T-076", logs.RUNS["T-076"])

    def test_changed_compressed_content_rejected(self):
        payload = logs.build_logs("T-076", self.root)
        payload["files"][0]["sha256"] = "0" * 64
        logs.seal_payload(payload)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            logs.validate_logs(payload, "T-076", logs.RUNS["T-076"])


if __name__ == "__main__":
    unittest.main()
