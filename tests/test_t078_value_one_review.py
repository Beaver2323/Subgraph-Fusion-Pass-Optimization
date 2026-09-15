"""T-078 增量必须绑定正确 dtype/value、负命中结构和原始数值证据。"""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import review_t078_value_one as review


class ValueOneReviewTests(unittest.TestCase):
    def test_current_archive_is_reproducible(self):
        actual = review.review()
        self.assertEqual(actual, json.loads((ROOT / review.OUTPUT).read_text()))
        self.assertFalse(actual["new_device_execution"])
        self.assertEqual(actual["actual_addcdiv_fma_fused"], 0)

    def test_wrong_dtype_and_false_fma_are_rejected(self):
        payload = review.handoff.load_input(ROOT / review.INPUT)
        run, files = review.handoff.validate_payload(payload)
        for mutation in ("dtype", "fma", "bitwise"):
            changed = copy.deepcopy(files)
            prefix = f"cases/{review.CASE}/"
            if mutation == "dtype":
                result = json.loads(changed[prefix + "reference_result.json"])
                result["execution"]["command"][-3] = "bfloat16"
                changed[prefix + "reference_result.json"] = json.dumps(result).encode()
            elif mutation == "fma":
                changed[prefix + "fx_after.txt"] += b"\ntorch.ops.aten.addcdiv.default\n"
            else:
                changed[prefix + "stdout.log"] = b"dtype=float16 value=1.0 bitwise_equal=False max_abs_error=0.01\n"
            with self.subTest(mutation=mutation), patch.object(review.handoff, "validate_payload", return_value=(run, changed)):
                with self.assertRaises(ValueError):
                    review.review()

    def test_foreign_run_cannot_replace_reviewed_run(self):
        payload = review.handoff.load_input(ROOT / review.INPUT)
        run, files = review.handoff.validate_payload(payload)
        payload["payload_sha256"] = "0" * 64
        with patch.object(review.handoff, "load_input", return_value=payload), patch.object(review.handoff, "validate_payload", return_value=(run, files)):
            with self.assertRaises(ValueError):
                review.review()
