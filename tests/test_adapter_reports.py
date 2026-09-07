"""适配代码与中文适配报告的一致性回归；从 /home/z50063656/tmp 执行。"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AdapterReportTests(unittest.TestCase):
    def test_every_adapter_has_a_contract_report(self):
        adapters = sorted((ROOT / "issues").glob("REF-*/npu_adapter.py"))
        self.assertEqual(len(adapters), 28)
        missing = [str(path.parent.relative_to(ROOT)) for path in adapters if not (path.parent / "适配报告.md").is_file()]
        self.assertEqual(missing, [])

    def test_reports_have_required_learning_and_audit_sections(self):
        required = (
            "## 1. 原生阻断",
            "## 2. 最小适配",
            "## 3. 必要调用链",
            "## 4. 结果与边界",
            "npu_adapter.py",
            "`triton_experimental`",
            "product_gate_bypassed=false",
            "```python",
        )
        timestamp = re.compile(r"2026-09-07 11:20 CST（UTC\+08:00）")
        incomplete = []
        for adapter in sorted((ROOT / "issues").glob("REF-*/npu_adapter.py")):
            report = adapter.parent / "适配报告.md"
            text = report.read_text(encoding="utf-8")
            missing = [item for item in required if item not in text]
            if missing or not timestamp.search(text):
                incomplete.append(f"{report.relative_to(ROOT)}: {missing or ['timestamp']}")
        self.assertEqual(incomplete, [])


if __name__ == "__main__":
    unittest.main()
