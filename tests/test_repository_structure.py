"""仓库文件树收束回归；从 /home/z50063656/tmp 执行。"""

from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "report"
LEGACY_ROOT = REPORT_ROOT / "archive/legacy-20260820-0828"


class RepositoryStructureTests(unittest.TestCase):
    def test_legacy_reports_stay_out_of_current_report_root(self):
        legacy_task = re.compile(r"^t(?:0(?:1[2-9]|[2-6][0-9]|7[0-2]))(?:_|$)")
        misplaced = [
            path.name
            for path in REPORT_ROOT.iterdir()
            if path.name.startswith("p0_")
            or path.name in {
                "pass_inventory.md",
                "pass_src_20260820",
                "triton_experimental_20260826",
            }
            or legacy_task.match(path.name)
        ]
        self.assertEqual(misplaced, [])
        self.assertTrue((LEGACY_ROOT / "pass_inventory.md").is_file())
        self.assertTrue((LEGACY_ROOT / "t072_matmul_fold_20260828.md").is_file())

    def test_active_files_do_not_reference_pre_archive_report_paths(self):
        legacy_names = [path.name for path in LEGACY_ROOT.iterdir()]
        needles = [f"report/{name}" for name in legacy_names]
        roots = [
            ROOT / "README.md",
            ROOT / "TODO.md",
            ROOT / "WORKFLOW.md",
            ROOT / "docs",
            ROOT / "upstream",
            ROOT / "scripts",
            ROOT / "runners",
            REPORT_ROOT,
        ]
        suffixes = {".csv", ".json", ".md", ".py", ".sh", ".yaml", ".yml"}
        stale = []
        for source_root in roots:
            paths = [source_root] if source_root.is_file() else source_root.rglob("*")
            for path in paths:
                if not path.is_file() or path.suffix not in suffixes:
                    continue
                if LEGACY_ROOT in path.parents or ROOT / "docs/archive" in path.parents:
                    continue
                text = path.read_text(encoding="utf-8")
                for needle in needles:
                    if needle in text:
                        stale.append(f"{path.relative_to(ROOT)} -> {needle}")
        self.assertEqual(stale, [])

    def test_nonempty_incoming_directories_have_no_placeholders(self):
        incoming = ROOT / "results/incoming"
        for task_dir in sorted(path for path in incoming.glob("T-*") if path.is_dir()):
            real_files = [path for path in task_dir.iterdir() if path.name != ".gitkeep"]
            if real_files:
                self.assertFalse((task_dir / ".gitkeep").exists(), task_dir.name)

        self.assertFalse((incoming / "T-076/text-handoff.json").exists())
        self.assertTrue((incoming / "T-076/manifest.json").is_file())
        self.assertEqual(len(list((incoming / "T-076").glob("part-*.json"))), 5)


if __name__ == "__main__":
    unittest.main()
