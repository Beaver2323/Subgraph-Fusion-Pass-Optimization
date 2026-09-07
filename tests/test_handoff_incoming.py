"""接收目录与 Git 忽略边界的零设备回归；从 /home/z50063656/tmp 执行。"""

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/home/z50063656/tmp")
TASKS = ("T-076", "T-077", "T-078", "T-079", "T-080")


class HandoffIncomingTests(unittest.TestCase):
    def ignored(self, paths):
        result = subprocess.run(
            ["git", "-C", str(ROOT), "check-ignore", "--no-index", "--stdin"],
            input="\n".join(paths) + "\n", text=True, capture_output=True,
            cwd=WORK, check=False,
        )
        self.assertIn(result.returncode, (0, 1), result.stderr)
        return set(result.stdout.splitlines())

    def test_nonempty_task_directories_and_readme_exist(self):
        readme = "results/incoming/README.md"
        self.assertTrue((ROOT / readme).is_file(), readme)
        self.assertEqual(self.ignored([readme]), set())
        for task in TASKS:
            task_dir = ROOT / "results/incoming" / task
            self.assertTrue(task_dir.is_dir(), task)
            self.assertTrue(any(task_dir.iterdir()), task)
            self.assertFalse((task_dir / ".gitkeep").exists(), task)

    def test_handoff_logs_and_nested_payloads_remain_ignored(self):
        paths = ["results/incoming/text-handoff.json", "results/incoming/unknown/README.md"]
        for task in TASKS:
            paths.extend(f"results/incoming/{task}/{name}" for name in (
                "text-handoff.json", "stderr.log", "raw/environment.json", "bundle.tar.gz",
            ))
        self.assertEqual(self.ignored(paths), set(paths))


if __name__ == "__main__":
    unittest.main()
