"""统一门禁必须检查嵌套 overlay；2026-09-07，无 torch/设备依赖。"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec = importlib.util.spec_from_file_location("recursive_syntax_gate", ROOT / "scripts/validate_all.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
finally:
    sys.path.pop(0)


class RecursiveSyntaxGateTests(unittest.TestCase):
    def test_nested_python_syntax_error_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/home/z50063656/tmp") as directory:
            root = Path(directory)
            overlay = root / "runners/source_overlay/sitecustomize.py"
            overlay.parent.mkdir(parents=True)
            overlay.write_text("if True:\npass\n")
            with self.assertRaises(SyntaxError):
                gate.validate_source_syntax(root, root, os.environ.copy())

    def test_nested_shell_syntax_error_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/home/z50063656/tmp") as directory:
            root = Path(directory)
            worker = root / "scripts/workers/run.sh"
            worker.parent.mkdir(parents=True)
            worker.write_text("if true; then\n")
            with self.assertRaises(subprocess.CalledProcessError):
                gate.validate_source_syntax(root, root, os.environ.copy())

    def test_valid_nested_code_is_not_executed(self):
        with tempfile.TemporaryDirectory(dir="/home/z50063656/tmp") as directory:
            root = Path(directory)
            overlay = root / "runners/source_overlay/sitecustomize.py"
            overlay.parent.mkdir(parents=True)
            overlay.write_text("raise RuntimeError('syntax check must not execute this')\n")
            gate.validate_source_syntax(root, root, os.environ.copy())


if __name__ == "__main__":
    unittest.main()
