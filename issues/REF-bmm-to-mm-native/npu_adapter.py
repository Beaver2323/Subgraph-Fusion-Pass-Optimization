#!/usr/bin/env python3
"""T-079 bmm→mm 社区用例最小 NPU 入口。"""

from pathlib import Path
import runpy
import sys


runner = Path(__file__).parents[2] / "runners" / "t079_npu_case.py"
sys.argv = [str(runner), "--case-id", "REF-bmm-to-mm-native", *sys.argv[1:]]
runpy.run_path(str(runner), run_name="__main__")
