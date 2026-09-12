#!/usr/bin/env python3
"""T-096 REF-e8m0-log2-one-ulp-native：共享只读观察器入口。"""
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[2]
sys.argv.extend(['--case', 'REF-e8m0-log2-one-ulp-native'])
runpy.run_path(str(root/'runners/t096_npu_case.py'), run_name='__main__')
