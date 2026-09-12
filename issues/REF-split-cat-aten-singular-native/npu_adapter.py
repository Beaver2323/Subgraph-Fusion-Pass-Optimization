#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys
if __name__ == '__main__':
    sys.argv[1:1] = ['--task', 'T-088', '--case', 'REF-split-cat-aten-singular-native']
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'runners/t088_t090_npu_case.py'), run_name='__main__')
