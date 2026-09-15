#!/usr/bin/env python3
"""T-105 pattern 19 的原社区方法 NPU 入口。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'runners'))
from attention_community_npu import main

if __name__ == '__main__':
    main(19)

