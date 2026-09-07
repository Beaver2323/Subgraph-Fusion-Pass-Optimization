#!/usr/bin/env python3
"""T-079 单 case 原生预检/最小适配 fresh-process 调度器。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


WORK = Path("/home/z50063656/tmp")
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/home/z50063656/Pass/src/pytorch/test/inductor/test_pattern_matcher.py"
)
METHODS = {
    "REF-bmm-to-mm-native": "TestPatternMatcher.test_bmm_to_mm",
    "REF-cat-slice-cat-native": "TestPatternMatcher.test_cat_slice_cat_cuda",
    "REF-splitwithsizes-cat-native": "TestPatternMatcher.test_splitwithsizes_cat",
    "REF-cat-splitwithsizes-native": "TestPatternMatcher.test_cat_splitwithsizes",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", choices=sorted(METHODS), required=True)
    parser.add_argument("--adapter", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != WORK:
        raise SystemExit("必须从 /home/z50063656/tmp 启动")

    result_root = WORK / "t079-npu-results" / args.case_id
    result_root.mkdir(parents=True, exist_ok=True)
    run = Path(
        tempfile.mkdtemp(prefix="adapter-" if args.adapter else "direct-", dir=result_root)
    )
    env = dict(
        os.environ,
        TORCHINDUCTOR_NPU_BACKEND="triton_experimental",
        TORCH_DEVICE_BACKEND_AUTOLOAD="1",
        TORCHINDUCTOR_FORCE_DISABLE_CACHES="1",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        TORCH_COMPILE_DEBUG="1",
        TORCH_COMPILE_DEBUG_DIR=str(run / "debug"),
        TORCH_TRACE=str(run / "trace"),
        TORCHINDUCTOR_CACHE_DIR=str(run / "inductor-cache"),
        TRITON_CACHE_DIR=str(run / "triton-cache"),
    )
    if args.adapter:
        command = [
            sys.executable,
            str(ROOT / "runners" / "t079_npu_case.py"),
            "--case-id",
            args.case_id,
            "--artifact-dir",
            str(run),
        ]
    else:
        # 原生模块的 __main__ 会因 HAS_GPU=false 静默执行 0 tests。输出为空且
        # return_code=0 也必须分类为 NO_TESTS，不能记为 PASS。
        command = [sys.executable, str(SOURCE), METHODS[args.case_id]]

    with (run / "stdout.log").open("w") as out, (run / "stderr.log").open("w") as err:
        try:
            completed = subprocess.run(
                command, cwd=WORK, env=env, stdout=out, stderr=err, timeout=900
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            return_code = 124
    stdout = (run / "stdout.log").read_text(errors="replace")
    tests_run = None if args.adapter else 0
    status = (
        "adapter-complete" if args.adapter and return_code == 0
        else "no-tests" if not args.adapter and return_code == 0 and not stdout.strip()
        else "failed"
    )
    summary = {
        "task_id": "T-079",
        "case_id": args.case_id,
        "source_test": f"{SOURCE}::{METHODS[args.case_id]}",
        "mode": "adapter" if args.adapter else "direct",
        "status": status,
        "return_code": return_code,
        "tests_run": tests_run,
        "raw_logs": str(run),
        "body_or_assertions_modified": False,
    }
    (run / "run_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return return_code if args.adapter else (0 if status == "no-tests" else return_code)


if __name__ == "__main__":
    raise SystemExit(main())
