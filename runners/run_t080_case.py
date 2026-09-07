#!/usr/bin/env python3
"""T-080 单 case 原生预检/最小适配 fresh-process 调度器。"""

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
SCATTER_SOURCE = Path(
    "/home/z50063656/Pass/src/pytorch/test/inductor/test_scatter_optimization.py"
)
ONLINE_SOFTMAX_SOURCE = Path(
    "/home/z50063656/Pass/src/pytorch/test/inductor/test_online_softmax.py"
)
TORCHINDUCTOR_SOURCE = Path(
    "/home/z50063656/Pass/src/pytorch/test/inductor/test_torchinductor.py"
)
METHODS = {
    "REF-scatter-const-3d-native": (SCATTER_SOURCE, "TestScatterOpt.test_3d_tensor"),
    "REF-scatter-const-non-last-dim-native": (SCATTER_SOURCE, "TestScatterOpt.test_non_last_dim"),
    "REF-scatter-const-negative-dim-native": (SCATTER_SOURCE, "TestScatterOpt.test_neg_scatter_dim"),
    "REF-scatter-const-short-index-negative-native": (SCATTER_SOURCE, "TestScatterOpt.test_shorter_index_tensor"),
    "REF-scatter-const-dense-negative-native": (SCATTER_SOURCE, "TestScatterOpt.test_can_not_optimize_due_to_dense"),
    "REF-scatter-const-nonconst-negative-native": (SCATTER_SOURCE, "TestScatterOpt.test_can_not_optimize_due_to_non_const"),
    "REF-scatter-const-dtype-regression-native": (SCATTER_SOURCE, "TestScatterOpt.test_dtype_preserved"),
    "REF-scatter-const-cross-entropy-e2e-native": (SCATTER_SOURCE, "TestScatterOpt.test_cross_entropy_loss"),
    "REF-prepare-softmax-fast-math-native": (TORCHINDUCTOR_SOURCE, "GPUTests.test_prepare_softmax_with_fast_math_cuda"),
    "REF-prepare-softmax-signed-zero-native": (ONLINE_SOFTMAX_SOURCE, "TestOnlineSoftmax.test_prepare_softmax_signed_zero"),
    "REF-prepare-softmax-community-perf-native": (ONLINE_SOFTMAX_SOURCE, "TestOnlineSoftmax.test_prepare_softmax_perf"),
    "REF-move-constructors-arange-native": (TORCHINDUCTOR_SOURCE, "GPUTests.test_move_arange_cuda"),
    "REF-move-constructors-index-put-negative-native": (TORCHINDUCTOR_SOURCE, "TritonCodeGenTests.test_ctr_not_moved_to_cuda_when_used_in_index_put"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", choices=sorted(METHODS), required=True)
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument("--enable-softmax-probe", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != WORK:
        raise SystemExit("必须从 /home/z50063656/tmp 启动")

    result_root = WORK / "t080-npu-results" / args.case_id
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
            str(ROOT / "runners" / "t080_npu_case.py"),
            "--case-id",
            args.case_id,
            "--artifact-dir",
            str(run),
        ]
        if args.enable_softmax_probe:
            command.append("--enable-softmax-probe")
    else:
        source, method = METHODS[args.case_id]
        command = [sys.executable, str(source), method]

    with (run / "stdout.log").open("w") as out, (run / "stderr.log").open("w") as err:
        try:
            completed = subprocess.run(
                command, cwd=WORK, env=env, stdout=out, stderr=err, timeout=1200
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            return_code = 124
    stdout = (run / "stdout.log").read_text(errors="replace")
    stderr = (run / "stderr.log").read_text(errors="replace")
    generated_gpu_class_absent = (
        not args.adapter
        and args.case_id
        in {
            "REF-prepare-softmax-fast-math-native",
            "REF-move-constructors-arange-native",
            "REF-move-constructors-index-put-negative-native",
        }
        and (
            "has no attribute 'GPUTests'" in stderr
            or "has no attribute 'TritonCodeGenTests'" in stderr
        )
    )
    status = (
        "adapter-complete"
        if args.adapter and return_code == 0
        else "no-tests"
        if not args.adapter
        and ((return_code == 0 and not stdout.strip()) or generated_gpu_class_absent)
        else "failed"
    )
    summary = {
        "task_id": "T-080",
        "case_id": args.case_id,
        "source_test": f"{METHODS[args.case_id][0]}::{METHODS[args.case_id][1]}",
        "mode": "adapter" if args.adapter else "direct",
        "status": status,
        "return_code": return_code,
        "tests_run": None if args.adapter else 0,
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
