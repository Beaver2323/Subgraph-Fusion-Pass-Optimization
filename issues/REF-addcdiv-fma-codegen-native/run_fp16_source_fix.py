#!/usr/bin/env python3
"""一键以 fresh process 验证 T-078 FP16 正式源码修复。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


WORK = Path("/home/z50063656/tmp")
TRACKER = Path("/home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization")
ISSUE = TRACKER / "issues/REF-addcdiv-fma-codegen-native"
OVERLAY = TRACKER / "runners/t078_source_overlay"
# value=1 在 decomposition 中消去乘一，不属于 div->mul->add FMA pattern。
# 它仍纳入一键回归，但期望 counter=0 且普通 div->add 位级正确。
VALUES = (0.3, 1.0, 2.0, 7.7)
GUARDS = ("integer-self-guard", "tensor-value-guard")


def run_worker(
    command: list[str], artifact_dir: Path, device: int
) -> subprocess.CompletedProcess[str]:
    env = dict(
        os.environ,
        ASCEND_RT_VISIBLE_DEVICES=str(device),
        TORCHINDUCTOR_NPU_BACKEND="triton_experimental",
        TORCH_DEVICE_BACKEND_AUTOLOAD="1",
        TORCHINDUCTOR_FORCE_DISABLE_CACHES="1",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        TORCH_COMPILE_DEBUG="1",
        TORCH_COMPILE_DEBUG_DIR=str(artifact_dir / "debug"),
        TORCH_TRACE=str(artifact_dir / "trace"),
        TORCHINDUCTOR_CACHE_DIR=str(artifact_dir / "inductor-cache"),
        TRITON_CACHE_DIR=str(artifact_dir / "triton-cache"),
        PYTHONPATH=(
            f"{OVERLAY}{os.pathsep}{os.environ['PYTHONPATH']}"
            if os.environ.get("PYTHONPATH")
            else str(OVERLAY)
        ),
    )
    completed = subprocess.run(
        command,
        cwd=WORK,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    (artifact_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (artifact_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORK / "t078-npu-results/REF-addcdiv-fma-codegen-native",
    )
    args = parser.parse_args()
    if Path.cwd().resolve() != WORK:
        raise SystemExit("必须从 /home/z50063656/tmp 启动")

    run_id = datetime.now().astimezone().strftime(
        "fp16-source-fix-%Y%m%dT%H%M%S%z"
    )
    run = args.output_root / run_id
    run.mkdir(parents=True, exist_ok=False)
    records = []

    for value in VALUES:
        tag = str(value).replace("-", "m").replace(".", "p")
        artifact_dir = run / f"value-{tag}"
        artifact_dir.mkdir(parents=True)
        completed = run_worker(
            [
                sys.executable,
                str(ISSUE / "verify_lowp_three_arm.py"),
                "--artifact-dir",
                str(artifact_dir),
                "--dtype",
                "float16",
                "--arm",
                "re-fused",
                "--value",
                str(value),
                "--source-fix",
            ],
            artifact_dir,
            args.device,
        )
        result = json.loads(
            (artifact_dir / "arm_result.json").read_text(encoding="utf-8")
        )
        result["return_code"] = completed.returncode
        records.append(result)
        print(
            f"value={value} return_code={completed.returncode} "
            f"status={result['status']}"
        )

    guards = []
    for case_name in GUARDS:
        artifact_dir = run / case_name
        artifact_dir.mkdir(parents=True)
        completed = run_worker(
            [
                sys.executable,
                str(ISSUE / "verify_neighbors.py"),
                "--artifact-dir",
                str(artifact_dir),
                "--case",
                case_name,
            ],
            artifact_dir,
            args.device,
        )
        result = json.loads(
            (artifact_dir / "neighbor_result.json").read_text(encoding="utf-8")
        )
        result["return_code"] = completed.returncode
        guards.append(result)
        print(
            f"case={case_name} return_code={completed.returncode} "
            f"status={result['status']}"
        )

    valid = all(
        item.get("status") == "passed"
        and item.get("return_code") == 0
        and item.get("bitwise_equal_to_addcdiv_eager") is True
        and item.get("addcdiv_fma_fused")
        == (0 if item.get("value") == 1 else 1)
        and item.get("codegen_contract_valid") is True
        for item in records
    ) and all(
        item.get("status") == "passed"
        and item.get("return_code") == 0
        and item.get("actual_addcdiv_fma_fused") == 0
        for item in guards
    )
    summary = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": "T-078",
        "acceptance_unit_id": "AU-post-grad-fuse-addcdiv-to-fma",
        "backend": "triton_experimental",
        "dtype": "float16",
        "status": "valid-source-fix" if valid else "invalid-source-fix",
        "fresh_process_count": len(records) + len(guards),
        "values": records,
        "guards": guards,
        "performance_measured": False,
        "performance_disposition": (
            "待单独记录；修复前 OFF 与 eager 不等价，不能作为有效收益 denominator"
        ),
    }
    (run / "fp16_source_fix_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"fp16_source_fix_status={summary['status']}")
    print(f"artifacts={run}")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
