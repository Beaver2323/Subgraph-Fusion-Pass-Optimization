#!/usr/bin/env python3
"""在六个 fresh process 中运行 FP16/BF16 addcdiv 三臂精度归因。"""

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


def main():
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
    run_id = datetime.now().astimezone().strftime("lowp-three-arm-%Y%m%dT%H%M%S%z")
    run = args.output_root / run_id
    run.mkdir(parents=True, exist_ok=False)
    records = []
    for dtype in ("float16", "bfloat16"):
        for arm in ("off", "decomposed", "re-fused"):
            artifact_dir = run / dtype / arm
            env = dict(
                os.environ,
                ASCEND_RT_VISIBLE_DEVICES=str(args.device),
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
            command = [
                sys.executable,
                str(ISSUE / "verify_lowp_three_arm.py"),
                "--artifact-dir",
                str(artifact_dir),
                "--dtype",
                dtype,
                "--arm",
                arm,
            ]
            completed = subprocess.run(
                command,
                cwd=WORK,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            (run / f"{dtype}-{arm}.stdout.log").write_text(
                completed.stdout, encoding="utf-8"
            )
            (run / f"{dtype}-{arm}.stderr.log").write_text(
                completed.stderr, encoding="utf-8"
            )
            result_path = artifact_dir / "arm_result.json"
            record = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.is_file()
                else {
                    "dtype": dtype,
                    "arm": arm,
                    "status": "missing-result",
                }
            )
            record["return_code"] = completed.returncode
            records.append(record)
            print(
                f"dtype={dtype} arm={arm} return_code={completed.returncode} "
                f"status={record['status']}"
            )

    by_dtype = {}
    for dtype in ("float16", "bfloat16"):
        arms = {item["arm"]: item for item in records if item["dtype"] == dtype}
        input_sets = {tuple(item.get("input_sha256", ())) for item in arms.values()}
        eager_sets = {item.get("eager_sha256") for item in arms.values()}
        by_dtype[dtype] = {
            "same_inputs_across_fresh_processes": len(input_sets) == 1,
            "same_addcdiv_eager_across_fresh_processes": len(eager_sets) == 1,
            "off_equals_decomposed": (
                arms["off"].get("compiled_sha256")
                == arms["decomposed"].get("compiled_sha256")
            ),
            "re_fused_equals_decomposed": (
                arms["re-fused"].get("compiled_sha256")
                == arms["decomposed"].get("compiled_sha256")
            ),
            "arms": arms,
        }
    valid = all(
        item.get("status") == "passed" and item.get("return_code") == 0
        for item in records
    ) and all(
        item["same_inputs_across_fresh_processes"]
        and item["same_addcdiv_eager_across_fresh_processes"]
        for item in by_dtype.values()
    )
    summary = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": "T-078",
        "backend": "triton_experimental",
        "device": args.device,
        "fresh_process_count": 6,
        "performance_measured": False,
        "status": "valid-three-arm-observation" if valid else "invalid-run",
        "dtypes": by_dtype,
    }
    (run / "three_arm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"three_arm_status={summary['status']}")
    print(f"artifacts={run}")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
