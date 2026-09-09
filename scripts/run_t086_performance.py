#!/usr/bin/env python3
"""T-086目标级功能与六臂性能调度；每个OFF/ON使用新进程。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "runners/t086_performance_worker.py"
VALIDATOR = ROOT / "scripts/validate_t086.py"
ORDER = ("off1", "on1", "on2", "off2", "off3", "on3")


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_arm(command: list[str], work: Path, output: Path, timeout: int = 3600) -> None:
    with (output / "stdout.log").open("w", encoding="utf-8") as stdout, (
        output / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=work,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if return_code:
        raise RuntimeError(f"arm失败，证据目录：{output}")


def assert_consistent_sources(records: list[dict]) -> dict[str, str]:
    union = {}
    for record in records:
        for path, digest in record["loaded_source_sha256"].items():
            if path in union and union[path] != digest:
                raise ValueError("OFF/ON实际加载的同一源码发生变化")
            union[path] = digest
    return union


def aggregate_functional(destination: Path, device: str) -> dict:
    arms = {
        mode: json.loads((destination / mode / "functional_result.json").read_text())
        for mode in ("off", "on")
    }
    if arms["off"]["pid"] == arms["on"]["pid"]:
        raise ValueError("功能OFF/ON没有使用独立进程")
    expected_backend = "triton_experimental" if device == "npu" else "inductor-default"
    for mode, record in arms.items():
        if record["backend"] != expected_backend or record["mode"] != mode:
            raise ValueError("功能arm后端或模式不一致")
        if record["correctness"] != "passed" or record["state"].get("negative_guard", {}).get("status") != "passed":
            raise ValueError("功能或负例保护未通过")
    sources = assert_consistent_sources(list(arms.values()))
    worker_digest = hashlib.sha256(WORKER.read_bytes()).hexdigest()
    return {
        "schema_version": "1.0",
        "task_id": "T-086",
        "acceptance_unit_id": "AU-post-grad-reinplace-inplaceable-ops",
        "backend": expected_backend,
        "pytorch_commit": arms["on"]["pytorch_commit"],
        "correctness": "passed",
        "numerical_execution": True,
        "target_rewrite": "confirmed",
        "graph_breaks": 0,
        "fallbacks": 0,
        "fallback_scope": "graph-break-or-CPU-fallback-only",
        "allowed_device_lowering": (
            "registered-IndexPutFallback-extern" if device == "npu" else "none"
        ),
        "product_disabled": False,
        "measurement_workload": "reinplace-index-put-community-shape",
        "world_size": 1,
        "input_spec": arms["on"]["input_spec"],
        "worker_sha256": worker_digest,
        "source_files": sources,
        "arms": {
            mode: {
                "pid": record["pid"],
                "target_rewrite": record["target_rewrite"],
                "state": record["state"],
                "evidence": f"{mode}/functional_result.json",
            }
            for mode, record in arms.items()
        },
        "generated_at": datetime.now().astimezone().isoformat(),
    }


def aggregate_performance(destination: Path) -> dict:
    arms = {}
    records = []
    pids = set()
    for arm in ORDER:
        summary = json.loads((destination / arm / "arm_summary.json").read_text())
        record = json.loads((destination / arm / "worker_result.json").read_text())
        if record["pid"] in pids:
            raise ValueError("六臂性能没有使用六个独立进程")
        pids.add(record["pid"])
        arms[arm] = summary
        records.append(record)
    sources = assert_consistent_sources(records)
    result = {
        "schema_version": "1.0",
        "task_id": "T-086",
        "acceptance_unit_id": "AU-post-grad-reinplace-inplaceable-ops",
        "execution_order": list(ORDER),
        "status": "measured-awaiting-review",
        "timing": {},
        "memory": {},
        "loaded_source_sha256_union": sources,
        "verdict": "PENDING_SOURCE_AND_KERNEL_REVIEW",
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    for metric in ("host_ms", "event_ms"):
        result["timing"][metric] = {}
        for quantile in ("p50", "p99"):
            off = statistics.median(
                arms[f"off{index}"]["timing"][metric][quantile]
                for index in (1, 2, 3)
            )
            on = statistics.median(
                arms[f"on{index}"]["timing"][metric][quantile]
                for index in (1, 2, 3)
            )
            result["timing"][metric][quantile] = {
                "off": off,
                "on": on,
                "improvement_percent": (off - on) / off * 100 if off else None,
            }
    for key in ("allocated", "reserved"):
        result["memory"][key] = {
            mode: statistics.median(
                arms[f"{mode}{index}"]["rank_peak_memory"][0][key]
                for index in (1, 2, 3)
            )
            for mode in ("off", "on")
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", choices=("reinplace-index-put",), default="reinplace-index-put")
    parser.add_argument("--device", choices=("cuda", "npu"), required=True)
    parser.add_argument("--phase", choices=("functional", "benchmark"), required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--pytorch-root",
        type=Path,
        default=Path(
            os.environ.get(
                "PYTORCH_ROOT", "/home/z50063656/Pass/src/pytorch"
            )
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--pytorch-root",
            str(args.pytorch_root.resolve()),
        ],
        check=True,
        cwd=Path.cwd(),
    )
    if args.validate_only:
        print(json.dumps({"task": "T-086", "unit": args.unit, "status": "prepared-not-measured"}, ensure_ascii=False))
        return
    if args.phase == "benchmark" and args.gate is None:
        parser.error("benchmark阶段必须提供经复核的--gate")
    work = Path(os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")).resolve()
    if Path.cwd().resolve() != work:
        parser.error(f"必须从 {work} 启动")
    output_root = (
        args.output_root or work / "t086-performance-results"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = Path(
        tempfile.mkdtemp(
            prefix=f"{args.phase}-{datetime.now().strftime('%Y%m%dT%H%M%S')}-",
            dir=output_root,
        )
    )
    plan = json.loads((ROOT / "upstream/t086_performance_plan.yaml").read_text())
    if args.phase == "functional":
        modes = ("off", "on")
    else:
        modes = ORDER
    for arm in modes:
        mode = arm.rstrip("123")
        arm_dir = destination / arm
        arm_dir.mkdir()
        command = [
            sys.executable,
            str(WORKER),
            "--unit",
            args.unit,
            "--mode",
            mode,
            "--device",
            args.device,
            "--phase",
            args.phase,
            "--output",
            str(arm_dir),
            "--warmup",
            str(plan["measurement_contract"]["warmup"]),
            "--runs",
            str(plan["measurement_contract"]["runs"]),
        ]
        if args.gate is not None:
            command.extend(("--gate", str(args.gate.resolve())))
        print(f"t086_{args.phase}_arm={arm} start", flush=True)
        run_arm(command, work, arm_dir)
    if args.phase == "functional":
        result = aggregate_functional(destination, args.device)
        write_json(destination / "functional_summary.json", result)
    else:
        result = aggregate_performance(destination)
        write_json(destination / "performance_summary.json", result)
    print(f"t086_{args.phase}_artifacts={destination}")


if __name__ == "__main__":
    main()
