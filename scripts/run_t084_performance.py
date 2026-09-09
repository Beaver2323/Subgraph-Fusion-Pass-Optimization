#!/usr/bin/env python3
"""T-084 六臂性能调度；validate-only 不导入 torch 或访问设备。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ORDER = ("off1", "on1", "on2", "off2", "off3", "on3")


def run_arm(command, work, stdout, stderr, timeout=3600):
    process = subprocess.Popen(
        command,
        cwd=work,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise


def aggregate(destination: Path) -> dict:
    arms = {}
    pids = set()
    source_digests = {}
    rank_fingerprints = {}
    for arm in ORDER:
        summary = json.loads((destination / arm / "arm_summary.json").read_text())
        for rank in range(summary["world_size"]):
            worker = json.loads(
                (destination / arm / f"rank-{rank}/worker_result.json").read_text()
            )
            if worker["pid"] in pids:
                raise ValueError("OFF/ON未使用独立新进程")
            pids.add(worker["pid"])
            fingerprint = {
                key: worker[key]
                for key in ("backend", "pytorch_commit", "input_contract", "gate_sha256")
            }
            if rank in rank_fingerprints and rank_fingerprints[rank] != fingerprint:
                raise ValueError("OFF/ON后端、revision、输入或gate不一致")
            rank_fingerprints[rank] = fingerprint
            for path, digest in worker["loaded_source_sha256"].items():
                if path in source_digests and source_digests[path] != digest:
                    raise ValueError("OFF/ON实际加载源码发生变化")
                source_digests[path] = digest
        arms[arm] = summary

    result = {
        "execution_order": list(ORDER),
        "status": "measured-awaiting-human-verdict",
        "timing": {},
        "memory": {},
        "loaded_source_sha256_union": source_digests,
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
                max(item[key] for item in arms[f"{mode}{index}"]["rank_peak_memory"])
                for index in (1, 2, 3)
            )
            for mode in ("off", "on")
        }
    result["verdict"] = "PENDING_SOURCE_AND_KERNEL_REVIEW"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", choices=("dedup-reduce-scatter",), default="dedup-reduce-scatter")
    parser.add_argument("--device", choices=("cuda", "npu"), default="npu")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_prepared_tasks.py"),
            "--task",
            "T-084",
        ],
        check=True,
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "task": "T-084",
                    "units": [args.unit],
                    "status": "prepared-not-measured",
                    "minimum_performance_devices": 2,
                },
                ensure_ascii=False,
            )
        )
        return
    if args.gate is None:
        parser.error("实测必须提供已人工复核的--gate")

    work = Path(
        os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")
    ).resolve()
    if Path.cwd().resolve() != work:
        parser.error(f"必须从 {work} 启动")
    output_root = (
        args.output_root or work / "t084-performance-results"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = Path(
        tempfile.mkdtemp(
            prefix="dedup-reduce-scatter-" + datetime.now().strftime("%Y%m%dT%H%M%S") + "-",
            dir=output_root,
        )
    )
    plan = json.loads((ROOT / "upstream/t084_performance_plan.yaml").read_text())
    for arm in ORDER:
        arm_dir = destination / arm
        arm_dir.mkdir()
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "runners/t084_performance_worker.py"),
            "--unit",
            args.unit,
            "--mode",
            arm[:-1],
            "--device",
            args.device,
            "--gate",
            str(args.gate.resolve()),
            "--output",
            str(arm_dir),
            "--warmup",
            str(plan["measurement_contract"]["warmup"]),
            "--runs",
            str(plan["measurement_contract"]["runs"]),
        ]
        print(f"performance_arm={arm} start", flush=True)
        with (arm_dir / "stdout.log").open("w") as stdout, (
            arm_dir / "stderr.log"
        ).open("w") as stderr:
            try:
                return_code = run_arm(command, work, stdout, stderr)
            except subprocess.TimeoutExpired:
                raise SystemExit(f"性能超时；证据={arm_dir}")
        if return_code:
            raise SystemExit(f"性能门禁或运行失败；证据={arm_dir}")
    result = aggregate(destination)
    result.update(
        task_id="T-084",
        unit=args.unit,
        generated_at=datetime.now().astimezone().isoformat(),
    )
    (destination / "performance_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"performance_artifacts={destination}")


if __name__ == "__main__":
    main()
