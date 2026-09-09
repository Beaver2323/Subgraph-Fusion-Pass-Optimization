#!/usr/bin/env python3
"""T-085六臂性能调度；validate-only不导入torch或访问设备。"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ORDER = ("off1", "on1", "on2", "off2", "off3", "on3")
UNITS = {"pointless-cumsum": 1, "overlap-device-put": 2}


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


def load_aggregator():
    path = ROOT / "scripts/aggregate_t085_performance.py"
    spec = importlib.util.spec_from_file_location("t085_aggregator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", choices=UNITS)
    parser.add_argument("--device", choices=("cuda", "npu"), default="npu")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_t085_preparation.py")],
        check=True,
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "task": "T-085",
                    "status": "prepared-not-measured",
                    "runnable_units": UNITS,
                    "blocked_units": ["partitioned-scatter"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.unit is None or args.gate is None:
        parser.error("实测必须提供--unit和人工签署--gate")

    work = Path(
        os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")
    ).resolve()
    if Path.cwd().resolve() != work:
        parser.error(f"必须从{work}启动")
    output_root = (
        args.output_root or work / "t085-performance-results"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = Path(
        tempfile.mkdtemp(
            prefix=args.unit + "-" + datetime.now().strftime("%Y%m%dT%H%M%S") + "-",
            dir=output_root,
        )
    )
    plan = json.loads((ROOT / "upstream/t085_performance_plan.yaml").read_text())
    for arm in ORDER:
        arm_dir = destination / arm
        arm_dir.mkdir()
        command = [sys.executable]
        if UNITS[args.unit] == 2:
            command.extend(
                ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]
            )
        command.extend(
            [
                str(ROOT / "runners/t085_performance_worker.py"),
                "--unit",
                args.unit,
                "--mode",
                arm[:-1],
                "--device",
                args.device,
                "--phase",
                "benchmark",
                "--gate",
                str(args.gate.resolve()),
                "--output",
                str(arm_dir),
                "--warmup",
                str(plan["measurement_contract"]["warmup"]),
                "--runs",
                str(plan["measurement_contract"]["runs"]),
            ]
        )
        print(f"performance_arm={arm} start", flush=True)
        with (arm_dir / "stdout.log").open("w") as stdout, (
            arm_dir / "stderr.log"
        ).open("w") as stderr:
            try:
                return_code = run_arm(command, work, stdout, stderr)
            except subprocess.TimeoutExpired:
                raise SystemExit(f"性能臂超时；证据={arm_dir}")
        if return_code:
            raise SystemExit(f"性能门禁或运行失败；证据={arm_dir}")

    result = load_aggregator().aggregate(destination)
    path = destination / "performance_summary.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"performance_artifacts={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
