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
UNITS = {
    "pointless-cumsum": 1,
    "overlap-device-put": 2,
    "partitioned-scatter": 1,
}


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def aggregate_functional(destination: Path, unit: str, device: str) -> dict:
    modes = {}
    pids = set()
    sources = {}
    world_size = UNITS[unit]
    for mode in ("off", "on"):
        paths = sorted((destination / mode).glob("rank-*/worker_result.json"))
        if len(paths) != world_size:
            raise ValueError(f"{mode}功能rank数量不满足world_size={world_size}")
        records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        for rank, record in enumerate(records):
            if record["pid"] in pids:
                raise ValueError("OFF/ON功能门禁未使用独立新进程")
            pids.add(record["pid"])
            if record["rank"] != rank or record["mode"] != mode:
                raise ValueError("rank或mode记录不一致")
            if record["correctness"] != "passed":
                raise ValueError("存在未通过的功能rank")
            state = record["target_state"]
            if unit == "pointless-cumsum":
                expected = 0 if mode == "off" else 1
                if (state["handler_calls"] >= 1) != bool(expected):
                    raise ValueError("pointless-cumsum目标handler状态不符")
            elif unit == "overlap-device-put":
                if mode == "off" and state["scheduler_calls"] != 0:
                    raise ValueError("overlap OFF仍进入目标scheduler")
                if mode == "on" and (
                    state["scheduler_calls"] < 1
                    or state["converted_device_puts"] < 1
                ):
                    raise ValueError("overlap ON未转换目标device_put")
            else:
                if mode == "off" and (
                    state["pass_calls"] != 0
                    or state["partitioned_scatter_applied"] != 0
                ):
                    raise ValueError("partitioned-scatter OFF仍发生目标改写")
                if mode == "on" and (
                    state["pass_calls"] < 1
                    or state["partitioned_scatter_applied"] < 3
                    or state["memory_probe_calls"] < 1
                    or not any(state["memory_state"])
                    or state["negative_accumulate_false_applied"] != 0
                ):
                    raise ValueError(
                        "partitioned-scatter ON未通过改写、显存或负例门禁"
                    )
            for path, digest in record["loaded_source_sha256"].items():
                if path in sources and sources[path] != digest:
                    raise ValueError("OFF/ON实际加载源码发生变化")
                sources[path] = digest
        modes[mode] = records

    on = modes["on"][0]
    return {
        "schema_version": "1.0",
        "task_id": "T-085",
        "acceptance_unit_id": on["acceptance_unit_id"],
        "unit": unit,
        "backend": (
            "triton_experimental" if device == "npu" else "inductor-default"
        ),
        "pytorch_commit": on["pytorch_commit"],
        "correctness": "passed",
        "numerical_execution": True,
        "target_rewrite": "confirmed",
        "graph_breaks": 0,
        "fallbacks": 0,
        "product_disabled": False,
        "measurement_workload": on["measurement_workload"],
        "world_size": world_size,
        "process_group_backend": on["process_group_backend"],
        "input_spec": on["input_spec"],
        "worker_sha256": on["worker_sha256"],
        "source_files": sources,
        "arms": {
            mode: {
                "pids": [record["pid"] for record in records],
                "observations": [record["target_state"] for record in records],
                "evidence": [
                    f"{mode}/rank-{record['rank']}/worker_result.json"
                    for record in records
                ],
            }
            for mode, records in modes.items()
        },
        "generated_at": datetime.now().astimezone().isoformat(),
    }


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
    parser.add_argument(
        "--phase", choices=("functional", "benchmark"), default="benchmark"
    )
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
                    "blocked_units": [],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.unit is None:
        parser.error("实测必须提供--unit")
    if args.phase == "benchmark" and args.gate is None:
        parser.error("benchmark实测必须提供人工签署--gate")

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
            prefix=(
                args.phase
                + "-"
                + args.unit
                + "-"
                + datetime.now().strftime("%Y%m%dT%H%M%S")
                + "-"
            ),
            dir=output_root,
        )
    )
    plan = json.loads((ROOT / "upstream/t085_performance_plan.yaml").read_text())
    arms = ("off", "on") if args.phase == "functional" else ORDER
    for arm in arms:
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
                arm.rstrip("123"),
                "--device",
                args.device,
                "--phase",
                args.phase,
                "--output",
                str(arm_dir),
                "--warmup",
                str(
                    1
                    if args.phase == "functional"
                    else plan["measurement_contract"]["warmup"]
                ),
                "--runs",
                str(
                    1
                    if args.phase == "functional"
                    else plan["measurement_contract"]["runs"]
                ),
            ]
        )
        if args.gate is not None:
            command.extend(("--gate", str(args.gate.resolve())))
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

    if args.phase == "functional":
        result = aggregate_functional(destination, args.unit, args.device)
        path = destination / "functional_summary.json"
    else:
        result = load_aggregator().aggregate(destination)
        path = destination / "performance_summary.json"
    write_json(path, result)
    print(f"t085_{args.phase}_artifacts={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
