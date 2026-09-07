#!/usr/bin/env python3
"""T-081～T-083 性能调度；--validate-only仅校验准备，不运行GPU/NPU。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
ORDER = ("off1", "on1", "on2", "off2", "off3", "on3")


def run_arm(command, work, out, err, timeout=3600):
    process = subprocess.Popen(command, cwd=work, stdout=out, stderr=err, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        # 仅终止本调用创建的新session/process-group；不扫描设备用户进程。
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise


def load_worker():
    spec = importlib.util.spec_from_file_location("prepared_worker", ROOT / "runners/t081_t083_performance_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aggregate(destination: Path) -> dict:
    arms = {}
    pids = set()
    source_by_rank = {}
    loaded_sources = {}
    for arm in ORDER:
        data = json.loads((destination / arm / "arm_summary.json").read_text())
        for rank in range(data["world_size"]):
            worker = json.loads((destination / arm / f"rank-{rank}/worker_result.json").read_text())
            if worker["pid"] in pids:
                raise ValueError("OFF/ON未使用不同进程")
            pids.add(worker["pid"])
            fingerprint = {key: worker[key] for key in
                           ("backend", "pytorch_commit", "input_contract", "gate_sha256")}
            if rank in source_by_rank and fingerprint != source_by_rank[rank]:
                raise ValueError("OFF/ON源码/输入/门禁/后端不一致，拒绝聚合")
            source_by_rank[rank] = fingerprint
            # OFF/ON允许加载不同模块，但同一个实际源码文件不能在两臂之间改变。
            for path, digest in worker["loaded_source_sha256"].items():
                if path in loaded_sources and loaded_sources[path] != digest:
                    raise ValueError("OFF/ON实际加载源码发生变化，拒绝聚合")
                loaded_sources[path] = digest
        arms[arm] = data
    result = {"execution_order": list(ORDER), "status": "measured", "timing": {}, "memory": {},
              "loaded_source_sha256_union": loaded_sources}
    for metric in ("host_ms", "event_ms"):
        result["timing"][metric] = {}
        for percentile in ("p50", "p99"):
            off = statistics.median(arms[f"off{i}"]["timing"][metric][percentile] for i in (1, 2, 3))
            on = statistics.median(arms[f"on{i}"]["timing"][metric][percentile] for i in (1, 2, 3))
            result["timing"][metric][percentile] = {
                "off": off, "on": on, "improvement_percent": (off - on) / off * 100 if off else None,
            }
    for key in ("allocated", "reserved"):
        result["memory"][key] = {
            mode: statistics.median(max(r[key] for r in arms[f"{mode}{i}"]["rank_peak_memory"]) for i in (1, 2, 3))
            for mode in ("off", "on")
        }
    # 视图/常量恒等图event可能接近0，保留原始测量但禁止自动报设备加速。
    result["verdict"] = "PENDING_SOURCE_AND_KERNEL_REVIEW"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("T-081", "T-082", "T-083"), required=True)
    parser.add_argument("--unit")
    parser.add_argument("--device", choices=("cuda", "npu"), default="npu")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts/validate_prepared_tasks.py"), "--task", args.task], check=True)
    worker = load_worker()
    targets = [key for key, value in worker.TARGETS.items() if value[0] == args.task]
    if args.unit is not None and args.unit not in targets:
        parser.error("unit不属于所选task")
    if args.validate_only:
        print(json.dumps({"task": args.task, "units": targets, "status": "prepared-not-measured",
                          "minimum_performance_devices": 2 if args.task == "T-083" else 1}, ensure_ascii=False))
        return
    if not args.unit or not args.gate:
        parser.error("实测必须同时指定 --unit 和 --gate 已复核功能证据")
    worker.read_gate(args.gate.resolve(), args.unit, args.device)
    work = Path(os.environ.get("PASS_TRACKER_WORK_DIR", "/home/z50063656/tmp")).resolve()
    if Path.cwd().resolve() != work:
        parser.error(f"必须从 {work} 启动")
    output_root = (args.output_root or work / f"{args.task.lower().replace('-', '')}-performance-results").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=args.unit + "-" + datetime.now().strftime("%Y%m%dT%H%M%S") + "-", dir=output_root))
    plan = json.loads((ROOT / f"upstream/{args.task.lower().replace('-', '')}_performance_plan.yaml").read_text())
    for arm in ORDER:
        arm_dir = destination / arm
        arm_dir.mkdir()
        prefix = [sys.executable]
        if args.task == "T-083":
            prefix += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]
        command = prefix + [str(ROOT / "runners/t081_t083_performance_worker.py"), "--unit", args.unit,
                            "--mode", arm[:-1], "--device", args.device, "--gate", str(args.gate.resolve()),
                            "--output", str(arm_dir), "--warmup", str(plan["measurement_contract"]["warmup"]),
                            "--runs", str(plan["measurement_contract"]["runs"])]
        print(f"performance_arm={arm} start", flush=True)
        with (arm_dir / "stdout.log").open("w") as out, (arm_dir / "stderr.log").open("w") as err:
            try:
                return_code = run_arm(command, work, out, err)
            except subprocess.TimeoutExpired:
                raise SystemExit(f"性能超时，停止聚合；证据={arm_dir}")
        if return_code:
            raise SystemExit(f"性能门禁或运行失败，停止聚合；证据={arm_dir}")
    result = aggregate(destination)
    result.update(task_id=args.task, unit=args.unit, generated_at=datetime.now().astimezone().isoformat())
    (destination / "performance_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"performance_artifacts={destination}")


if __name__ == "__main__":
    main()
