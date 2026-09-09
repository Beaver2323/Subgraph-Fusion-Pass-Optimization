#!/usr/bin/env python3
"""聚合T-085六臂结果；只读JSON，不导入torch。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from datetime import datetime


ORDER = ("off1", "on1", "on2", "off2", "off3", "on3")


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (pos - lower)


def aggregate(root: Path) -> dict:
    arms = {}
    pids = set()
    frozen = None
    for arm in ORDER:
        rank_files = sorted((root / arm).glob("rank-*/worker_result.json"))
        if not rank_files:
            raise ValueError(f"缺少{arm} rank结果")
        ranks = [json.loads(path.read_text(encoding="utf-8")) for path in rank_files]
        if len(ranks) != ranks[0]["world_size"]:
            raise ValueError(f"{arm} rank数量不等于world_size")
        fingerprint = {
            key: ranks[0][key]
            for key in (
                "task_id",
                "acceptance_unit_id",
                "unit",
                "backend",
                "device",
                "world_size",
                "pytorch_commit",
                "measurement_workload",
                "input_spec",
                "worker_sha256",
                "gate_sha256",
            )
        }
        if frozen is None:
            frozen = fingerprint
        elif fingerprint != frozen:
            raise ValueError("六臂后端、源码、gate或workload不一致")
        for rank in ranks:
            if rank["pid"] in pids:
                raise ValueError("OFF/ON没有使用全新进程")
            pids.add(rank["pid"])
            if rank["correctness"] != "passed":
                raise ValueError("存在未通过正确性的rank")
            if rank["mode"] != arm[:-1]:
                raise ValueError("arm目录与mode不符")
        arm_result = {"compile_ms": max(rank["compile_ms"] for rank in ranks)}
        arm_result["timing"] = {}
        for metric in ("host_ms", "event_ms"):
            arrays = [rank["timing"][metric]["samples"] for rank in ranks]
            if len({len(values) for values in arrays}) != 1:
                raise ValueError("各rank样本数不一致")
            maxima = [max(values) for values in zip(*arrays)]
            arm_result["timing"][metric] = {
                "p50": statistics.median(maxima),
                "p99": percentile(maxima, 0.99),
                "cross_rank_max_samples": maxima,
            }
        arm_result["memory"] = {
            "peak_allocated": max(rank["memory"]["peak_allocated"] for rank in ranks),
            "peak_reserved": max(rank["memory"]["peak_reserved"] for rank in ranks),
        }
        arms[arm] = arm_result

    result = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        **frozen,
        "execution_order": list(ORDER),
        "status": "measured-awaiting-human-source-and-kernel-review",
        "arms": arms,
        "comparison": {},
        "verdict": "PENDING_HUMAN_REVIEW",
    }
    for metric in ("host_ms", "event_ms"):
        result["comparison"][metric] = {}
        for quantile in ("p50", "p99"):
            off = statistics.median(
                arms[f"off{i}"]["timing"][metric][quantile] for i in (1, 2, 3)
            )
            on = statistics.median(
                arms[f"on{i}"]["timing"][metric][quantile] for i in (1, 2, 3)
            )
            result["comparison"][metric][quantile] = {
                "off": off,
                "on": on,
                "improvement_percent": (off - on) / off * 100 if off else None,
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.root.resolve())
    path = args.root.resolve() / "performance_summary.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"performance_summary={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
