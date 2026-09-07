#!/usr/bin/env python3
"""聚合 T-080 三轮 OFF/ON 性能结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


EXECUTION_ORDER = ["off1", "on1", "on2", "off2", "off3", "on3"]


def improvement(off: float, on: float) -> float:
    return (off - on) / off * 100 if off else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    workers = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.run_dir / "workers").glob("*/result.json"))
    ]
    if len(workers) != 6:
        raise RuntimeError(f"预期 6 个 worker，实际 {len(workers)}")
    if any(item["backend"] != "triton_experimental" for item in workers):
        raise RuntimeError("存在非 triton_experimental worker")
    worker_ids = {f"{item['mode']}{item['round']}" for item in workers}
    if worker_ids != set(EXECUTION_ORDER):
        raise RuntimeError(f"OFF/ON worker 集合不完整：{sorted(worker_ids)}")
    workload_ids = [item["workload_id"] for item in workers[0]["workloads"]]
    comparisons = []
    for workload_id in workload_ids:
        row = {"workload_id": workload_id}
        for metric in ("event", "host"):
            for percentile in ("p50_ms", "p99_ms"):
                medians = {}
                for mode in ("off", "on"):
                    medians[mode] = statistics.median([
                        next(
                            work for work in worker["workloads"]
                            if work["workload_id"] == workload_id
                        )[metric][percentile]
                        for worker in workers if worker["mode"] == mode
                    ])
                row[f"{metric}_{percentile}"] = {
                    "off_median_ms": medians["off"],
                    "on_median_ms": medians["on"],
                    "improvement_percent": improvement(
                        medians["off"], medians["on"]
                    ),
                }
        for metric in ("max_memory_allocated", "max_memory_reserved"):
            medians = {
                mode: statistics.median(
                    next(
                        work for work in worker["workloads"]
                        if work["workload_id"] == workload_id
                    )[metric]
                    for worker in workers if worker["mode"] == mode
                )
                for mode in ("off", "on")
            }
            row[metric] = {
                "off_median_bytes": medians["off"],
                "on_median_bytes": medians["on"],
                "improvement_percent": improvement(
                    medians["off"], medians["on"]
                ),
            }
        comparisons.append(row)
    primary = next(
        row for row in comparisons
        if next(
            work for work in workers[0]["workloads"]
            if work["workload_id"] == row["workload_id"]
        )["community_primary"]
    )
    p50 = primary["event_p50_ms"]["improvement_percent"]
    p99 = primary["event_p99_ms"]["improvement_percent"]
    verdict = (
        "PERF_IMPROVED" if p50 >= 10 and p99 >= 0
        else "PERF_REGRESSED" if p50 <= -10 or p99 <= -10
        else "PERF_NEUTRAL"
    )
    summary = {
        "schema_version": "1.0",
        "task_id": "T-080",
        "unit": workers[0]["unit"],
        "acceptance_unit_id": workers[0]["acceptance_unit_id"],
        "backend": "triton_experimental",
        "worker_count": 6,
        "execution_order": EXECUTION_ORDER,
        "aggregated_worker_order": sorted(worker_ids),
        "comparisons": comparisons,
        "verdict": verdict,
        "primary_event_p50_improvement_percent": p50,
        "primary_event_p99_improvement_percent": p99,
    }
    (args.run_dir / "performance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("T080_PERFORMANCE_SUMMARY=" + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
