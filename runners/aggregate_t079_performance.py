#!/usr/bin/env python3
"""聚合 T-079 三轮 OFF/ON 性能结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    workers = []
    for path in sorted((args.run_dir / "workers").glob("*/result.json")):
        workers.append(json.loads(path.read_text(encoding="utf-8")))
    if len(workers) != 6:
        raise RuntimeError(f"预期 6 个 worker，实际 {len(workers)}")
    if any(item["backend"] != "triton_experimental" for item in workers):
        raise RuntimeError("存在非 triton_experimental worker")
    units = {item["unit"] for item in workers}
    if len(units) != 1:
        raise RuntimeError(f"worker unit 不一致: {units}")

    workload_ids = [item["workload_id"] for item in workers[0]["workloads"]]
    comparisons = []
    for workload_id in workload_ids:
        row = {"workload_id": workload_id}
        for metric in ("event", "host"):
            for percentile in ("p50_ms", "p99_ms"):
                arms = {}
                for mode in ("off", "on"):
                    values = [
                        next(
                            work for work in worker["workloads"]
                            if work["workload_id"] == workload_id
                        )[metric][percentile]
                        for worker in workers if worker["mode"] == mode
                    ]
                    arms[mode] = statistics.median(values)
                row[f"{metric}_{percentile}"] = {
                    "off_median_ms": arms["off"],
                    "on_median_ms": arms["on"],
                    "improvement_percent": (
                        (arms["off"] - arms["on"]) / arms["off"] * 100
                    ),
                }
        comparisons.append(row)
    primary = next(item for item in comparisons if item["workload_id"] == "community-shape")
    event_gain = primary["event_p50_ms"]["improvement_percent"]
    event_p99_gain = primary["event_p99_ms"]["improvement_percent"]
    verdict = (
        "PERF_IMPROVED" if event_gain >= 10 and event_p99_gain >= 0
        else "PERF_REGRESSED" if event_gain <= -10 or event_p99_gain <= -10
        else "PERF_NEUTRAL"
    )
    summary = {
        "schema_version": "1.0",
        "task_id": "T-079",
        "unit": next(iter(units)),
        "acceptance_unit_id": workers[0]["acceptance_unit_id"],
        "backend": "triton_experimental",
        "source_revision_aligned": len({item["torch_commit"] for item in workers}) == 1,
        "worker_count": len(workers),
        "order": [f"{item['mode']}{item['round']}" for item in workers],
        "comparisons": comparisons,
        "verdict": verdict,
        "primary_event_p50_improvement_percent": event_gain,
        "primary_event_p99_improvement_percent": event_p99_gain,
    }
    (args.run_dir / "performance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("T079_PERFORMANCE_SUMMARY=" + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
