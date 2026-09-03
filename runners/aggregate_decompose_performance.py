#!/usr/bin/env python3
"""聚合 T-077 decompose 单元三轮 fresh-process OFF/ON 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time


def improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result_paths = sorted((args.run_dir / "workers").glob("*/result.json"))
    workers = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    if len(workers) != 6:
        raise RuntimeError(f"应有 6 个 worker，实际为 {len(workers)}")
    units = {item["acceptance_unit_id"] for item in workers}
    if len(units) != 1:
        raise RuntimeError(f"worker acceptance unit 不一致: {sorted(units)}")
    sources = {
        json.dumps(item["performance_case_source"], sort_keys=True)
        for item in workers
    }
    if len(sources) != 1:
        raise RuntimeError("六个 worker 的性能测例来源不一致")

    grouped = {
        mode: [item for item in workers if item["mode"] == mode]
        for mode in ("off", "on")
    }
    if any(len(items) != 3 for items in grouped.values()):
        raise RuntimeError("OFF/ON 必须各有三个 fresh-process worker")

    aggregate = {}
    for mode, items in grouped.items():
        aggregate[mode] = {
            "rounds": [item["round"] for item in items],
            "host_p50_ms": [item["timing"]["host"]["p50_ms"] for item in items],
            "host_p99_ms": [item["timing"]["host"]["p99_ms"] for item in items],
            "device_p50_ms": [
                item["timing"]["device_event"]["p50_ms"] for item in items
            ],
            "device_p99_ms": [
                item["timing"]["device_event"]["p99_ms"] for item in items
            ],
            "compile_ms": [
                item["timing"]["compile_and_first_run_ms"] for item in items
            ],
            "max_allocated_bytes": [
                item["memory"]["max_allocated_bytes"] for item in items
            ],
            "max_reserved_bytes": [
                item["memory"]["max_reserved_bytes"] for item in items
            ],
        }
        aggregate[mode]["median"] = {
            key: statistics.median(value)
            for key, value in aggregate[mode].items()
            if isinstance(value, list) and key != "rounds"
        }

    off = aggregate["off"]["median"]
    on = aggregate["on"]["median"]
    comparison = {
        "host_p50_improvement_percent": improvement(
            off["host_p50_ms"], on["host_p50_ms"]
        ),
        "host_p99_improvement_percent": improvement(
            off["host_p99_ms"], on["host_p99_ms"]
        ),
        "device_p50_improvement_percent": improvement(
            off["device_p50_ms"], on["device_p50_ms"]
        ),
        "device_p99_improvement_percent": improvement(
            off["device_p99_ms"], on["device_p99_ms"]
        ),
        "allocated_delta_bytes": (
            on["max_allocated_bytes"] - off["max_allocated_bytes"]
        ),
        "reserved_delta_bytes": (
            on["max_reserved_bytes"] - off["max_reserved_bytes"]
        ),
    }
    result = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_id": "T-077",
        "acceptance_unit_id": workers[0]["acceptance_unit_id"],
        "performance_case_source": workers[0]["performance_case_source"],
        "status": "measured",
        "method": {
            "process_isolation": "fresh-process-per-mode-and-round",
            "order": "OFF1-ON1-ON2-OFF2-OFF3-ON3",
            "warmup": workers[0]["warmup"],
            "runs": workers[0]["runs"],
            "backend": "triton_experimental",
            "correctness_gate": "六个 worker 均通过 community 正例的 1e-3 atol/rtol",
            "candidate_scope": {
                mode: items[0]["candidate_scope"]
                for mode, items in grouped.items()
            },
        },
        "aggregate": aggregate,
        "comparison": comparison,
        "workers": [str(path.relative_to(args.run_dir)) for path in result_paths],
    }
    output = args.run_dir / "performance_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"performance_summary={output}")
    print(json.dumps(comparison, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
