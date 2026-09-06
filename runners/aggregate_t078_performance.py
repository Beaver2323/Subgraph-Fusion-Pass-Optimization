#!/usr/bin/env python3
"""聚合 T-078 三轮目标级 fresh-process OFF/ON 性能证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any


ORDER = ["off1", "on1", "on2", "off2", "off3", "on3"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline * 100.0


def classify(device_p50: float, device_p99: float) -> str:
    if device_p50 >= 10.0 and device_p99 >= 0.0:
        return "PERF_IMPROVED"
    if device_p50 <= -5.0 or device_p99 <= -5.0:
        return "PERF_REGRESSED"
    return "PERF_NEUTRAL"


def aggregate_unit(unit_dir: Path) -> Path:
    result_paths = [unit_dir / "workers" / key / "result.json" for key in ORDER]
    missing = [str(path) for path in result_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"缺少 worker 结果：{missing}")
    workers = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    units = {item["acceptance_unit_id"] for item in workers}
    sources = {
        json.dumps(item["performance_case_source"], sort_keys=True)
        for item in workers
    }
    if len(units) != 1 or len(sources) != 1:
        raise RuntimeError("六个 worker 的 acceptance unit 或测例来源不一致")
    if any(item["environment"]["backend"] != "triton_experimental" for item in workers):
        raise RuntimeError("存在非 triton_experimental worker")
    source_states = {
        json.dumps(item["source_state"], sort_keys=True) for item in workers
    }
    if len(source_states) != 1:
        raise RuntimeError("六个 worker 的源码状态不一致")

    grouped = {
        mode: [item for item in workers if item["mode"] == mode]
        for mode in ("off", "on")
    }
    if any(len(items) != 3 for items in grouped.values()):
        raise RuntimeError("OFF/ON 必须各有三个 fresh-process worker")
    workload_ids = [item["workload_id"] for item in workers[0]["workloads"]]
    if any(
        [item["workload_id"] for item in worker["workloads"]] != workload_ids
        for worker in workers
    ):
        raise RuntimeError("六个 worker 的 workload 集合或顺序不一致")

    workload_results = []
    for workload_id in workload_ids:
        by_mode: dict[str, Any] = {}
        for mode, mode_workers in grouped.items():
            entries = [
                next(
                    workload
                    for workload in worker["workloads"]
                    if workload["workload_id"] == workload_id
                )
                for worker in mode_workers
            ]
            raw = {
                "host_p50_ms": [entry["timing"]["host"]["p50_ms"] for entry in entries],
                "host_p99_ms": [entry["timing"]["host"]["p99_ms"] for entry in entries],
                "device_p50_ms": [
                    entry["timing"]["device_event"]["p50_ms"] for entry in entries
                ],
                "device_p99_ms": [
                    entry["timing"]["device_event"]["p99_ms"] for entry in entries
                ],
                "compile_ms": [
                    entry["timing"]["compile_and_first_run_ms"] for entry in entries
                ],
                "max_allocated_bytes": [
                    entry["memory"]["max_allocated_bytes"] for entry in entries
                ],
                "max_reserved_bytes": [
                    entry["memory"]["max_reserved_bytes"] for entry in entries
                ],
                "wrapper_dispatch_count": [
                    entry["code_observation"]["wrapper_dispatch_count"]
                    for entry in entries
                ],
            }
            by_mode[mode] = {
                "rounds": [entry["round"] for entry in mode_workers],
                "raw": raw,
                "median": {
                    key: statistics.median(values) for key, values in raw.items()
                },
                "target_hits_per_workload": [
                    entry["target_hits"] for entry in entries
                ],
                "target_hits_total_per_worker": [
                    entry["target_control"]["hits"] for entry in mode_workers
                ],
            }
        off = by_mode["off"]["median"]
        on = by_mode["on"]["median"]
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
            "wrapper_dispatch_delta": (
                on["wrapper_dispatch_count"] - off["wrapper_dispatch_count"]
            ),
        }
        comparison["verdict"] = classify(
            comparison["device_p50_improvement_percent"],
            comparison["device_p99_improvement_percent"],
        )
        first = next(
            workload
            for workload in workers[0]["workloads"]
            if workload["workload_id"] == workload_id
        )
        workload_results.append(
            {
                "workload_id": workload_id,
                "shape_contract": first["shape_contract"],
                "off_on": by_mode,
                "comparison": comparison,
            }
        )

    verdicts = {item["comparison"]["verdict"] for item in workload_results}
    if verdicts == {"PERF_IMPROVED"}:
        verdict = "PERF_IMPROVED"
    elif "PERF_REGRESSED" in verdicts:
        verdict = "PERF_REGRESSED"
    elif verdicts == {"PERF_NEUTRAL"}:
        verdict = "PERF_NEUTRAL"
    else:
        verdict = "PERF_MIXED"
    result = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_id": "T-078",
        "acceptance_unit_id": workers[0]["acceptance_unit_id"],
        "unit": workers[0]["unit"],
        "status": "measured",
        "verdict": verdict,
        "performance_case_source": workers[0]["performance_case_source"],
        "method": {
            "backend": "triton_experimental",
            "process_isolation": "fresh-process-per-arm-and-round",
            "order": "OFF1-ON1-ON2-OFF2-OFF3-ON3",
            "warmup": workers[0]["warmup"],
            "runs": workers[0]["runs"],
            "target_control": workers[0]["target_control"]["only_difference"],
            "verdict_threshold": (
                "IMPROVED: Event p50>=10% 且 p99不回退；"
                "REGRESSED: Event p50或p99回退>=5%；其余 NEUTRAL"
            ),
        },
        "workloads": workload_results,
        "source_state": workers[0]["source_state"],
        "tooling": {
            "worker_sha256": sha256(Path(__file__).with_name("t078_performance_worker.py")),
            "aggregator_sha256": sha256(Path(__file__)),
        },
        "workers": [str(path.relative_to(unit_dir)) for path in result_paths],
    }
    output = unit_dir / "performance_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"performance_summary={output}")
    return output


def aggregate_task(task_run_dir: Path) -> Path:
    summary_paths = sorted(task_run_dir.glob("*/performance_summary.json"))
    units = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    if len(units) != 4:
        raise RuntimeError(f"T-078 应有四个性能单元，实际为 {len(units)}")
    result = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_id": "T-078",
        "status": "performance-measurement-complete",
        "backend": "triton_experimental",
        "completion": {
            "acceptance_units": 4,
            "measured": 4,
            "explicitly_disabled_subvariants_exempt": 1,
        },
        "acceptance_units": [
            {
                "acceptance_unit_id": unit["acceptance_unit_id"],
                "unit": unit["unit"],
                "verdict": unit["verdict"],
                "evidence": str(path.relative_to(task_run_dir)),
                "evidence_sha256": sha256(path),
            }
            for path, unit in zip(summary_paths, units)
        ],
        "explicit_exemption": {
            "variant_id": "half-dtype-gate-disabled-positive",
            "reason": (
                "NPU 默认 keep_addmm_fused_for_half_dtypes=true；"
                "不切为 false 人为制造性能 ON 路径"
            ),
        },
    }
    output = task_run_dir / "performance_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"task_performance_summary={output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--unit-dir", type=Path)
    group.add_argument("--task-run-dir", type=Path)
    args = parser.parse_args()
    if args.unit_dir is not None:
        aggregate_unit(args.unit_dir)
    else:
        aggregate_task(args.task_run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
