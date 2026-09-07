#!/usr/bin/env python3
"""复核 T-081～T-083 NPU 功能原件，生成紧凑证据和性能门禁。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/home/z50063656/tmp")
TASK_UNITS = {
    "T-081": ("constant-fold", "convert"),
    "T-082": ("permute", "view"),
    "T-083": ("all-gather", "all-reduce", "reduce-scatter"),
}
EXPECTED_COLLECTIVES = {
    "all-gather": (3, 1),
    "all-reduce": (2, 1),
    "reduce-scatter": (2, 1),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_worker():
    path = ROOT / "runners/t081_t083_performance_worker.py"
    spec = importlib.util.spec_from_file_location("t081_t083_worker_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_sources(record: dict) -> None:
    sources = record.get("source_files", {})
    require(isinstance(sources, dict) and sources, "功能原件缺少实际加载源码哈希")
    for source, digest in sources.items():
        path = Path(source)
        require(path.is_file() and sha256(path) == digest, f"实际加载源码已变化：{source}")


def selected_sources(record: dict, distributed: bool) -> dict[str, str]:
    suffixes = (
        ("/bucketing.py", "/post_grad.py", "/triton_experimental/config.py")
        if distributed
        else ("/joint_graph.py", "/triton_experimental/config.py")
    )
    selected = {
        path: digest
        for path, digest in record["source_files"].items()
        if path.endswith(suffixes)
    }
    require(selected, "未在实际加载模块中找到目标pass/backend源码")
    return selected


def graph_after(run_dir: Path, unit: str) -> str:
    rank_dir = run_dir / unit / "on" / "rank-0"
    if unit in EXPECTED_COLLECTIVES:
        path = rank_dir / "post-grad-final.txt"
        require(path.is_file(), f"缺少post-grad最终图：{path}")
        return path.read_text(encoding="utf-8")
    paths = sorted(rank_dir.glob("target-*-after.txt"))
    require(paths, f"缺少目标改写after图：{rank_dir}")
    return "\n\n".join(path.read_text(encoding="utf-8") for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_UNITS, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reviewer", default="tracker-control-node")
    args = parser.parse_args()
    require(Path.cwd().resolve() == WORK, f"必须从 {WORK} 启动")

    worker = load_worker()
    worker_path = ROOT / "runners/t081_t083_performance_worker.py"
    worker_digest = sha256(worker_path)
    run_dir = args.run_dir.resolve()
    require(run_dir.is_dir(), f"运行目录不存在：{run_dir}")
    destination = ROOT / "results/current" / args.task
    functional_dir = destination / "functional"
    gate_dir = destination / "performance_gates"
    functional_dir.mkdir(parents=True, exist_ok=True)
    gate_dir.mkdir(parents=True, exist_ok=True)
    gpu_review = destination / "gpu_reference_review.json"
    require(gpu_review.is_file(), f"缺少GPU复核快照：{gpu_review}")
    gpu_digest = sha256(gpu_review)
    reviewed_at = datetime.now().astimezone().isoformat()
    task_results = []

    for unit in TASK_UNITS[args.task]:
        task_id, acceptance_unit_id, _ = worker.TARGETS[unit]
        world_size = 2 if args.task == "T-083" else 1
        mode_records = {}
        for mode in ("off", "on"):
            summary_path = run_dir / unit / mode / "functional_summary.json"
            require(summary_path.is_file(), f"缺少功能summary：{summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            require(summary.get("task_id") == task_id, "task_id不匹配")
            require(summary.get("acceptance_unit_id") == acceptance_unit_id, "acceptance_unit_id不匹配")
            require(summary.get("mode") == mode, "OFF/ON目录与记录不匹配")
            require(summary.get("backend") == "triton_experimental", "不是triton_experimental")
            require(summary.get("world_size") == world_size, "world_size不满足合同")
            require(summary.get("correctness") == "passed", "数值功能未通过")
            ranks = summary.get("rank_results", [])
            require(len(ranks) == world_size, "rank证据不完整")
            for rank, record in enumerate(ranks):
                require(record.get("rank") == rank, "rank编号不连续")
                require(record.get("mode") == mode, "rank mode不匹配")
                require(record.get("backend") == "triton_experimental", "rank backend不匹配")
                require(record.get("backend_selected_before_import") is True, "后端未在import前选择")
                require(record.get("pytorch_commit") == worker.COMMIT, "PyTorch commit不匹配")
                require(record.get("pytorch_worktree_status") == "", "PyTorch工作树不干净")
                require(record.get("worker_sha256") == worker_digest, "功能原件与当前worker源码不一致")
                require(record.get("input_spec") == worker.input_spec(unit), "输入合同不一致")
                require(record.get("numerical_execution") is True, "不是实际数值执行")
                require(record.get("graph_breaks") == 0 and record.get("fallbacks") == 0, "存在graph break/fallback")
                require(record.get("product_disabled") is False, "产品明确关闭时不得签门禁")
                require(record.get("world_size") == world_size, "rank world_size不匹配")
                require(record.get("correctness") == "passed", "rank正确性失败")
                require(record.get("target_rewrite") == ("confirmed" if mode == "on" else "disabled-control"), "目标改写状态不符")
                if world_size == 2:
                    require(record.get("process_group_backend") == "hccl", "NPU通信必须真实HCCL")
                verify_sources(record)
                state = record.get("state", {})
                if mode == "off":
                    require(state.get("handler_calls") == 0, "OFF仍进入目标handler")
                else:
                    require(state.get("handler_calls", 0) > 0 and state.get("graph_changes", 0) > 0, "ON未发生目标改写")
                    if unit in EXPECTED_COLLECTIVES:
                        before, after = EXPECTED_COLLECTIVES[unit]
                        require({"before": before, "after": after} in state.get("collective_counts", []), "collective分桶数量不符")
                        require(state.get("post_grad_counts") == [after], "post-grad最终collective数量不符")
            mode_records[mode] = (summary_path, summary)

        on_path, on_summary = mode_records["on"]
        off_path, off_summary = mode_records["off"]
        on_record = on_summary["rank_results"][0]
        compact = {
            "schema_version": "1.0",
            "task_id": task_id,
            "acceptance_unit_id": acceptance_unit_id,
            "unit": unit,
            "backend": "triton_experimental",
            "pytorch_commit": worker.COMMIT,
            "correctness": "passed",
            "numerical_execution": True,
            "target_rewrite": "confirmed",
            "graph_breaks": 0,
            "fallbacks": 0,
            "product_disabled": False,
            "measurement_workload": unit + "-community-shape",
            "world_size": world_size,
            "process_group_backend": "hccl" if world_size == 2 else None,
            "input_spec": worker.input_spec(unit),
            "worker_sha256": worker_digest,
            "source_files": selected_sources(on_record, world_size == 2),
            "off_observation": [record["state"] for record in off_summary["rank_results"]],
            "on_observation": [record["state"] for record in on_summary["rank_results"]],
            "target_graph_after": graph_after(run_dir, unit),
            "source_run_dir": str(run_dir),
            "source_off_summary": {"path": str(off_path), "sha256": sha256(off_path)},
            "source_on_summary": {"path": str(on_path), "sha256": sha256(on_path)},
            "reviewed_at": reviewed_at,
            "reviewer": args.reviewer,
        }
        compact_path = functional_dir / f"{unit}-on.json"
        compact_path.write_text(json.dumps(compact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        gate = {
            key: compact[key]
            for key in (
                "task_id", "acceptance_unit_id", "backend", "pytorch_commit",
                "correctness", "target_rewrite", "graph_breaks", "fallbacks",
                "product_disabled", "measurement_workload", "world_size",
                "input_spec", "worker_sha256",
            )
        }
        gate.update(
            schema_version="1.0",
            reviewed_at=reviewed_at,
            reviewer=args.reviewer,
            gpu_reference={"path": "../gpu_reference_review.json", "sha256": gpu_digest},
            target_functional={"path": f"../functional/{unit}-on.json", "sha256": sha256(compact_path)},
        )
        gate_path = gate_dir / f"{unit}.json"
        gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        worker.read_gate(gate_path, unit, "npu")
        task_results.append(
            {
                "unit": unit,
                "acceptance_unit_id": acceptance_unit_id,
                "status": "functional-passed-performance-gate-signed",
                "gate": str(gate_path.relative_to(ROOT)),
                "functional_evidence": str(compact_path.relative_to(ROOT)),
            }
        )

    summary = {
        "schema_version": "1.0",
        "task_id": args.task,
        "generated_at": reviewed_at,
        "backend": "triton_experimental",
        "source_run_dir": str(run_dir),
        "status": "functional-passed-performance-gates-signed",
        "units": task_results,
    }
    summary_path = destination / "npu_functional_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"functional_review=OK task={args.task} units={len(task_results)}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
