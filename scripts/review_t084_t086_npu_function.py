#!/usr/bin/env python3
"""复核T-084～T-086真实NPU功能结果，归档图证据并签发性能门禁。"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/home/z50063656/tmp")
ARTIFACT_NAMES = (
    "fx_graph_readable.py",
    "fx_graph_transformed.py",
    "ir_pre_fusion.txt",
    "ir_post_fusion.txt",
    "output_code.py",
)
COMMUNITY_ALIGNMENT = {
    "dedup-reduce-scatter": {
        "status": "ALIGNED_WITH_ADAPTER",
        "aligned_scope": ["线性avg reduce-scatter从2次收敛为1次", "两rank数值正确"],
        "divergent_scope": [],
        "open_scope": ["GPU原生reference为单rank，NPU补充为真实两rankHCCL"],
        "disposition": "设备/进程组作最小适配；合同与社区一致，性能只按NPU两rank实测判定。",
    },
    "overlap-device-put": {
        "status": "ALIGNED_WITH_ADAPTER",
        "aligned_scope": ["异步device_put转同步", "真实两rank数值与collective顺序"],
        "divergent_scope": [],
        "open_scope": ["NPU分析估时使用显式NPU带宽，社区CUDA路径运行时探测CUDA带宽"],
        "disposition": "保留社区安全改写；带宽来源是后端最小适配，不把高方差数据写成收益。",
    },
    "partitioned-scatter": {
        "status": "PARTIALLY_ALIGNED",
        "aligned_scope": ["三个高争用scatter改写", "accumulate=False负例", "社区benchmark输入合同"],
        "divergent_scope": ["社区默认只为HIP开启，NPU能力验证需独立启用", "NPU六臂性能回退"],
        "open_scope": [],
        "disposition": "能力路径正确但不改变默认关闭；NPU显存探针替换CUDA专用探针。",
    },
    "pointless-cumsum": {
        "status": "PARTIALLY_ALIGNED",
        "aligned_scope": ["开启候选时命中full+cumsum并保持数值/dtype"],
        "divergent_scope": ["NPU性能显著回退，因此产品默认关闭；CUDA社区仍允许命中"],
        "open_scope": [],
        "disposition": "保留上游CPU/CUDA行为，仅在triton_experimental NPU通过可逆gate关闭。",
    },
    "reinplace-index-put": {
        "status": "ALIGNED_WITH_BACKEND_LOWERING_DIFFERENCE",
        "aligned_scope": ["dead input改为index_put_", "live input负例保持functional", "mutation/alias正确"],
        "divergent_scope": ["NPU index_put_最终走已注册extern lowering，并非Triton融合scatter核"],
        "open_scope": ["尾延迟仍需更大模型复核"],
        "disposition": "保留功能改写；性能结论为混合，不外推稳定收益。",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict:
    require(path.is_file(), f"原件不存在：{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"原件顶层不是object：{path}")
    return value


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"无法加载：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_sources(sources: dict[str, str]) -> dict[str, str]:
    markers = (
        "/fx_passes/fsdp.py",
        "/fx_passes/post_grad.py",
        "/fx_passes/overlap_scheduling.py",
        "/fx_passes/reduced_atomic_contention.py",
        "/fx_passes/reinplace.py",
        "/triton_experimental/config.py",
        "/triton_experimental/runtime_estimation.py",
        "/triton_experimental/overrides.py",
    )
    selected = {
        path: digest
        for path, digest in sources.items()
        if path.endswith(markers)
    }
    require(selected, "功能原件没有目标pass或triton_experimental源码哈希")
    for name, digest in selected.items():
        path = Path(name)
        require(path.is_file() and sha256(path) == digest, f"实际源码已变化：{name}")
    return selected


def copy_artifacts(
    run_dir: Path, task: str, unit: str, rank_style: bool
) -> list[dict[str, object]]:
    destination = ROOT / "results/current" / task / "functional/artifacts" / unit
    inventory = []
    for mode in ("off", "on"):
        base = run_dir / mode
        if rank_style:
            base = base / "rank-0"
        debug_candidates = sorted(
            path
            for path in (base / "debug/torch_compile_debug").glob(
                "run_*/torchinductor/model__0_*/*"
            )
            if path.name in ARTIFACT_NAMES
        )
        explicit = sorted(base.glob("target-*-before.txt"))
        explicit += sorted(base.glob("target-*-after.txt"))
        explicit += sorted(base.glob("post-grad-final.txt"))
        require(debug_candidates, f"缺少{task}/{unit}/{mode}编译图证据")
        mode_dir = destination / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        for source in debug_candidates + explicit:
            target = mode_dir / source.name
            shutil.copy2(source, target)
            inventory.append(
                {
                    "mode": mode,
                    "path": str(target.relative_to(ROOT)),
                    "sha256": sha256(target),
                    "bytes": target.stat().st_size,
                }
            )
    return inventory


def validate_common(summary: dict, task: str, worker: Path) -> None:
    require(summary.get("task_id") == task, f"{task} task_id不匹配")
    require(summary.get("backend") == "triton_experimental", f"{task}后端错误")
    require(summary.get("pytorch_commit") == "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b", f"{task} commit错误")
    require(summary.get("correctness") == "passed", f"{task}正确性未通过")
    require(summary.get("numerical_execution") is True, f"{task}非真实数值执行")
    require(summary.get("target_rewrite") == "confirmed", f"{task}目标改写未确认")
    require(summary.get("graph_breaks") == 0, f"{task}存在graph break")
    require(summary.get("fallbacks") == 0, f"{task}存在CPU/图外fallback")
    require(summary.get("product_disabled") is False, f"{task}产品已关闭")
    require(summary.get("worker_sha256") == sha256(worker), f"{task} worker已变化")
    require(set(summary.get("arms", {})) == {"off", "on"}, f"{task}缺少OFF/ON")
    sources = summary.get("source_files", {})
    require(isinstance(sources, dict) and sources, f"{task}缺少源码哈希")


def compact_and_gate(
    task: str,
    unit: str,
    summary: dict,
    source_run: Path,
    artifacts: list[dict[str, object]],
    gate_reader,
    reviewer: str,
) -> dict:
    destination = ROOT / "results/current" / task
    functional_dir = destination / "functional"
    gate_dir = destination / "performance_gates"
    functional_dir.mkdir(parents=True, exist_ok=True)
    gate_dir.mkdir(parents=True, exist_ok=True)
    reviewed_at = datetime.now().astimezone().isoformat()
    compact = dict(summary)
    compact["source_files"] = selected_sources(summary["source_files"])
    compact["source_run_dir"] = str(source_run)
    compact["artifact_inventory"] = artifacts
    compact["reviewed_at"] = reviewed_at
    compact["reviewer"] = reviewer
    compact_path = functional_dir / f"{unit}.json"
    compact_path.write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    gate_fields = (
        "task_id",
        "acceptance_unit_id",
        "backend",
        "pytorch_commit",
        "correctness",
        "target_rewrite",
        "graph_breaks",
        "fallbacks",
        "product_disabled",
        "measurement_workload",
        "world_size",
        "input_spec",
        "worker_sha256",
    )
    if task == "T-086":
        gate_fields += ("fallback_scope", "allowed_device_lowering")
    gate = {key: compact[key] for key in gate_fields}
    gpu_review = destination / "gpu_reference_review.json"
    require(gpu_review.is_file(), f"缺少GPU复核原件：{gpu_review}")
    gate.update(
        schema_version="1.0",
        reviewed_at=reviewed_at,
        reviewer=reviewer,
        gpu_reference={
            "path": "../gpu_reference_review.json",
            "sha256": sha256(gpu_review),
        },
        target_functional={
            "path": f"../functional/{unit}.json",
            "sha256": sha256(compact_path),
        },
    )
    gate_path = gate_dir / f"{unit}.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    gate_reader(gate_path)

    # 社区对齐结论是控制节点复核元数据，不能写回已由 performance gate
    # 按 SHA256 绑定的功能原件。使用独立 sidecar，并反向绑定功能原件。
    alignment_dir = destination / "community_alignment"
    alignment_dir.mkdir(parents=True, exist_ok=True)
    alignment_path = alignment_dir / f"{unit}.json"
    alignment = {
        "schema_version": "1.0",
        "task_id": task,
        "acceptance_unit_id": compact["acceptance_unit_id"],
        "unit": unit,
        "generated_at": reviewed_at,
        "source_functional": {
            "path": f"../functional/{unit}.json",
            "sha256": sha256(compact_path),
        },
        "community_alignment": COMMUNITY_ALIGNMENT[unit],
    }
    alignment_path.write_text(
        json.dumps(alignment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "unit": unit,
        "acceptance_unit_id": compact["acceptance_unit_id"],
        "status": "functional-passed-performance-gate-signed",
        "functional_evidence": str(compact_path.relative_to(ROOT)),
        "gate": str(gate_path.relative_to(ROOT)),
        "community_alignment": str(alignment_path.relative_to(ROOT)),
        "artifacts": len(artifacts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t084-run", type=Path, required=True)
    parser.add_argument("--t085-cumsum-run", type=Path, required=True)
    parser.add_argument("--t085-overlap-run", type=Path, required=True)
    parser.add_argument("--t085-scatter-run", type=Path, required=True)
    parser.add_argument("--t086-run", type=Path, required=True)
    parser.add_argument("--reviewer", default="tracker-control-node")
    args = parser.parse_args()
    require(Path.cwd().resolve() == WORK, f"必须从 {WORK} 启动")

    t084_worker = ROOT / "runners/t084_performance_worker.py"
    t085_worker = ROOT / "runners/t085_performance_worker.py"
    t086_worker = ROOT / "runners/t086_performance_worker.py"
    t084_launcher = load_module("t084_review_launcher", ROOT / "scripts/run_t084_performance.py")
    t084_module = load_module("t084_review_worker", t084_worker)
    t085_module = load_module("t085_review_worker", t085_worker)
    t086_module = load_module("t086_review_worker", t086_worker)

    runs = {
        "T-084": args.t084_run.resolve(),
        "T-085-cumsum": args.t085_cumsum_run.resolve(),
        "T-085-overlap": args.t085_overlap_run.resolve(),
        "T-085-scatter": args.t085_scatter_run.resolve(),
        "T-086": args.t086_run.resolve(),
    }
    for name, path in runs.items():
        require(path.is_dir(), f"{name}运行目录不存在：{path}")

    t084 = t084_launcher.aggregate_functional(runs["T-084"], "npu")
    t085 = {
        "pointless-cumsum": load(runs["T-085-cumsum"] / "functional_summary.json"),
        "overlap-device-put": load(runs["T-085-overlap"] / "functional_summary.json"),
        "partitioned-scatter": load(runs["T-085-scatter"] / "functional_summary.json"),
    }
    t086 = load(runs["T-086"] / "functional_summary.json")

    validate_common(t084, "T-084", t084_worker)
    require(t084.get("world_size") == 2 and t084.get("process_group_backend") == "hccl", "T-084必须是2-rank HCCL")
    require(
        all(item["post_grad_counts"] == [1] for item in t084["arms"]["on"]["observations"]),
        "T-084 ON未把两个reduce-scatter去重为一个",
    )
    task_results: dict[str, list[dict]] = {"T-084": [], "T-085": [], "T-086": []}
    artifacts = copy_artifacts(runs["T-084"], "T-084", "dedup-reduce-scatter", True)
    task_results["T-084"].append(
        compact_and_gate(
            "T-084",
            "dedup-reduce-scatter",
            t084,
            runs["T-084"],
            artifacts,
            lambda path: t084_module.read_gate(path, "npu"),
            args.reviewer,
        )
    )

    t085_runs = {
        "pointless-cumsum": runs["T-085-cumsum"],
        "overlap-device-put": runs["T-085-overlap"],
        "partitioned-scatter": runs["T-085-scatter"],
    }
    for unit, summary in t085.items():
        validate_common(summary, "T-085", t085_worker)
        off = summary["arms"]["off"]["observations"]
        on = summary["arms"]["on"]["observations"]
        if unit == "pointless-cumsum":
            require(off[0]["handler_calls"] == 0 and on[0]["handler_calls"] >= 1, "cumsum目标状态错误")
        elif unit == "overlap-device-put":
            require(summary.get("world_size") == 2 and summary.get("process_group_backend") == "hccl", "overlap必须是2-rank HCCL")
            require(all(item["scheduler_calls"] == 0 for item in off), "overlap OFF错误")
            require(all(item["converted_device_puts"] >= 1 for item in on), "overlap ON未改写")
        else:
            require(off[0]["partitioned_scatter_applied"] == 0, "scatter OFF错误")
            require(on[0]["partitioned_scatter_applied"] >= 3, "scatter ON改写不足")
            require(on[0]["memory_probe_calls"] >= 1 and any(on[0]["memory_state"]), "scatter未使用NPU显存门禁")
            require(on[0]["negative_accumulate_false_applied"] == 0, "scatter负例误命中")
        artifacts = copy_artifacts(t085_runs[unit], "T-085", unit, True)
        task_results["T-085"].append(
            compact_and_gate(
                "T-085",
                unit,
                summary,
                t085_runs[unit],
                artifacts,
                lambda path, name=unit: t085_module.read_gate(path, name, "npu"),
                args.reviewer,
            )
        )

    validate_common(t086, "T-086", t086_worker)
    require(t086.get("allowed_device_lowering") == "registered-IndexPutFallback-extern", "T-086 lowering边界错误")
    require(t086["arms"]["off"]["target_rewrite"] == "disabled-control", "T-086 OFF错误")
    require(t086["arms"]["on"]["target_rewrite"] == "confirmed", "T-086 ON错误")
    artifacts = copy_artifacts(runs["T-086"], "T-086", "reinplace-index-put", False)
    task_results["T-086"].append(
        compact_and_gate(
            "T-086",
            "reinplace-index-put",
            t086,
            runs["T-086"],
            artifacts,
            lambda path: t086_module.read_gate(path, "npu"),
            args.reviewer,
        )
    )

    generated_at = datetime.now().astimezone().isoformat()
    for task, results in task_results.items():
        summary = {
            "schema_version": "1.0",
            "task_id": task,
            "generated_at": generated_at,
            "backend": "triton_experimental",
            "status": "functional-passed-performance-gates-signed",
            "units": results,
        }
        path = ROOT / "results/current" / task / "npu_functional_summary.json"
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"functional_review=OK task={task} units={len(results)}")
        print(f"summary={path}")


if __name__ == "__main__":
    main()
