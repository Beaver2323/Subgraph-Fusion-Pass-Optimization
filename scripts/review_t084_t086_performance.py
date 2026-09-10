#!/usr/bin/env python3
"""复核T-084～T-086六臂性能、归档原始样本/代码并生成正式结论。"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics


ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/home/z50063656/tmp")
ORDER = ("off1", "on1", "on2", "off2", "off3", "on3")
DEBUG_FILES = (
    "fx_graph_readable.py",
    "fx_graph_transformed.py",
    "ir_pre_fusion.txt",
    "ir_post_fusion.txt",
    "output_code.py",
)
UNITS = {
    "dedup-reduce-scatter": {
        "task": "T-084",
        "acceptance_unit_id": "AU-post-grad-dedup-reduce-scatters",
        "world_size": 2,
        "intent": "把两个等价reduce-scatter输入先相加，再只发起一次collective，减少通信次数。",
        "source": [
            "torch/_inductor/fx_passes/fsdp.py:dedup_reduce_scatters",
            "torch/_inductor/fx_passes/post_grad.py:post_grad_passes",
        ],
        "gpu": "A100原生社区reference通过；该reference为单rank结构合同，不外推双rank性能。",
        "npu": "Ascend910B2真实2-rank HCCL中，ON把2个reduce-scatter收敛为1个并保持数值。",
    },
    "pointless-cumsum": {
        "task": "T-085",
        "acceptance_unit_id": "AU-post-grad-pointless-cumsum",
        "world_size": 1,
        "intent": "把常量full的cumsum替换为arange乘常量，期望消除通用scan。",
        "source": [
            "torch/_inductor/fx_passes/post_grad.py:pointless_cumsum_replacement",
        ],
        "gpu": "A100原生社区11个dtype/输出dtype变体通过；GPU结果只证明语义合同。",
        "npu": "社区最大shape上ON由1个Triton full+原生cumsum变为2个Triton kernel，稳定回退，已增加NPU专属默认关闭门禁。",
    },
    "overlap-device-put": {
        "task": "T-085",
        "acceptance_unit_id": "AU-post-grad-overlap-scheduling-device-put-sync",
        "world_size": 2,
        "intent": "调度重排前把异步device_put改成同步，避免CPU过早读取未完成拷贝。",
        "source": [
            "torch/_inductor/fx_passes/overlap_scheduling.py:make_all_device_put_sync",
            "torch/_inductor/fx_passes/post_grad.py:schedule_overlap_bucketing_from_inductor_configs",
        ],
        "gpu": "A100真实2-rank NCCL社区reference直接断言device_put同步化和数值。",
        "npu": "Ascend910B2真实2-rank HCCL完成同一安全改写；分析估时器的CUDA带宽探针已最小替换为NPU显式带宽配置。",
    },
    "partitioned-scatter": {
        "task": "T-085",
        "acceptance_unit_id": "AU-post-grad-partitioned-scatter-optimization",
        "world_size": 1,
        "intent": "把高争用原子scatter拆成多个分区buffer，最后归约，以空间换更低原子冲突。",
        "source": [
            "torch/_inductor/fx_passes/reduced_atomic_contention.py:partitioned_scatter_optimization_pass",
            "torch/_inductor/fx_passes/post_grad.py:post_grad_passes",
        ],
        "gpu": "A100原生功能reference三次改写、负例与显式OFF通过；CUDA默认关闭且未测性能。",
        "npu": "直接复用社区百万行benchmark，ON改写3次并使用真实NPU显存预算，但六臂稳定回退。",
    },
    "reinplace-index-put": {
        "task": "T-086",
        "acceptance_unit_id": "AU-post-grad-reinplace-inplaceable-ops",
        "world_size": 1,
        "intent": "在别名/生命周期安全时把index_put+copy_改为index_put_，消除中间结果和复制。",
        "source": [
            "torch/_inductor/fx_passes/reinplace.py:reinplace_inplaceable_ops",
            "torch/_inductor/fx_passes/post_grad.py:post_grad_passes",
        ],
        "gpu": "A100原生正例发生index_put_改写，负例保持不变。",
        "npu": "Ascend910B2发生相同FX改写；最终index_put_走已注册NPU extern lowering，不属于CPU fallback或Triton融合核。",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict:
    require(path.is_file(), f"缺少原件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def timing_and_memory(raw: dict) -> tuple[dict, dict]:
    if "comparison" in raw and raw["comparison"] is not None:
        memory = {}
        arms = raw["arms"]
        for key in ("peak_allocated", "peak_reserved"):
            memory[key] = {
                mode: statistics.median(
                    arms[f"{mode}{index}"]["memory"][key]
                    for index in (1, 2, 3)
                )
                for mode in ("off", "on")
            }
        return raw["comparison"], memory
    return raw["timing"], raw["memory"]


def arm_worker_paths(run: Path, unit: str, arm: str) -> list[Path]:
    world_size = UNITS[unit]["world_size"]
    rank_paths = [run / arm / f"rank-{rank}/worker_result.json" for rank in range(world_size)]
    if all(path.is_file() for path in rank_paths):
        return rank_paths
    require(world_size == 1, f"{unit}/{arm}缺少rank结果")
    path = run / arm / "worker_result.json"
    require(path.is_file(), f"缺少worker结果：{path}")
    return [path]


def validate_run(run: Path, unit: str) -> tuple[dict, str]:
    meta = UNITS[unit]
    raw = load(run / "performance_summary.json")
    require(raw.get("task_id") == meta["task"], f"{unit} task_id错误")
    if raw.get("unit") is not None:
        require(raw.get("unit") == unit, f"{unit}单元名错误")
    pids = set()
    gate_hashes = set()
    source_fingerprints = {}
    for arm in ORDER:
        records = [load(path) for path in arm_worker_paths(run, unit, arm)]
        require(len(records) == meta["world_size"], f"{unit}/{arm} world_size错误")
        for record in records:
            require(record.get("backend") == "triton_experimental", f"{unit}后端错误")
            require(record.get("correctness") == "passed", f"{unit}性能臂数值失败")
            require(record.get("mode") == arm.rstrip("123"), f"{unit}/{arm} mode错误")
            require(record.get("pid") not in pids, f"{unit}六臂复用了进程")
            pids.add(record["pid"])
            gate_hashes.add(record.get("gate_sha256"))
            for path, digest in record.get("loaded_source_sha256", {}).items():
                if path in source_fingerprints:
                    require(source_fingerprints[path] == digest, f"{unit}六臂源码变化")
                source_fingerprints[path] = digest
        states = [record.get("target_state") or record.get("state") for record in records]
        if unit == "dedup-reduce-scatter":
            expected = [2] if arm.startswith("off") else [1]
            require(all(state["post_grad_counts"] == expected for state in states), f"{unit}/{arm}结构错误")
        elif unit == "pointless-cumsum":
            calls = 0 if arm.startswith("off") else 1
            require(all((state["handler_calls"] >= 1) == bool(calls) for state in states), f"{unit}/{arm}命中错误")
        elif unit == "overlap-device-put":
            require(meta["world_size"] == 2 and all(record.get("process_group_backend") == "hccl" for record in records), "overlap必须真实HCCL")
            if arm.startswith("on"):
                require(all(state["scheduler_calls"] >= 1 and state["converted_device_puts"] >= 1 for state in states), "overlap ON未改写")
            else:
                require(all(state["scheduler_calls"] == 0 for state in states), "overlap OFF错误")
        elif unit == "partitioned-scatter":
            if arm.startswith("on"):
                require(all(state["partitioned_scatter_applied"] >= 3 and state["memory_probe_calls"] >= 1 for state in states), "scatter ON门禁错误")
            else:
                require(all(state["partitioned_scatter_applied"] == 0 for state in states), "scatter OFF错误")
        else:
            if arm.startswith("on"):
                require(all(state["graph_changes"] >= 1 for state in states), "reinplace ON未改写")
            else:
                require(all(state["graph_changes"] == 0 for state in states), "reinplace OFF错误")
    require(len(gate_hashes) == 1 and None not in gate_hashes, f"{unit} gate不唯一")
    return raw, next(iter(gate_hashes))


def find_debug_dir(run: Path, unit: str, arm: str) -> Path:
    base = arm_worker_paths(run, unit, arm)[0].parent
    candidates = sorted(
        path.parent
        for path in (base / "debug/torch_compile_debug").rglob("output_code.py")
        if "model__0_" in str(path)
    )
    require(candidates, f"{unit}/{arm}缺少model__0 debug产物")
    return candidates[0]


def archive(run: Path, unit: str, raw: dict) -> tuple[dict, list[dict]]:
    task = UNITS[unit]["task"]
    base = ROOT / "results/current" / task / "performance"
    raw_dir = base / "raw" / unit
    code_dir = base / "codegen" / unit
    raw_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    summary_target = raw_dir / "performance_summary.json"
    shutil.copy2(run / "performance_summary.json", summary_target)
    inventory = [
        {
            "path": str(summary_target.relative_to(ROOT)),
            "sha256": sha256(summary_target),
            "bytes": summary_target.stat().st_size,
        }
    ]
    for arm in ORDER:
        for index, source in enumerate(arm_worker_paths(run, unit, arm)):
            suffix = f"-rank-{index}" if UNITS[unit]["world_size"] > 1 else ""
            target = raw_dir / f"{arm}{suffix}-worker_result.json"
            shutil.copy2(source, target)
            inventory.append(
                {
                    "path": str(target.relative_to(ROOT)),
                    "sha256": sha256(target),
                    "bytes": target.stat().st_size,
                }
            )
    codegen = {}
    for mode, arm in (("off", "off1"), ("on", "on1")):
        source_dir = find_debug_dir(run, unit, arm)
        mode_dir = code_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in DEBUG_FILES:
            source = source_dir / name
            require(source.is_file(), f"缺少{unit}/{mode}/{name}")
            target = mode_dir / name
            shutil.copy2(source, target)
            files[name] = {
                "path": str(target.relative_to(ROOT)),
                "sha256": sha256(target),
                "bytes": target.stat().st_size,
            }
            inventory.append(files[name])
        text = (source_dir / "output_code.py").read_text(
            encoding="utf-8", errors="replace"
        )
        codegen[mode] = {
            "files": files,
            "triton_kernel_definitions": len(
                re.findall(r"^\w+ = async_compile\.triton\(", text, re.MULTILINE)
            ),
            "reduce_scatter_calls": len(
                re.findall(
                    r"^\s+buf\d+ = torch\.ops\._c10d_functional\.reduce_scatter_tensor\.default\(",
                    text,
                    re.MULTILINE,
                )
            ),
            "native_cumsum_calls": len(
                re.findall(
                    r"^\s+buf\d+ = torch\.ops\.aten\.cumsum\.default\(",
                    text,
                    re.MULTILINE,
                )
            ),
            "npu_index_put_inplace_calls": len(
                re.findall(r"^\s+aten\.index_put_\(", text, re.MULTILINE)
            ),
        }
    return codegen, inventory


def round_spread(raw: dict, metric: str = "event_ms") -> dict[str, float] | None:
    arms = raw.get("arms")
    if not arms:
        return None
    result = {}
    for mode in ("off", "on"):
        values = [arms[f"{mode}{index}"]["timing"][metric]["p50"] for index in (1, 2, 3)]
        result[mode] = max(values) / min(values) if min(values) else float("inf")
    return result


def classify(unit: str, timing: dict, spread: dict[str, float] | None) -> tuple[str, str, str]:
    event_p50 = timing["event_ms"]["p50"]["improvement_percent"]
    event_p99 = timing["event_ms"]["p99"]["improvement_percent"]
    host_p99 = timing["host_ms"]["p99"]["improvement_percent"]
    if spread and max(spread.values()) > 1.5:
        return (
            "PERF_MIXED",
            "measured-mixed-high-variance",
            "不声明性能收益；保留安全功能并扩大稳定拓扑/模型复核",
        )
    if event_p50 >= 5 and event_p99 >= 5 and host_p99 >= -5:
        return "PERF_IMPROVED", "measured-improved", "保留现状"
    if event_p50 <= -5 and event_p99 <= -5:
        action = "保持默认关闭"
        if unit == "pointless-cumsum":
            action = "已增加triton_experimental NPU专属默认关闭门禁"
        return "PERF_REGRESSED", "measured-regressed", action
    if event_p50 >= 5 or event_p99 <= -5 or host_p99 <= -5:
        return (
            "PERF_MIXED",
            "measured-mixed",
            "不声明稳定收益；保留功能行为并扩大尾延迟复核",
        )
    return "PERF_NEUTRAL", "measured-neutral", "不因微小差异改变产品配置"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, help="UNIT=/absolute/run_dir"
    )
    parser.add_argument("--pointless-cumsum-product-gate", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd().resolve() == WORK, f"必须从{WORK}启动")
    runs = {}
    for item in args.run:
        unit, separator, path = item.partition("=")
        require(separator and unit not in runs, f"无效或重复--run：{item}")
        runs[unit] = Path(path).resolve()
    require(set(runs) == set(UNITS), "--run必须精确覆盖五个单元")

    product_gate_path = args.pointless_cumsum_product_gate.resolve()
    product_gate = load(product_gate_path)
    require(product_gate.get("device_execution") is True, "产品门禁不是实际NPU执行")
    require(product_gate.get("correctness") == "passed", "产品门禁验证失败")
    require(product_gate.get("handler_calls") == 0, "产品门禁仍命中handler")
    require(product_gate.get("native_cumsum_calls") == 1, "产品门禁未保留原生cumsum")
    product_target = ROOT / "results/current/T-085/fixes/pointless-cumsum/product_gate_result.json"
    product_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(product_gate_path, product_target)

    task_units = {"T-084": [], "T-085": [], "T-086": []}
    for unit, meta in UNITS.items():
        raw, gate_digest = validate_run(runs[unit], unit)
        timing, memory = timing_and_memory(raw)
        spread = round_spread(raw)
        verdict, performance_status, action = classify(unit, timing, spread)
        codegen, inventory = archive(runs[unit], unit, raw)
        record = {
            "acceptance_unit_id": meta["acceptance_unit_id"],
            "unit": unit,
            "pattern_explanation": {
                "intent": meta["intent"],
                "source": meta["source"],
                "gpu_behavior": meta["gpu"],
                "npu_behavior": meta["npu"],
            },
            "performance_status": performance_status,
            "verdict": verdict,
            "product_action": action,
            "timing": timing,
            "round_p50_spread_ratio": spread,
            "memory": memory,
            "codegen": codegen,
            "raw_run_dir": str(runs[unit]),
            "raw_summary_sha256": sha256(runs[unit] / "performance_summary.json"),
            "gate_sha256": gate_digest,
            "artifact_inventory": inventory,
        }
        if unit == "pointless-cumsum":
            record["product_gate_verification"] = {
                "path": str(product_target.relative_to(ROOT)),
                "sha256": sha256(product_target),
                "device_execution": True,
            }
        task_units[meta["task"]].append(record)

    generated_at = datetime.now().astimezone().isoformat()
    for task, records in task_units.items():
        verdict_counts = {}
        for record in records:
            verdict = record["verdict"]
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        result = {
            "schema_version": "1.0",
            "generated_at": generated_at,
            "task_id": task,
            "status": "performance-disposition-complete",
            "backend": "triton_experimental",
            "measurement_contract": {
                "order": list(ORDER),
                "process_isolation": "fresh-process-per-arm",
                "warmup": 10,
                "runs": 100,
                "timing": "同步host wall与NPU Event p50/p99",
                "scope": "社区目标图或社区功能图派生microbenchmark；非完整模型端到端",
            },
            "completion": {
                "acceptance_units": len(records),
                "measured_units": len(records),
                "pending_units": 0,
            },
            "verdict_counts": verdict_counts,
            "acceptance_units": records,
        }
        output = ROOT / "results/current" / task / "performance_summary.json"
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"performance_review=OK task={task} units={len(records)}")
        print(f"summary={output}")


if __name__ == "__main__":
    main()
