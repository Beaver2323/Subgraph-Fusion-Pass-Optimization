#!/usr/bin/env python3
"""复核 T-087～T-090 社区合同、独立 OFF/ON 和六臂性能，保留逐例图证据。"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_current_acceptance_matrix import npu_contract_progress
from review_t084_t086_npu_function import require, load, sha256

SPEC = importlib.util.spec_from_file_location("worker", ROOT / "runners/t087_t090_performance_worker.py")
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)
CASES = {
    "reorder-locality": "REF-reorder-locality-training-native",
    "select-cat-aten": "REF-select-cat-aten-native",
    "split-cat-aten": "REF-split-cat-aten-native",
    "move-view-after-cat": "REF-move-view-after-cat-aten-native",
    "normalize-cat-aten": "REF-normalize-cat-default-aten-native",
}
NAMES = {"fx_graph_readable.py", "fx_graph_transformed.py", "ir_pre_fusion.txt",
         "ir_post_fusion.txt", "output_code.py", "result.json", "stdout.log", "stderr.log", "execution.json"}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def ref(path, parent):
    import os
    return {"path": os.path.relpath(path, parent), "sha256": sha256(path)}


def audit_code(base):
    codes = sorted((base / "debug").rglob("output_code.py"))
    require(codes, f"无生成代码：{base}")
    extern = set()
    aten_calls = {}
    for code in codes:
        text = code.read_text()
        # 全extern图不会import Triton后端模块；实际backend由worker注册检查+源码哈希证明。
        require("torch.npu" in text, f"生成代码没有NPU设备操作：{code}")
        require(not re.search(r"\.cpu\(|device=['\"]cpu|\.to\(['\"]cpu", text), f"CPU路径：{code}")
        calls = set(re.findall(r"extern_kernels\.([\w]+)\(", text))
        # 此批仅训练矩阵乘允许已注册 NPU mm；不能把任意 extern 当作零 fallback。
        require(calls <= {"mm"}, f"需独立审查 extern：{calls}")
        ops = re.findall(r"torch\.ops\.((?:aten|npu)\.[\w.]+)\(", text)
        require(set(ops) <= {"aten.cat.default", "aten.view.default", "aten.matmul_backward.default"}, f"需独立审查 op 调用：{code}: {ops}")
        for op in ops:
            aten_calls[op] = aten_calls.get(op, 0) + 1
        extern.update(calls)
    log = (base / "stderr.log").read_text()
    require(not re.search(r"fall back to run on the CPU|fallback to CPU|graph break", log, re.I), f"异常回退日志：{base}")
    return {"generated_graphs": len(codes), "registered_npu_extern_ops": sorted(extern),
            "registered_npu_aten_calls": aten_calls,
            "lowering_difference": "cat由npu_cat显式注册为NPU extern；matmul_backward由后端fallback注册保留图内NPU调用（op-plugin有原生实现）；view为设备视图。不把这些算作Triton融合核，也不与CPU/图外fallback混淆。",
            "unexpected_cpu_or_graph_external_fallbacks": 0}


def check_arm(base, unit, mode):
    row = load(base / "result.json")
    execution = load(base / "execution.json")
    task, au, _ = WORKER.TARGETS[unit]
    for key, expected in {"task_id": task, "acceptance_unit_id": au,
            "backend": "triton_experimental", "pytorch_commit": WORKER.COMMIT,
            "worker_sha256": sha256(Path(WORKER.__file__)), "correctness": "passed",
            "numerical_execution": True, "graph_breaks": 0, "mode": mode,
            "pytorch_worktree_status": "", "input_spec": WORKER.INPUT_SPECS[unit]}.items():
        require(row.get(key) == expected, f"{base}: {key} 不匹配")
    require(execution["pid"] == row["pid"] and execution["return_code"] == 0 and not execution["timed_out"], "进程记录错误")
    state = row["state"]
    require((state["handler_calls"] == 0) if mode == "off" else
            (state["handler_calls"] > 0 and state["graph_changes"] > 0), "目标 ON/OFF 不符")
    sources = row["loaded_source_sha256"]
    require(any("triton_experimental" in p for p in sources), "缺少后端源码绑定")
    for path, digest in sources.items():
        require(Path(path).is_file() and sha256(Path(path)) == digest, f"源码漂移：{path}")
    return row, audit_code(base)


def implementation_sources(row):
    # loaded modules 同时含各臂生成的 cache module；这些应不同且单独归档，不是产品源码漂移。
    return {p: h for p,h in row["loaded_source_sha256"].items()
            if not Path(p).is_relative_to(Path("/home/z50063656/tmp"))}


def merged_sources(arms):
    union = {}
    for row in arms.values():
        sources = implementation_sources(row)
        require(any(p.endswith("/triton_experimental/device.py") for p in sources), "缺少实际设备后端源码")
        for path, digest in sources.items():
            require(path not in union or union[path] == digest, f"臂间源码改变：{path}")
            union[path] = digest
    return union


def archive(base, unit, phase):
    destination = ROOT / "issues" / CASES[unit] / "evidence" / phase / base.parent.name / base.name
    inventory = []
    for source in sorted(base.rglob("*")):
        relative = source.relative_to(base)
        if not source.is_file() or any(part.endswith("-cache") or part == "trace" for part in relative.parts):
            continue
        if source.name not in NAMES and not source.name.startswith("target-"):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(sha256(target) == sha256(source), f"拒绝覆盖旧证据：{target}")
        else:
            shutil.copy2(source, target)
        inventory.append({"path": str(target.relative_to(ROOT)), "sha256": sha256(target), "bytes": target.stat().st_size})
    return inventory


def functional(run, unit, reviewer):
    task, au, _ = WORKER.TARGETS[unit]
    progress = npu_contract_progress(task, au)
    require(progress and progress[1]["contract_complete"], "原社区合同尚未完整通过")
    gpu_path = ROOT / "results/current" / task / "gpu_reference_review.json"
    gpu = load(gpu_path)
    require(gpu["status"] == "gpu-contract-reviewed-awaiting-npu" and au in gpu["acceptance_units"], "GPU尚需补证")
    require(gpu["expected_pytorch_commit"] == WORKER.COMMIT and gpu["tests_skipped"] == 0, "GPU合同不符")
    arms, audits, inventory = {}, {}, []
    for mode in ("off", "on"):
        arms[mode], audits[mode] = check_arm(run / mode, unit, mode)
        inventory.extend(archive(run / mode, unit, "functional-" + run.parent.name))
    require(arms["off"]["pid"] != arms["on"]["pid"], "OFF/ON必须独立进程")
    sources = merged_sources(arms)
    path = ROOT / "results/current" / task / "functional" / f"{unit}.json"
    gate_path = path.parent.parent / "performance_gates" / path.name
    require(not path.exists() and not gate_path.exists(), "当前功能或gate已存在，不能覆盖；先归档复核")
    compact = {key: arms["on"][key] for key in ("task_id", "acceptance_unit_id", "backend", "pytorch_commit",
        "correctness", "numerical_execution", "target_rewrite", "graph_breaks", "product_disabled", "worker_sha256", "measurement_workload", "input_spec")}
    compact.update(schema_version="1.0", generated_at=datetime.now().astimezone().isoformat(),
        reviewed_at=datetime.now().astimezone().isoformat(), reviewer=reviewer, fallbacks=0,
        source_run_dir=str(run), source_files=sources,
        artifact_inventory=inventory, arms={k: {field: v[field] for field in ("pid", "mode", "state", "compile_ms", "physical_device")} for k,v in arms.items()},
        codegen_review=audits, gpu_reference=ref(gpu_path, path.parent),
        community_contract=ref(progress[0], path.parent),
        community_alignment={"status": "ALIGNED_WITH_BACKEND_LOWERING_DIFFERENCE", "aligned_scope": ["冻结社区输入、数值与正负例目标合同"],
            "divergent_scope": ["训练matmul_backward保留图内NPU extern，CUDA反向分解不同" if unit == "reorder-locality" else "cat使用后端显式注册的NPU extern lowering；不声称生成CUDA相同kernel"],
            "open_scope": ["没有声称模型级端到端收益"], "disposition": "只适配测试设备入口；保留 NPU 原生 pass 与数值实现。"})
    dump(path, compact)
    gate = {key: compact[key] for key in ("task_id", "acceptance_unit_id", "backend", "pytorch_commit", "correctness", "target_rewrite", "graph_breaks", "fallbacks", "product_disabled", "measurement_workload", "input_spec", "worker_sha256", "reviewed_at", "reviewer")}
    gate.update(gpu_reference=ref(gpu_path, gate_path.parent), target_functional=ref(path, gate_path.parent))
    dump(gate_path, gate)
    WORKER.read_gate(gate_path, unit, "npu")
    print(f"functional_review=passed task={task} unit={unit} gate={gate_path}")


def performance(run, unit):
    task, au, _ = WORKER.TARGETS[unit]
    gate = ROOT / "results/current" / task / "performance_gates" / f"{unit}.json"
    WORKER.read_gate(gate, unit, "npu")
    arms, audits, inventory = {}, {}, []
    for name in ("off1", "on1", "on2", "off2", "off3", "on3"):
        mode = name.rstrip("123")
        row, audits[name] = check_arm(run / name, unit, mode)
        require(row["phase"] == "benchmark" and row["gate_sha256"] == sha256(gate), "性能原件未绑定gate")
        require(all(len(row["samples"][k]) >= 100 for k in ("host_ms", "event_ms")), "不足100采样")
        arms[name] = row
        inventory.extend(archive(run / name, unit, "performance-" + run.parent.name))
    require(len({row["pid"] for row in arms.values()}) == 6, "六臂进程重复")
    sources = merged_sources(arms)
    timing, improvements, spread = {}, {}, {}
    for clock in ("host_ms", "event_ms"):
        timing[clock] = {}
        for mode in ("off", "on"):
            timing[clock][mode] = {q: statistics.median(row["timing"][clock][q] for name,row in arms.items() if name.startswith(mode)) for q in ("p50", "p99")}
            medians = [row["timing"][clock]["p50"] for name,row in arms.items() if name.startswith(mode)]
            spread[clock + "_" + mode] = max(medians)/min(medians)
        improvements[clock] = {q: (1-timing[clock]["on"][q]/timing[clock]["off"][q])*100 for q in ("p50","p99")}
    e, h = improvements["event_ms"], improvements["host_ms"]
    if max(spread.values()) > 1.5:
        verdict = "PERF_MIXED"
    elif min(e.values()) >= 5 and h["p99"] >= -5:
        verdict = "PERF_IMPROVED"
    elif max(e.values()) <= -5:
        verdict = "PERF_REGRESSED"
    elif min(e.values()) <= -5 or max(e.values()) >= 5:
        verdict = "PERF_MIXED"
    else:
        verdict = "PERF_NEUTRAL"
    path = ROOT / "results/current" / task / "performance_summary.json"
    summary = load(path) if path.exists() else {"schema_version":"1.0", "task_id":task, "backend":"triton_experimental", "acceptance_units":[]}
    require(not any(r["acceptance_unit_id"] == au for r in summary["acceptance_units"]), "不可覆盖旧性能结论")
    summary["generated_at"] = datetime.now().astimezone().isoformat()
    summary["status"] = "performance-disposition-recorded"
    summary["acceptance_units"].append({"acceptance_unit_id":au, "unit":unit, "performance_status":"measured", "verdict":verdict,
        "product_action":"保留现有默认配置；局部实测不自动推广为全模型默认开启",
        "measurement_workload":unit+"-community-shape", "timing":timing, "improvement_percent":improvements,
        "round_p50_spread_ratio":spread, "memory":{k:v["memory"] for k,v in arms.items()},
        "compile_ms":{k:v["compile_ms"] for k,v in arms.items()}, "codegen_review":audits,
        "gate":ref(gate,path.parent), "source_run_dir":str(run), "artifact_inventory":inventory,
        "measurement_contract":{"order":list(arms), "fresh_processes":6, "warmup":10, "runs_per_arm":100,
            "scope":"训练前向+反向（不含优化器）" if unit=="reorder-locality" else "社区子图端到端（不含编译）",
            "clocks":["同步 host", "NPU Event"], "model_e2e":False}})
    dump(path, summary)
    print(f"performance_review={verdict} task={task} unit={unit} improvement={improvements}")


def validate_archived():
    """提交时校验已归档正文及门禁绑定；不运行设备、不要求历史安装态仍存在。"""
    count = 0
    for unit in CASES:
        task, au, _ = WORKER.TARGETS[unit]
        path = ROOT / "results/current" / task / "functional" / f"{unit}.json"
        if not path.exists():
            continue
        record = load(path)
        require(record["backend"] == "triton_experimental" and record["acceptance_unit_id"] == au,
                "功能记录任务/后端错误")
        require(record["numerical_execution"] is True and record["correctness"] == "passed", "非设备正确性通过")
        gate_path = path.parent.parent / "performance_gates" / path.name
        WORKER.read_gate(gate_path, unit, "npu")
        items = list(record["artifact_inventory"])
        perf_path = path.parent.parent / "performance_summary.json"
        if perf_path.exists():
            for item in load(perf_path)["acceptance_units"]:
                if item["acceptance_unit_id"] == au:
                    require((perf_path.parent / item["gate"]["path"]).resolve() == gate_path.resolve()
                            and item["gate"]["sha256"] == sha256(gate_path), "性能未绑定当前功能gate")
                    items.extend(item["artifact_inventory"])
        require(items, "没有归档正文")
        for item in items:
            source = (ROOT / item["path"]).resolve()
            require(source.is_relative_to(ROOT) and source.is_file() and sha256(source) == item["sha256"], f"归档正文哈希失效：{source}")
        count += 1
    print(f"t087_t090_archive_validation=OK units={count} device_execution=false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="含 off/on 或 off1/on1/... 的单元目录")
    parser.add_argument("--unit", choices=CASES)
    parser.add_argument("--phase", choices=("functional","performance"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--reviewer", default="Codex：冻结社区合同与实际生成代码复核")
    args = parser.parse_args()
    if args.check:
        validate_archived()
        return
    if not args.run or not args.unit or not args.phase:
        parser.error("复核须提供 --run --unit --phase")
    if args.phase == "functional":
        functional(args.run.resolve(), args.unit, args.reviewer)
    else:
        performance(args.run.resolve(), args.unit)


if __name__ == "__main__":
    main()
