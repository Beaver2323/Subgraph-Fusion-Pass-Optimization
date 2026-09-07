#!/usr/bin/env python3
"""只读复核历史证据，生成独立清单；不导入 torch，不改写历史 verdict。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparison = load_module("history_comparison", ROOT / "scripts/validate_comparison_data.py")
reference = load_module("history_reference", ROOT / "runners/reference_runner.py")
# 导入器只处理文本，不导入 torch；复用同一分片、压缩及 SHA256 校验。
sys.path.insert(0, str(ROOT / "scripts"))
import import_reference_text as handoff
import history_evidence_catalog as catalog
from export_history_logs import validate_logs


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(status, reason, **details):
    return {"status": status, "reason": reason, **details}


def combined(checks):
    statuses = {item["status"] for item in checks}
    if not statuses or not statuses <= {"passed", "pending", "failed", "exempt"}:
        raise ValueError("复核状态为空或包含未知值")
    return "failed" if "failed" in statuses else "pending" if "pending" in statuses else "passed"


def observe(path, evidence, expected=None):
    """不存在是待补，存在但哈希冲突是失败；只记录实际读到的文件哈希。"""
    if not path.is_file():
        return check("pending", "控制节点缺少原文件，不能由摘要/哈希推断内容。", path=str(path))
    actual = digest(path)
    evidence[str(path)] = {"sha256": actual, "bytes": path.stat().st_size}
    if expected is not None and actual != expected:
        return check("failed", "原始文件与已登记 SHA256 不一致。", path=str(path), expected=expected, actual=actual)
    return check("passed", "原文件可读；已校验登记哈希。" if expected else "原文件可读；本次记录内容哈希。", path=str(path))


def inside(root, relative):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"证据路径越界：{relative}")
    return candidate


def audit_gpu_case(run_dir, case, recorded, commit, evidence, text_files=None, source_root=None):
    checks = {}
    case_dir = inside(run_dir / "cases", case["case_id"])
    prefix = f"cases/{case['case_id']}/"
    implied_empty = {}

    def content(name):
        path = inside(case_dir, name)
        if path.is_file():
            return path.read_bytes()
        return (text_files or {}).get(prefix + name, implied_empty.get(name))

    def observe_case(name, expected=None):
        data = content(name)
        if data is None:
            return check("pending", "原件未在本机或已校验 handoff 正文中提供。", path=str(case_dir / name))
        actual = hashlib.sha256(data).hexdigest()
        location = str(case_dir / name) if (case_dir / name).is_file() else f"handoff:{prefix}{name}"
        evidence[location] = {"sha256": actual, "bytes": len(data)}
        return check("failed" if expected and actual != expected else "passed",
                     "由已绑定 inventory 的 bytes=0 与空文件 SHA256 确定空内容，无需传输。" if name in implied_empty else "实际读取原文并核对 SHA256。",
                     path=location, actual=actual, expected=expected, derived_empty=name in implied_empty)

    for name, key in (("reference_result.json", "reference_result_sha256"),
                      ("artifact_inventory.json", "artifact_inventory_sha256")):
        checks[name] = observe_case(name, recorded[key])
    if combined(checks.values()) != "passed":
        return {"case_id": case["case_id"], "status": combined(checks.values()), "checks": checks}
    result = json.loads(content("reference_result.json"))
    if result["source"]["actual_commit"] != commit or result["case"]["case_id"] != case["case_id"]:
        checks["identity"] = check("failed", "case 或 PyTorch revision 不一致。")
    inventory = json.loads(content("artifact_inventory.json"))
    indexed = {}
    for item in inventory:
        relative = item["path"]
        if relative in indexed:
            raise ValueError(f"inventory 重复项：{relative}")
        indexed[relative] = item
        if item.get("bytes") == 0 and item["sha256"] == hashlib.sha256(b"").hexdigest() and content(relative) is None:
            implied_empty[relative] = b""
        checks[f"inventory:{relative}"] = observe_case(relative, item["sha256"])
    for name in ("stdout.log", "stderr.log", "fx_before.txt", "fx_after.txt"):
        if name not in indexed:
            checks[f"required:{name}"] = check("pending", "缺少必需文件的已绑定 inventory 哈希。")
    logs_ready = all(checks.get(f"inventory:{name}", {}).get("status") == "passed" for name in ("stdout.log", "stderr.log"))
    if logs_ready:
        expected = len(case.get("direct_args") or [case["source_test"]])
        parsed = reference.parse_unittest_output(
            content("stdout.log").decode("utf-8"), content("stderr.log").decode("utf-8"),
            result["execution"]["return_code"], expected,
        )
        checks["unittest_reparse"] = check(
            "passed" if parsed["success"] else "failed", "使用当前 parser 重解析原始日志。", parsed=parsed,
        )
    else:
        checks["unittest_reparse"] = check("pending", "缺成功测例 stdout/stderr 正文；摘要中的 tests_ran/OK 不替代重解析。")
    # 这是更强日志门禁的复核，不把日志 OK 当作逐分支数值断言的源码审查。
    checks["assertion_semantics"] = catalog.review_source(case, commit, source_root, evidence)
    core_names = ("reference_result.json", "artifact_inventory.json", "fx_before.txt", "fx_after.txt")
    core = {name: observe_case(name, indexed.get(name, {}).get("sha256")) for name in core_names}
    checks["key_text_available"] = check(combined(core.values()), "关键结果、inventory、FX 正文完整性；不等于全 archive 或数值认证。", files=core)
    return {"case_id": case["case_id"], "status": combined(checks.values()), "checks": checks}


def load_history_handoff(repo_root, task_id, run_id, evidence):
    path = repo_root / "results/incoming" / task_id / "manifest.json"
    if not path.is_file():
        return {}, check("pending", "尚未提供带原文 handoff。")
    observe(path, evidence)
    outer = read_json(path)
    for part in outer.get("parts", []):
        observe(inside(path.parent, part["path"]), evidence)
    payload = handoff.load_input(path)
    actual_run, data = handoff.validate_payload(payload)
    if actual_run != run_id:
        raise ValueError(f"{task_id} handoff run_id 与正式 comparison 不一致")
    supplements = [p for p in (path.parent / "history-logs.json", path.parent / "history-logs/manifest.json") if p.is_file()]
    if len(supplements) > 1:
        raise ValueError("同一历史任务存在两份日志补证，请只保留一个明确入口")
    supplement_check = check("pending", "未提供 history-logs.json 或 history-logs/manifest.json。")
    if supplements:
        supplement = supplements[0]
        observe(supplement, evidence)
        for part in read_json(supplement).get("parts", []):
            observe(inside(supplement.parent, part["path"]), evidence)
        logs = validate_logs(handoff.load_input(supplement), task_id, run_id)
        indexed = {record["path"]: record for record in payload["evidence_files"]}
        for relative, value in logs.items():
            record = indexed.get(relative)
            if record is None or len(value) != record["bytes"] or hashlib.sha256(value).hexdigest() != record["sha256"]:
                raise ValueError(f"补证日志未绑定原 handoff inventory：{relative}")
            if relative in data and data[relative] != value:
                raise ValueError("补证日志与既有原文冲突")
            data[relative] = value
        supplement_check = check("passed", "日志补证逐项绑定原 handoff SHA256。", nonempty_logs=len(logs))
    return data, check("passed", "分片、整包和每份正文均验证；不执行反序列化代码。",
                       run_id=run_id, payload_sha256=payload["payload_sha256"],
                       restorable_text_files=len(data), omitted_files=len(payload["raw_text_transfer"]["omitted_files"]),
                       log_supplement=supplement_check)


def audit_workers(raw, unit_id, policy, evidence):
    """旧性能格式的保守检查：可验证的逐项检查，缺失元数据不从汇总补造。"""
    root = Path(raw["root"])
    checks = {"raw_summary": observe(root / "performance_summary.json", evidence, raw.get("summary_sha256"))}
    if checks["raw_summary"]["status"] != "passed":
        return checks
    names = policy["performance_round_order"]
    worker_root = root / "workers"
    if not worker_root.is_dir():
        checks["worker_selection"] = check("pending", "缺少原始 worker 目录。")
        return checks
    if {path.name for path in worker_root.iterdir() if path.is_dir()} != set(names):
        checks["worker_selection"] = check("pending", "OFF/ON worker 目录未齐备，不能认证完整六轮。")
    workers = []
    sample_checks = {}
    for name in names:
        path = root / "workers" / name / "result.json"
        checks[name] = observe(path, evidence, raw.get("worker_sha256", {}).get(name))
        if checks[name]["status"] != "passed":
            continue
        worker = read_json(path)
        workers.append(worker)
        for channel in ("host", "device_event"):
            timing = worker.get("timing", {}).get(channel, {})
            samples = timing.get("samples_ms")
            if samples is None:
                sample_checks[f"{name}:{channel}"] = check("pending", "原始 worker 未持久化逐样本时延，统计量无法逆推出样本。")
                continue
            valid_samples = isinstance(samples, list) and len(samples) == worker.get("runs") and bool(samples) and all(
                type(value) in (int, float) and math.isfinite(value) and value > 0 for value in samples
            )
            if valid_samples:
                values = sorted(samples)
                for label, fraction in (("p50_ms", .5), ("p99_ms", .99)):
                    pos = (len(values) - 1) * fraction
                    lo = int(pos)
                    actual = values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)
                    valid_samples = valid_samples and type(timing.get(label)) in (float, int) and math.isclose(timing[label], actual, abs_tol=1e-9)
            sample_checks[f"{name}:{channel}"] = check("passed" if valid_samples else "failed", "按线性插值重算逐轮 p50/p99，并校验原始样本数量及有限正值。")
        mode, round_id = name[:-1], int(name[-1])
        correctness = worker.get("correctness")
        if isinstance(correctness, dict):
            correctness = correctness.get("status")
        pattern = worker.get("pattern", {})
        valid = (
            worker.get("acceptance_unit_id") == unit_id and worker.get("mode") == mode
            and type(worker.get("round")) is int and worker["round"] == round_id
            and correctness == "passed"
            and type(worker.get("warmup")) is int and worker["warmup"] > 0
            and type(worker.get("runs")) is int and worker["runs"] > 0
            and pattern.get("actual_count") == pattern.get("expected_count") == (mode == "on")
        )
        checks[f"{name}:functional_round"] = check("passed" if valid else "failed", "核对唯一轮次、正整数迭代数、正确性与 OFF=0/ON=1 目标 counter。")
        env = dict(worker.get("environment", {}))
        if "torch_commit" not in env and "torch_git" in env:
            env["torch_commit"] = env["torch_git"]
        backend = env.get("backend")
        checks[f"{name}:backend"] = check(
            "pending" if backend is None else "passed" if backend == policy["required_npu_backend"] else "failed",
            "逐 worker 后端实录；缺失时不借用汇总声明。",
        )
        required = ("torch_commit", "torch_npu_commit", "triton_ascend_commit", "backend_selected_before_import", "process_start", "pid")
        missing = [key for key in required if key not in env]
        valid_meta = (
            all(isinstance(env.get(key), str) and len(env[key]) == 40 and all(c in "0123456789abcdef" for c in env[key])
                for key in ("torch_commit", "torch_npu_commit", "triton_ascend_commit"))
            and env.get("backend_selected_before_import") is True
            and type(env.get("pid")) is int and env["pid"] > 0
            and isinstance(env.get("process_start"), str)
        )
        checks[f"{name}:provenance"] = check(
            "pending" if missing else "passed" if valid_meta else "failed",
            "逐 worker 原始溯源字段核对；torch_git 按原值映射 torch_commit，generated_at 不是 process_start。",
            missing_fields=missing, recorded_generated_at=worker.get("generated_at"),
            available_revisions={key: env[key] for key in ("torch_commit", "torch_npu_commit", "triton_ascend_commit") if key in env},
        )
        for artifact_name in ("stdout.log", "stderr.log", "generated_code.py"):
            checks[f"{name}:{artifact_name}"] = observe(path.parent / artifact_name, evidence)
    if len(workers) == 6:
        contracts = [json.dumps({key: worker.get(key) for key in (
            "input", "input_shapes", "dynamic", "performance_case_source", "warmup", "runs"
        )}, sort_keys=True) for worker in workers]
        checks["same_input_method"] = check("passed" if len(set(contracts)) == 1 else "failed", "六轮输入与测例来源、迭代数一致；不等同于完整测量方法审核。")
        summary = read_json(root / "performance_summary.json")
        comparisons = {}
        for channel, prefix in (("host", "host"), ("device_event", "device")):
            for percentile in ("p50", "p99"):
                medians = {}
                for mode in ("off", "on"):
                    values = [w.get("timing", {}).get(channel, {}).get(f"{percentile}_ms") for w in workers if w["mode"] == mode]
                    if len(values) != 3 or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values):
                        comparisons[f"{prefix}_{percentile}"] = check("pending", "逐轮 percentile 数据未提供或无效。")
                        break
                    medians[mode] = statistics.median(values)
                if len(medians) == 2:
                    actual = (1 - medians["on"] / medians["off"]) * 100
                    recorded = summary.get("comparison", {}).get(f"{prefix}_{percentile}_improvement_percent")
                    comparisons[f"{prefix}_{percentile}"] = check(
                        "pending" if recorded is None else "passed" if math.isclose(recorded, actual, abs_tol=1e-9) else "failed",
                        "从原始六轮统计值独立重算三轮中位数与改善百分比。", recalculated_percent=actual, recorded_percent=recorded,
                    )
        checks["aggregate_recalculation"] = check(combined(comparisons.values()), "独立复算历史汇总，不据此证明逐样本分位数或同安装态。", metrics=comparisons)
    checks["measurement_review"] = check(combined(sample_checks.values()) if sample_checks else "pending", "逐样本统计复核；计时同步的执行版本和后端生命周期另需源码溯源，不从当前脚本回填。", samples=sample_checks,
                                         required_evidence=["逐样本 host/Event 时延", "当轮 worker/launcher 源码快照或哈希", "三库精确 revision 与实际加载源码", "导入前 backend、PID/启动时间及隔离顺序"])
    return checks


def build(repo_root=ROOT, gpu_runs=None):
    repo_root = repo_root.resolve()
    gpu_runs = gpu_runs or {}
    policy = read_json(repo_root / "schemas/audit_policy.json")
    if policy["history_tasks"] != ["T-076", "T-077"]:
        raise ValueError("本版历史审计范围必须完整包含 T-076/T-077，不得通过删任务获得通过")
    if datetime.fromisoformat(policy["updated_at"]).tzinfo is None:
        raise ValueError("审计规则时间戳必须含时区")
    if policy["required_npu_backend"] != comparison.REQUIRED_NPU_BACKEND:
        raise ValueError("审计规则与 NPU validator 后端不一致")
    evidence = {}
    tasks = []
    for task_id in policy["history_tasks"]:
        prefix = "" if task_id == "T-076" else task_id.lower().replace("-", "") + "_"
        plan_path = repo_root / f"upstream/{prefix}reference_plan.yaml"
        manifest_path = repo_root / f"upstream/{prefix}manifest.yaml"
        for path in (plan_path, manifest_path):
            observe(path, evidence)
        plan, manifest = read_json(plan_path), read_json(manifest_path)
        units = {item["acceptance_unit_id"]: item for item in manifest["acceptance_units"]}
        records, gpu_cases, performance = [], [], []
        npu_results = {}
        recorded_cases = {}
        seen = set()
        run_ids = set()
        for path in sorted((repo_root / "results/current").glob("*/comparison_result.json")):
            record = read_json(path)
            unit_id = record["acceptance_unit_id"]
            if unit_id not in units:
                continue
            if unit_id in seen:
                raise ValueError(f"重复历史 comparison：{unit_id}")
            seen.add(unit_id)
            result_path = inside(repo_root, record["npu"]["result_path"])
            observe(path, evidence)
            observe(result_path, evidence)
            result = read_json(result_path)
            npu_results[unit_id] = result
            comparison.validate_npu_result(result, result_path, repo_root, units[unit_id])
            comparison.validate_comparison(record, path, result, result_path, units[unit_id])
            for artifact in result["artifacts"]:
                if artifact["availability"] == "repository":
                    observe(inside(repo_root, artifact["path"]), evidence)
            records.append({"acceptance_unit_id": unit_id, "record_check": check("passed", "当前 validator 的结构、后端、正确性、comparison 哈希约束通过。"),
                            "runtime_reaudit": catalog.runtime_inventory(unit_id, evidence, observe, check, combined),
                            "original_verdict": record["final_verdict"], "original_repair_status": record["repair_status"],
                            "variant_count": len(result["variants"])})
            run_ids.add(record["reference"]["run_id"])
            for item in record["reference"]["cases"]:
                if item["case_id"] in recorded_cases and recorded_cases[item["case_id"]] != item:
                    raise ValueError("历史 reference case 哈希冲突")
                recorded_cases[item["case_id"]] = item
        if seen != set(units) or set(recorded_cases) != {case["case_id"] for case in plan["cases"]} or len(run_ids) != 1:
            raise ValueError(f"{task_id} 历史 unit/case/run 覆盖不完整")
        run_id = next(iter(run_ids))
        text_files, transport_check = load_history_handoff(repo_root, task_id, run_id, evidence)
        run_dir = gpu_runs.get(task_id, Path(f"/data/z50063656/tmp/{task_id.lower().replace('-', '')}-reference-results/{run_id}"))
        for case in plan["cases"]:
            gpu_cases.append(audit_gpu_case(run_dir, case, recorded_cases[case["case_id"]], plan["manifest"]["pytorch_commit"], evidence,
                                            text_files=text_files, source_root=catalog.SOURCE_ROOT))
        perf_path = repo_root / f"results/current/{task_id}/performance_summary.json"
        observe(perf_path, evidence)
        perf = read_json(perf_path)
        if perf.get("backend") != policy["required_npu_backend"]:
            raise ValueError(f"{task_id} 性能汇总后端不符")
        if Counter(item["acceptance_unit_id"] for item in perf["acceptance_units"]) != Counter({key: 1 for key in units}):
            raise ValueError(f"{task_id} 性能处置 unit 覆盖不符")
        for item in perf["acceptance_units"]:
            unit_id = item["acceptance_unit_id"]
            if item["performance_status"] == "not-required-explicitly-disabled":
                control = npu_results[unit_id]["npu_control"]
                if control["state"] != "disabled" or control["product_gate_bypassed"] or not control["source_control"]:
                    raise ValueError(f"{unit_id} 性能免测缺少对应显式关闭记录")
                performance.append({"acceptance_unit_id": unit_id, "status": "exempt", "reason": item["reason"],
                                    "boundary": "保留已登记的显式关闭处置，不补测、不代表性能通过；运行态来源另由 NPU 复核项跟踪。"})
                continue
            detail = item
            path_text = item.get("evidence", "")
            if path_text.startswith("report/"):
                # 旧报告可能附有中文分号后的交叉引用；只读明确文件部分。
                observe(inside(repo_root, path_text.split("；", 1)[0]), evidence)
            if path_text.endswith(".json"):
                path = inside(repo_root, path_text)
                observe(path, evidence)
                detail = read_json(path)
            raw = detail.get("raw_artifacts", {})
            if raw.get("root") and raw.get("summary_sha256"):
                checks = audit_workers(raw, unit_id, policy, evidence)
            elif unit_id == "AU-b2b-gemm" and raw.get("root"):
                checks = catalog.capability_inventory(raw, evidence, observe, check, combined)
            elif task_id == "T-076":
                checks = catalog.legacy_performance_inventory(unit_id, evidence, observe, check)
            else:
                checks = {"manual_provenance": check("pending", "旧报告/能力网格证据需人工建立 revision、输入、后端和测量合同映射；不由旧 verdict 自动通过。", source=path_text)}
            performance.append({"acceptance_unit_id": unit_id, "status": combined(checks.values()),
                                "original_verdict": item["verdict"], "checks": checks})
        tasks.append({"task_id": task_id, "gpu_run_id": run_id, "gpu_handoff": transport_check, "gpu_cases": gpu_cases,
                      "npu_records": records, "performance_units": performance})
    files = [repo_root / "schemas/audit_policy.json"]
    for directory in ("scripts", "runners", "tests", "schemas", "upstream"):
        files.extend(path for path in (repo_root / directory).glob("*") if path.is_file() and path.suffix in {".py", ".sh", ".json", ".yaml"})
    validator_files = {str(path.relative_to(repo_root)): digest(path) for path in sorted(set(files))}
    code_hash = hashlib.sha256(json.dumps(validator_files, sort_keys=True).encode()).hexdigest()
    statuses = [item["status"] for task in tasks for section in ("gpu_cases", "performance_units") for item in task[section]]
    statuses.extend(item["runtime_reaudit"]["status"] for task in tasks for item in task["npu_records"])
    components = {
        "gpu_key_text_cases_passed": sum(c["checks"].get("key_text_available", {}).get("status") == "passed" for t in tasks for c in t["gpu_cases"]),
        "gpu_log_reparse_cases_passed": sum(c["checks"].get("unittest_reparse", {}).get("status") == "passed" for t in tasks for c in t["gpu_cases"]),
        "gpu_source_semantics_passed": sum(c["checks"].get("assertion_semantics", {}).get("status") == "passed" for t in tasks for c in t["gpu_cases"]),
        "gpu_missing_nonempty_logs": sum(v["status"] == "pending" for t in tasks for c in t["gpu_cases"] for k, v in c["checks"].items() if k in ("inventory:stdout.log", "inventory:stderr.log")),
        "gpu_empty_logs_verified_without_transfer": sum(v.get("derived_empty", False) for t in tasks for c in t["gpu_cases"] for k, v in c["checks"].items() if k in ("inventory:stdout.log", "inventory:stderr.log")),
        "npu_selected_raw_result_files_passed": sum(run["checks"]["adapter_result"]["status"] == "passed" for t in tasks for n in t["npu_records"] for run in n["runtime_reaudit"]["runs"]),
        "performance_aggregate_recalculations_passed": sum(p.get("checks", {}).get("aggregate_recalculation", {}).get("status") == "passed" for t in tasks for p in t["performance_units"]),
    }
    git = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, capture_output=True, check=True)
    return {"schema_version": "1.0", "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "rule_version": policy["rule_version"], "validator_version": policy["validator_version"],
            "validator_fingerprint_sha256": code_hash, "validator_files": validator_files,
            "repository_head": git.stdout.strip(), "source_files": evidence, "tasks": tasks,
            "component_counts": components,
            "status": combined(check(status, "") for status in statuses), "status_counts": dict(Counter(statuses)),
            "npu_record_checks_passed": sum(len(task["npu_records"]) for task in tasks),
            "boundary": "旧 verdict 原样保留。通过的记录校验不等于重新验证所有 GPU/NPU/性能证据；本清单不得用于扩大冻结分母。T-067 属旧 feature-family 格式，未自动迁入本清单。"}


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_gpu_runs(values):
    result = {}
    for value in values:
        task_id, separator, path = value.partition("=")
        if not separator or task_id not in {"T-076", "T-077"} or task_id in result or not path:
            raise ValueError("--gpu-run 应为 T-076=/原始/run/路径 或 T-077=/原始/run/路径，不得重复")
        result[task_id] = Path(path).resolve()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-run", action="append", default=[], help="原始 run 已复制到本机时指定 TASK=/path；不连接 GPU")
    parser.add_argument("--output", type=Path, help="只新建，拒绝覆盖；省略时输出 JSON")
    args = parser.parse_args()
    try:
        payload = build(gpu_runs=parse_gpu_runs(args.gpu_run))
        if args.output:
            output = args.output.resolve()
            if output.is_relative_to(ROOT / "results/current") or any(output.is_relative_to(path) for path in parse_gpu_runs(args.gpu_run).values()):
                raise ValueError("复核清单不得写入原始证据目录")
            write_new(args.output, payload)
            print(f"history_audit={args.output} status={payload['status']}")
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2 if payload["status"] == "failed" else 3 if payload["status"] == "pending" else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"历史复核失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
