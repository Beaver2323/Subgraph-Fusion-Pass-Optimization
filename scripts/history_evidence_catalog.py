"""历史补证目录与冻结源码审阅；只读，不运行 PyTorch，不补造历史元数据。"""

import ast
import hashlib
import json
from pathlib import Path
import subprocess


SOURCE_ROOT = Path("/home/z50063656/Pass/src/pytorch")
RAW_ROOT = Path("/home/z50063656/tmp")
# 明确选择最终证据及必要反例，不用 latest 或成功过滤隐藏历史失败。
NPU_RUNS = {
    "AU-post-grad-mm-plus-mm": ["t076-npu-results/REF-mm-plus-mm-native/adapter-20260902T025636+0800"],
    "AU-pad-mm-mm": [
        "t076-npu-results/REF-pad-mm-dynamic-m-native/adapter-20260902T034500+0800",
        "t076-npu-results/REF-pad-mm-original-aten-native/adapter-20260902T035000+0800",
        "t076-npu-results/REF-pad-mm-stride-native/adapter-20260902T041300+0800",
        "t076-npu-results/REF-pad-mm-exclusion-native/adapter-20260902T041500+0800",
    ],
    "AU-pad-mm-bmm": [
        "t076-npu-results/REF-pad-bmm-dynamic-batch-native/adapter-20260902T043000+0800",
        "t076-npu-results/REF-pad-bmm-static-fp16-native/adapter-20260902T045100+0800",
        "t076-npu-results/REF-pad-bmm-autocast-regression-native/adapter-20260902T045200+0800",
    ],
    "AU-pad-mm-addmm": [
        "t076-npu-results/REF-pad-addmm-dynamic-m-native/adapter-20260902T045600+0800",
        "t076-npu-results/REF-pad-addmm-bias-native/adapter-20260902T045800+0800",
    ],
    "AU-post-grad-addmm": [
        "t076-npu-results/REF-addmm-contract-native/adapter-20260902T051000+0800",
        "t076-npu-results/REF-addmm-symbolic-scalar-negative-native/adapter-20260902T050600+0800",
        "t076-p018-recheck/contract-retry",
        "t076-p018-recheck/symbolic",
    ],
    "AU-apply-gumbel-max-trick": ["t077-npu-results/gumbel-retry1"],
    "AU-b2b-gemm": ["t077-npu-results/b2b-retry1"],
    "AU-decompose-mem-bound-mm-decompose-bmm": ["t077-npu-results/decompose-bmm-clean-final3"],
    "AU-decompose-mem-bound-mm-decompose-mm": ["t077-npu-results/decompose-mm-isolate0", "t077-npu-results/decompose-mm-fixed-suite0"],
    "AU-decompose-mem-bound-mm-decompose-addmm": ["t077-npu-results/decompose-addmm-clean0"],
}

NUMERICAL_GAPS = {
    "REF-pad-mm-original-aten-native": "只检查 pattern counter 与 triton_tem_fused_mm 名称；ret 未与 eager 比较。",
    "REF-pad-mm-exclusion-native": "只检查 pad cache 数量与 exclude_pad 键；两次 compiled 输出均未与 eager 比较。",
}
INPUT_GRADIENT_GAPS = {
    "REF-decompose-bmm-native", "REF-decompose-mm-fp32-native", "REF-decompose-mm-mixed-native",
}


def review_source(case, commit, source_root, evidence):
    """人工审阅的语义分类绑定到精确 Git blob；不把检测到 assert 当作通用证明。"""
    if source_root is None:
        return {"status": "pending", "reason": "尚未指定冻结 PyTorch 源码库。"}
    file, node = case["source_test"].split("::")
    if ":" in file or file.startswith("/") or ".." in Path(file).parts:
        raise ValueError("社区源码路径不合法")
    result = subprocess.run(["git", "-C", str(source_root), "show", f"{commit}:{file}"], capture_output=True)
    if result.returncode:
        return {"status": "pending", "reason": "本机 Git 不含冻结社区源码对象。", "source": f"{commit}:{file}"}
    content = result.stdout
    key = f"git:{source_root}:{commit}:{file}"
    evidence[key] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    text = content.decode()
    cls_name, method = node.split(".")
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and (method == n.name or method.startswith(n.name + "_"))]
    fn = max(methods, key=lambda n: len(n.name))
    excerpt = ast.get_source_segment(text, fn)
    case_id = case["case_id"]
    helper_names = ["common"] if case_id == "REF-mm-plus-mm-native" else (
        ["compare_pred", "compare_parameters", "compare_gradients", "compare_dict_tensors"] if "REF-decompose-" in case_id else []
    )
    helpers = []
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name in helper_names:
            helpers.append({"name": n.name, "line": n.lineno, "sha256": hashlib.sha256(ast.get_source_segment(text, n).encode()).hexdigest()})
    numerical_gap = NUMERICAL_GAPS.get(case_id)
    gradient_gap = case_id in INPUT_GRADIENT_GAPS
    return {
        "status": "pending" if numerical_gap or gradient_gap else "passed",
        "reason": numerical_gap or (
            "forward 数值断言有效；backward 执行与 counter 有效，但 compare_gradients 只比较无参数 module 的空字典，未对齐输入梯度。"
            if gradient_gap else "已核对冻结方法及相关 helper 的数值/分布/stride/NaN 与目标断言范围；运行是否成功仍由原始日志单独判定。"
        ),
        "source": key, "method": f"{cls_name}.{fn.name}", "line": fn.lineno,
        "method_sha256": hashlib.sha256(excerpt.encode()).hexdigest(), "helpers": helpers,
        "numerical_oracle": "absent" if numerical_gap else "present-with-declared-scope",
        "input_gradient_oracle": "absent-empty-module-parameter-comparison" if gradient_gap else "not-claimed",
        "review_scope": "test_no_autocast 仅保证 dtype/非NaN；mm_plus_mm 全局 counter 的负例 1/2 不代表目标融合命中。",
    }


def runtime_inventory(unit_id, evidence, observe, check, combined):
    """复核可读运行结果/生成代码；缺安装态快照仍明确 pending。"""
    runs = []
    for relative in NPU_RUNS[unit_id]:
        root = RAW_ROOT / relative
        path = root / "adapter_result.json"
        if not path.is_file():
            path = root / "artifacts/adapter_result.json"
        checks = {"adapter_result": observe(path, evidence)}
        item = {"run_root": str(root), "checks": checks}
        if relative.endswith("decompose-mm-isolate0"):
            diagnostic = root / "fp32-m-threshold-negative-diagnostic.json"
            item["failure_diagnostic"] = observe(diagnostic, evidence)
            if diagnostic.is_file():
                item["failure_observation"] = json.loads(diagnostic.read_text())
            item["boundary"] = "失败在最终 adapter_result 落盘之前；保留原始梯度诊断，不把缺结果文件解释成缺成功基线。"
        if checks["adapter_result"]["status"] == "passed":
            result = json.loads(path.read_text())
            item["recorded_execution"] = {key: result.get(key) for key in (
                "tests_run", "successful", "failures", "errors", "skipped", "logical_cases", "product_gate_bypassed", "candidate_small_mm_guard_applied"
            )}
            item["recorded_backend"] = result.get("loaded_backend") or result.get("backend")
            item["backend_declaration"] = result.get("adapter_deviation", [])
            item["recorded_observations"] = result.get("observations", [result["observation"]] if "observation" in result else [])
            logs = [p for p in (root / "execution.log", root / "run.log") if p.is_file()]
            item["logs"] = [observe(p, evidence) for p in logs]
            codes = sorted(set(root.glob("*generated_code*.py")) | set((root / "artifacts").glob("*generated_code*.py")))
            if not codes:
                codes = sorted(root.glob("debug/**/output_code.py"))
            item["generated_code"] = [observe(p, evidence) for p in codes]
            checks["generated_path_available"] = check("passed" if codes else "pending", "已读取生成代码正文；捕获本次哈希不代表运行时已绑定同一代码哈希。", count=len(codes))
            checks["source_runtime_binding"] = check("pending", "旧 adapter 结果未绑定实际加载文件、完整三库 revision 与导入前 backend 生命周期；当前脚本内容或短版本号不能追溯证明当时安装态。")
        item["status"] = combined(checks.values())
        runs.append(item)
    return check(combined(check(run["status"], "") for run in runs),
                 "已盘点并读取指定原始运行；失败基线与候选成功分别保留，仍需运行态源码绑定。",
                 runs=runs, historical_files_rehashed_now=True, new_device_execution=False)


def capability_inventory(raw, evidence, observe, check, combined):
    root = Path(raw["root"])
    records = []
    counts = {"points": 0, "matched": 0, "rejected": 0, "template_selected": 0}
    for path in sorted(root.glob("*/capability_result.json")):
        record = {"file": observe(path, evidence)}
        payload = json.loads(path.read_text())
        valid = payload.get("backend") == "triton_experimental" and payload.get("successful") is True
        for item in payload.get("observations", []):
            counts["points"] += 1
            counts["matched"] += item.get("target_count", 0) > 0
            counts["rejected"] += item.get("target_count") == 0
            counts["template_selected"] += item.get("template_selected") is True
            valid = valid and item.get("correctness") == "passed"
            generated = Path(item["generated_code"])
            if not generated.resolve().is_relative_to(root.resolve()):
                raise ValueError("B2B generated code 路径越界")
            record["generated_code"] = observe(generated, evidence)
        record["functional_backend"] = check("passed" if valid else "failed", "逐点核对实际 backend、成功状态、正确性和模板选择。")
        for name in ("stdout.log", "stderr.log"):
            record[name] = observe(path.parent / name, evidence)
        records.append(record)
    expected = {"points": 12, "matched": 8, "rejected": 4, "template_selected": 0}
    return {
        "grid_recount": check("passed" if counts == expected else "pending", "独立重数原始能力点；这是12点能力评估，不是90点全网格性能验收。", actual=counts, expected=expected),
        "point_evidence": check(combined(value for row in records for value in row.values()) if records else "pending", "读取逐点原文和生成代码。", records=records),
        "runtime_binding": check("pending", "逐点记录有 backend/torch_git，但未绑定完整 torch_npu/Triton revision、运行时探针哈希及导入生命周期；不能据本次重算认证历史完整性能。"),
    }


def legacy_performance_inventory(unit_id, evidence, observe, check):
    root = Path("/home/z50063656/Pass/inductor_pass_npu_audit/results")
    if unit_id == "AU-post-grad-mm-plus-mm":
        paths = [root / f"p0_sweep_perf_mmplus_{suffix}/p0_gate_probe.json" for suffix in (
            "dtype_20260820", "shape_20260820", "layout_20260820", "dynamic_20260821", "dynamic_retest300_20260821"
        )]
    else:
        paths = [root / f"t058_addmm_perf_{arm}{i}_20260826/result.json" for arm, i in (("d", 1), ("e", 1), ("e", 2), ("d", 2), ("d", 3), ("e", 3))]
    records = []
    for path in paths:
        record = {"file": observe(path, evidence)}
        if path.is_file():
            payload = json.loads(path.read_text())
            rows = payload.get("results", [payload])
            record["worker_rows"] = len(rows)
            record["correctness_passed_rows"] = sum(row.get("status") == "compile-correct" or row.get("passed") is True for row in rows)
            record["recorded_backends"] = sorted(set(row.get("backend", "not-recorded") for row in rows))
            record["active_config_backends"] = sorted({snap.get("npu_backend", "not-recorded") for row in rows for p in row.get("observed_passes", []) for snap in p.get("active_config_snapshots", [])})
            record["input_contracts"] = [row.get("case_config", {"shape_mkn": row.get("shape_mkn"), "dtype": row.get("dtype"), "layout": row.get("layout")}) for row in rows]
            record["boundary"] = "运行期 active_config 优先于 compile 作用域外恢复的 effective_config；后者显示 default 不自动判为其他后端。"
        records.append(record)
    return {
        "legacy_files": check("passed" if all(row["file"]["status"] == "passed" for row in records) else "pending", "已找到旧仓原始 JSON 并绑定本次读取哈希。", records=records),
        "same_contract_reuse": check("pending", "性能 shape/dtype 与社区功能测例不是同一输入；可作代表 workload 证据，不能无条件迁移为社区 case 性能。旧记录缺逐样本 Event 及完整运行态溯源；addmm 是历史审计启用/P-018候选，非默认安装态。"),
    }
