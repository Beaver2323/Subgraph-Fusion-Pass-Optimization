#!/usr/bin/env python3
"""复核 T-081～T-083 六臂性能原件并生成正式任务汇总。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/home/z50063656/tmp")
ORDER = ("off1", "on1", "on2", "off2", "off3", "on3")
TASK_UNITS = {
    "T-081": ("constant-fold", "convert"),
    "T-082": ("permute", "view"),
    "T-083": ("all-gather", "all-reduce", "reduce-scatter"),
}
UNIT_IDS = {
    "constant-fold": "AU-joint-graph-constant-fold-uniform-value",
    "convert": "AU-joint-graph-pointless-convert",
    "permute": "AU-joint-graph-pointless-permute-pair",
    "view": "AU-joint-graph-pointless-view-pair",
    "all-gather": "AU-post-grad-bucket-all-gathers",
    "all-reduce": "AU-post-grad-bucket-all-reduce",
    "reduce-scatter": "AU-post-grad-bucket-reduce-scatters",
}
DETAILS = {
    "constant-fold": {
        "intent": "把动态图中的均匀full常量传播到add/sub/mul，并在恒等情形删除设备kernel。",
        "source": ["torch/_inductor/fx_passes/joint_graph.py:501", "torch/_inductor/fx_passes/joint_graph.py:727"],
        "gpu_behavior": "A100原生动态形状与自指shape社区case通过；旧观察器显示缓存问题已单独修复。",
        "npu_behavior": "Ascend910B2上OFF不进入handler；ON命中并改图，最终由1个Triton kernel变为直接返回输入。",
    },
    "convert": {
        "intent": "在浮点精度与舍入保护允许时，把连续两次dtype转换收敛为一次最终转换。",
        "source": ["torch/_inductor/fx_passes/joint_graph.py:830", "torch/_inductor/fx_passes/joint_graph.py:842"],
        "gpu_behavior": "A100执行社区五组结构合同，正负转换链均按预期保留或折叠。",
        "npu_behavior": "Ascend910B2数值与目标改图通过；OFF/ON最终均为单个融合转换Triton kernel。",
    },
    "permute": {
        "intent": "两次互逆permute复合为恒等映射，直接返回原tensor并保持alias/stride。",
        "source": ["torch/_inductor/fx_passes/joint_graph.py:948", "torch/_inductor/fx_passes/joint_graph.py:957"],
        "gpu_behavior": "A100原生2D/3D互逆与中间用户保护结构case通过。",
        "npu_behavior": "Ascend910B2命中、数值、alias和stride通过；OFF/ON下游最终都消成零kernel恒等代码。",
    },
    "view": {
        "intent": "当第二个view恢复原shape时删除view-pair，直接复用原tensor。",
        "source": ["torch/_inductor/fx_passes/joint_graph.py:927", "torch/_inductor/fx_passes/joint_graph.py:936"],
        "gpu_behavior": "A100原生静态结构与动态数值社区case通过。",
        "npu_behavior": "Ascend910B2命中、数值、alias和stride通过；OFF/ON下游最终都消成零kernel恒等代码。",
    },
    "all-gather": {
        "intent": "把多个独立all-gather打包为一次collective，再切分恢复各输出。",
        "source": ["torch/_inductor/fx_passes/bucketing.py:394", "torch/_inductor/fx_passes/post_grad.py:380"],
        "gpu_behavior": "A100原生world_size=1结构合同通过，只作为reference，不授权跨rank性能。",
        "npu_behavior": "Ascend910B2真实HCCL双rank数值通过，collective 3→1；打包/拆包引入额外Triton kernel。",
    },
    "all-reduce": {
        "intent": "把多个独立all-reduce拼接后合并成一次collective，再按原shape切分。",
        "source": ["torch/_inductor/fx_passes/bucketing.py:759", "torch/_inductor/fx_passes/post_grad.py:366"],
        "gpu_behavior": "A100原生单rank eager/compiled数值与2→1结构通过。",
        "npu_behavior": "Ascend910B2真实HCCL双rank数值通过，collective 2→1，并保留并行mm结果。",
    },
    "reduce-scatter": {
        "intent": "把多个独立reduce-scatter输入打包成一次collective，再切分为原输出。",
        "source": ["torch/_inductor/fx_passes/bucketing.py:412", "torch/_inductor/fx_passes/post_grad.py:348"],
        "gpu_behavior": "A100原生单rank eager/compiled数值与2→1结构通过。",
        "npu_behavior": "Ascend910B2真实HCCL双rank数值通过，collective 2→1；打包与切分增加中间显存。",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def codegen_record(run_dir: Path, mode: str) -> dict:
    arm = "off1" if mode == "off" else "on1"
    files = sorted((run_dir / arm / "rank-0").rglob("output_code.py"))
    # compile-debug 与 inductor-cache 可能保存同一 output_code.py 的副本；
    # 按内容哈希去重，避免把一个 kernel/collective 统计成两个。
    unique = {}
    for path in files:
        digest = sha256(path)
        unique.setdefault(digest, path.read_text(encoding="utf-8", errors="replace"))
    texts = list(unique.values())
    return {
        "output_code_files": len(unique),
        "output_code_sha256": sorted(unique),
        "triton_kernels": sum(text.count("= async_compile.triton(") for text in texts),
        "all_gather_calls": sum(text.count(" = torch.ops._c10d_functional.all_gather_into_tensor") for text in texts),
        "all_reduce_calls": sum(text.count(" = torch.ops._c10d_functional.all_reduce") for text in texts),
        "reduce_scatter_calls": sum(text.count(" = torch.ops._c10d_functional.reduce_scatter_tensor") for text in texts),
    }


def classify(timing: dict) -> tuple[str, str, str]:
    p50 = timing["event_ms"]["p50"]["improvement_percent"]
    p99 = timing["event_ms"]["p99"]["improvement_percent"]
    if p50 >= 5 and p99 >= 5:
        return "PERF_IMPROVED", "measured-improved", "保留现状；若默认关闭，仅作为后续模型级启用候选"
    if p50 <= -5 and p99 <= -5:
        return "PERF_REGRESSED", "measured-regressed", "保持默认关闭或增加产品门禁"
    if (p50 >= 5 and p99 <= -5) or (p50 <= -5 and p99 >= 5):
        return "PERF_MIXED", "measured-mixed", "不据此启用；保持默认关闭并扩大真实模型复核"
    return "PERF_NEUTRAL", "measured-neutral", "不因微小差异改变产品配置"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_UNITS, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="UNIT=/absolute/run_dir；每个任务单元各传一次",
    )
    args = parser.parse_args()
    require(Path.cwd().resolve() == WORK, f"必须从 {WORK} 启动")
    runs = {}
    for item in args.run:
        unit, separator, raw_path = item.partition("=")
        require(bool(separator) and unit not in runs, f"无效或重复--run：{item}")
        runs[unit] = Path(raw_path).resolve()
    require(set(runs) == set(TASK_UNITS[args.task]), "--run必须精确覆盖任务全部单元")

    units = []
    verdict_counts = {}
    for unit in TASK_UNITS[args.task]:
        run_dir = runs[unit]
        summary_path = run_dir / "performance_summary.json"
        require(summary_path.is_file(), f"缺少性能summary：{summary_path}")
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
        require(raw.get("task_id") == args.task and raw.get("unit") == unit, "任务/单元不匹配")
        require(raw.get("status") == "measured", "原始性能未完成")
        require(raw.get("verdict") == "PENDING_SOURCE_AND_KERNEL_REVIEW", "原始结果已被非本流程改判")
        pids = set()
        gate_hashes = set()
        backends = set()
        for arm in ORDER:
            arm_summary = run_dir / arm / "arm_summary.json"
            require(arm_summary.is_file(), f"缺少六臂结果：{arm_summary}")
            parsed_arm = json.loads(arm_summary.read_text(encoding="utf-8"))
            for rank in range(parsed_arm["world_size"]):
                record_path = run_dir / arm / f"rank-{rank}/worker_result.json"
                require(record_path.is_file(), f"缺少rank原件：{record_path}")
                record = json.loads(record_path.read_text(encoding="utf-8"))
                require(record.get("correctness") == "passed", "性能臂正确性失败")
                require(record.get("backend") == "triton_experimental", "性能臂后端错误")
                require(record.get("pytorch_worktree_status") == "", "PyTorch工作树不干净")
                require(record["pid"] not in pids, "六臂复用了进程")
                pids.add(record["pid"])
                gate_hashes.add(record["gate_sha256"])
                backends.add(record["backend"])
        require(len(gate_hashes) == 1 and backends == {"triton_experimental"}, "六臂门禁/后端不一致")
        verdict, performance_status, action = classify(raw["timing"])
        # 上游三类collective默认均为none；单一目标子图不能直接授权产品默认开启。
        if unit in {"all-gather", "all-reduce", "reduce-scatter"}:
            if verdict == "PERF_IMPROVED":
                action = "保持上游默认none；作为同shape模型级复核后启用的候选"
            else:
                action = "保持上游默认none"
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        units.append(
            {
                "acceptance_unit_id": UNIT_IDS[unit],
                "unit": unit,
                "pattern_explanation": DETAILS[unit],
                "performance_status": performance_status,
                "verdict": verdict,
                "product_action": action,
                "timing": raw["timing"],
                "memory": raw["memory"],
                "codegen": {
                    "off": codegen_record(run_dir, "off"),
                    "on": codegen_record(run_dir, "on"),
                },
                "raw_summary": str(summary_path),
                "raw_summary_sha256": sha256(summary_path),
                "gate_sha256": next(iter(gate_hashes)),
            }
        )

    generated_at = datetime.now().astimezone().isoformat()
    result = {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "task_id": args.task,
        "status": "performance-disposition-complete",
        "backend": "triton_experimental",
        "measurement_contract": {
            "order": "OFF1-ON1-ON2-OFF2-OFF3-ON3",
            "process_isolation": "fresh-process-per-arm-and-round",
            "tracker_concurrency": "exclusive-global-lock",
            "warmup": 10,
            "runs": 100,
            "timing": "同步NPU Event与host wall clock p50/p99",
            "scope": "社区功能正例派生的目标子图一次compiled调用；不是完整模型端到端",
        },
        "completion": {
            "acceptance_units": len(units),
            "measured_units": len(units),
            "pending_units": 0,
        },
        "verdict_counts": verdict_counts,
        "acceptance_units": units,
        "limitations": [
            "社区没有这些单元的独立pass性能benchmark；本轮复用社区功能图、shape和dtype后增加OFF/ON计时。",
            "collective默认none；目标子图收益不能直接替代真实模型、拓扑和消息规模复核。",
            "失败的并发预跑及NPU 0一次HDC/TSD启动超时均未纳入本正式汇总。",
        ],
    }
    output = ROOT / "results/current" / args.task / "performance_summary.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"performance_review=OK task={args.task} units={len(units)}")
    print(f"summary={output}")


if __name__ == "__main__":
    main()
