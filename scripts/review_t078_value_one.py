#!/usr/bin/env python3
"""只读复核已人工检查的 FP16 value=1 GPU 补证及 NPU 修复原件，不执行还原代码。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import import_reference_text as handoff
from audit_history import reference


ROOT = Path(__file__).resolve().parents[1]
INPUT = "results/incoming/T-078/FP16-value1-text-handoff.json"
OUTPUT = "results/current/T-078/fp16_value_one_review_20260914.json"
CASE = "REF-addcdiv-fma-fp16-value1-derived"
COMMIT = "8e86e0a23e3679c2bf3406cf0837fcb6297a5d9b"
PAYLOAD = "799d5acb6e9a63e2b7db24d2a070fef3865d3735a8fc3d8b86c85dfae5d8d6cf"
NPU_ROOT = "issues/REF-addcdiv-fma-codegen-native/evidence/FP16精度修复/20260909-fixed"
NPU_RESULT_SHA = "2846bba5a4847f353a07dd15600b6ef6b51454beda86d2d6ca378fbc7fedaef6"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review(root=ROOT):
    payload = handoff.load_input(root / INPUT)
    run_id, files = handoff.validate_payload(payload)
    require(payload["payload_sha256"] == PAYLOAD, "不是本次人工审查的精确 GPU run，须重新审查")
    prefix = f"cases/{CASE}/"
    result = json.loads(files[prefix + "reference_result.json"])
    metadata = json.loads(files[prefix + "metadata.json"])
    execution = result["execution"]
    source = payload["environment"]["source"]
    require(source["actual_commit"] == source["expected_commit"] == COMMIT
            and source["working_tree_state"] == "clean", "GPU 源码基线不符")
    require(result["case"]["case_id"] == CASE and result["reference_valid"] is True
            and result["case"]["tracking_mode"] == "derived", "派生 case 身份不符")
    command = execution["command"]
    require(command[-4:] == ["--dtype", "float16", "--value", "1"], "dtype/value 执行合同不符")
    require(execution["tests_ran"] == 1 and execution["tests_skipped"] == 0
            and execution["return_code"] == 0, "测试没有真实通过")
    parsed = reference.parse_unittest_output(files[prefix + "stdout.log"].decode(),
                                            files[prefix + "stderr.log"].decode(), 0, 1)
    require(parsed["success"], "原始 unittest 日志重解析失败")
    require(files[prefix + "stdout.log"].decode().strip()
            == "dtype=float16 value=1.0 bitwise_equal=True max_abs_error=0.0", "缺少实际位级比较输出")
    require(metadata["expected_match"] is False, "value=1 不得当成 FMA 命中")
    for name in ("fx_before.txt", "fx_after.txt"):
        text = files[prefix + name].decode()
        require('f16[64, 64]' in text and "aten.div.Tensor" in text and "aten.add.Tensor" in text
                and "aten.mul" not in text and "aten.addcdiv" not in text, "GPU FX 不符合 div+add 合同")
    gpu_codes = [data for name, data in files.items() if name.startswith(prefix) and name.endswith("/output_code.py")]
    require(len(gpu_codes) == 1, "GPU 生成代码不唯一")
    gpu_code = gpu_codes[0].decode()
    require("tmp3 = (tmp1 / tmp2)" in gpu_code and "tmp4 = tmp0 + tmp3" in gpu_code
            and "tl.fma(" not in gpu_code and "div_rn(" not in gpu_code and ".run(" in gpu_code,
            "GPU 生成代码没有已审查的普通除加执行路径")

    inventory = json.loads((root / NPU_ROOT / "evidence_manifest.json").read_text())
    npu_evidence = []
    for item in inventory["files"]:
        if not item["path"].startswith("value-one/"):
            continue
        path = root / NPU_ROOT / item["path"]
        data = path.read_bytes()
        require(len(data) == item["bytes"] and digest(data) == item["sha256"], "NPU 归档原件哈希不符")
        npu_evidence.append({"path": str(path.relative_to(root)), "sha256": digest(data)})
    arm_bytes = (root / NPU_ROOT / "value-one/arm_result.json").read_bytes()
    require(digest(arm_bytes) == NPU_RESULT_SHA, "不是已审查的 NPU 修复运行")
    arm = json.loads(arm_bytes)
    require(arm["backend"] == "triton_experimental" and arm["torch_commit"] == COMMIT
            and arm["dtype"] == "float16" and arm["value"] == 1 and arm["shape"] == [64, 64]
            and arm["status"] == "passed" and arm["bitwise_equal_to_addcdiv_eager"] is True
            and arm["mismatch_count"] == 0 and arm["addcdiv_fma_fused"] == 0
            and arm["codegen_contract_valid"] is True, "NPU 修复合同不符")
    npu_code = (root / NPU_ROOT / "value-one/output_code.py").read_text()
    require("tmp4 = tmp3.to(tl.float16)" in npu_code and "tmp5 = tmp4.to(tl.float32)" in npu_code
            and "tmp6 = tmp0 + tmp5" in npu_code and "tl.fma(" not in npu_code, "NPU 舍入边界证据不符")
    return {
        "schema_version": "1.0", "generated_at": "2026-09-14T22:09:00+08:00",
        "task_id": "T-078", "acceptance_unit_id": "AU-post-grad-fuse-addcdiv-to-fma",
        "variant_id": "fp16-value1-bitwise-regression", "status": "passed",
        "backend": "triton_experimental", "dtype": "float16", "value": 1.0,
        "bitwise_equal": True, "max_abs_error": 0.0,
        "expected_addcdiv_fma_fused": 0, "actual_addcdiv_fma_fused": 0,
        "guard_preserved": True, "guard_scope": "value=1 不计 FMA pattern 命中；不是新增全 guard 覆盖",
        "gpu_reference": {"input": INPUT, "input_sha256": digest((root / INPUT).read_bytes()),
                          "payload_sha256": PAYLOAD, "run_id": run_id, "case_id": CASE,
                          "case_status": "valid-derived-reference", "suite_complete": False,
                          "tests_ran": 1, "tests_skipped": 0, "unittest_reparse": parsed,
                          "evidence_sha256": {name: digest(data) for name, data in files.items()},
                          "counter_evidence": "原派生入口内 counter=0 断言通过；FX/生成代码交叉核验。原输出未单独打印 counter。",
                          "runner_snapshot_in_original_run": False},
        "npu_evidence": npu_evidence,
        "community_alignment": "PARTIAL_ALIGNED",
        "alignment_boundary": "两端各自 compiled/eager 位级一致、普通 div+add、不命中 FMA；NPU 有显式 FP16 quotient 舍入，CUDA 本例没有该边界。不是跨设备逐位相等承诺。",
        "performance_status": "not-applicable-negative-neighbor",
        "new_device_execution": False, "product_modified_this_review": False,
        "counting_boundary": "仅关闭已登记 FP16 value=1 邻接缺口；不增加 acceptance-unit/性能分母，不把单例回传当完整 suite 重跑。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check-current", action="store_true")
    args = parser.parse_args()
    result = review()
    path = ROOT / OUTPUT
    if args.check_current:
        require(json.loads(path.read_text()) == result, "T-078 补证复核记录与原件不一致")
    if args.write:
        if path.exists():
            require(json.loads(path.read_text()) == result, "拒绝覆盖不同复核记录")
        else:
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("t078_value_one_review=passed gpu_tests=1 skip=0 npu_counter=0 new_device_execution=false")


if __name__ == "__main__":
    main()
