#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$(pwd -P)" != "/home/z50063656/tmp" ]]; then
    echo "错误：必须从 /home/z50063656/tmp 启动" >&2
    exit 2
fi

npu_id="${1:-0}"
mode="${2:-function}"
if [[ ! "${npu_id}" =~ ^[0-9]+$ ]]; then
    echo "用法：bash scripts/run_t080_npu_all.sh [NPU_ID] [function|performance|all]" >&2
    exit 2
fi
case "${mode}" in
    function|performance|all) ;;
    *)
        echo "用法：bash scripts/run_t080_npu_all.sh [NPU_ID] [function|performance|all]" >&2
        exit 2
        ;;
esac

if [[ "${mode}" == "function" || "${mode}" == "all" ]]; then
    echo "START T-080 NPU functional suite backend=triton_experimental"
    # Scatter 候选已因性能/显存回退进入默认产品门禁。默认复验最终
    # default-disabled/gate-disabled 双臂，不绕过门禁重跑整套候选正例。
    bash "${repo_root}/scripts/verify_t080_scatter_gate.sh" "${npu_id}"
    bash "${repo_root}/scripts/run_t080_softmax.sh" "${npu_id}"
    bash "${repo_root}/scripts/run_t080_constructors.sh" "${npu_id}"
    echo "END T-080 NPU functional suite"
fi

if [[ "${mode}" == "performance" || "${mode}" == "all" ]]; then
    echo "INFO prepare-softmax=PERF_EXEMPT reason=explicit-product-lowering-disable"
    bash "${repo_root}/scripts/run_npu_performance_task.sh" \
        --task T-080 \
        --validate-only
    echo "summary=${repo_root}/results/current/T-080/performance_summary.json"
    echo "product_gate=${repo_root}/results/current/T-080/product_gate_verification.json"
fi
