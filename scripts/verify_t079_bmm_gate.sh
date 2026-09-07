#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$(pwd -P)" != "/home/z50063656/tmp" ]]; then
    echo "错误：必须从 /home/z50063656/tmp 启动" >&2
    exit 2
fi

npu_id="${1:-0}"
if [[ ! "${npu_id}" =~ ^[0-9]+$ ]]; then
    echo "用法：bash scripts/verify_t079_bmm_gate.sh [NPU_ID]" >&2
    exit 2
fi

set +e
# shellcheck disable=SC1091
source /home/z50063656/Pass/activate_pass.sh >/tmp/t079-bmm-gate-activate.log 2>&1
activate_status=$?
set -e
if ((activate_status != 0)); then
    echo "错误：Pass 环境激活失败，详见 /tmp/t079-bmm-gate-activate.log" >&2
    exit "${activate_status}"
fi

export ASCEND_RT_VISIBLE_DEVICES="${npu_id}"
export SET_NPU_DEVICE=0
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH="${repo_root}/runners/t078_source_overlay${PYTHONPATH:+:${PYTHONPATH}}"

run_id="bmm-gate-verify-$(date +%Y%m%dT%H%M%S%z)"
output_root="/home/z50063656/tmp/t079-bmm-gate-results/${run_id}"
mkdir -p "${output_root}"

for mode in default-disabled gate-disabled; do
    mode_dir="${output_root}/${mode}"
    mkdir -p "${mode_dir}"
    python "${repo_root}/runners/verify_t079_bmm_gate.py" \
        --mode "${mode}" \
        --output "${mode_dir}/result.json" \
        >"${mode_dir}/stdout.log" \
        2>"${mode_dir}/stderr.log"
    echo "${mode}=PASS"
done

ln -sfn "${output_root}" /home/z50063656/tmp/t079-bmm-gate-results/latest
echo "artifacts=${output_root}"
