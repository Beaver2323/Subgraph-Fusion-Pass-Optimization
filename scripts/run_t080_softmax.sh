#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$(pwd -P)" != "/home/z50063656/tmp" ]]; then
    echo "错误：必须从 /home/z50063656/tmp 启动" >&2
    exit 2
fi
npu_id="${1:-0}"
if [[ ! "${npu_id}" =~ ^[0-9]+$ ]]; then
    echo "用法：bash scripts/run_t080_softmax.sh [NPU_ID]" >&2
    exit 2
fi

# shellcheck disable=SC1091
set +e
source /home/z50063656/Pass/activate_pass.sh >/tmp/t080-npu-activate.log
activate_status=$?
set -e
if ((activate_status != 0)); then
    echo "错误：Pass 环境激活失败，详见 /tmp/t080-npu-activate.log" >&2
    exit "${activate_status}"
fi
export ASCEND_RT_VISIBLE_DEVICES="${npu_id}"
export SET_NPU_DEVICE=0
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH="${repo_root}/runners/t078_source_overlay${PYTHONPATH:+:${PYTHONPATH}}"

cases=(
    REF-prepare-softmax-fast-math-native
    REF-prepare-softmax-signed-zero-native
    REF-prepare-softmax-community-perf-native
)

for case_id in "${cases[@]}"; do
    echo "START ${case_id} direct"
    python "${repo_root}/runners/run_t080_case.py" --case-id "${case_id}"
    echo "END ${case_id} direct"
    echo "START ${case_id} generic-guard-baseline"
    python "${repo_root}/runners/run_t080_case.py" --case-id "${case_id}" --adapter
    echo "END ${case_id} generic-guard-baseline"
    echo "START ${case_id} reviewed-minimal-probe"
    python "${repo_root}/runners/run_t080_case.py" \
        --case-id "${case_id}" \
        --adapter \
        --enable-softmax-probe
    echo "END ${case_id} reviewed-minimal-probe"
done
