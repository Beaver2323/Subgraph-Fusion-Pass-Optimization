#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
unit="${1:-}"
npu_id="${2:-0}"
case "${unit}" in
    bmm-to-mm|cat-slice-cat|split-cat|cat-split) ;;
    *)
        echo "用法：bash scripts/run_t079_performance.sh {bmm-to-mm|cat-slice-cat|split-cat|cat-split} [NPU_ID]" >&2
        exit 2
        ;;
esac
if [[ ! "${npu_id}" =~ ^[0-9]+$ ]]; then
    echo "错误：NPU_ID 必须是非负整数" >&2
    exit 2
fi
if [[ "$(pwd -P)" != "/home/z50063656/tmp" ]]; then
    echo "错误：必须从 /home/z50063656/tmp 启动" >&2
    exit 2
fi

# shellcheck disable=SC1091
set +e
source /home/z50063656/Pass/activate_pass.sh >/tmp/t079-performance-activate.log
activate_status=$?
set -e
if ((activate_status != 0)); then
    echo "错误：Pass 环境激活失败，详见 /tmp/t079-performance-activate.log" >&2
    exit "${activate_status}"
fi
export ASCEND_RT_VISIBLE_DEVICES="${npu_id}"
export SET_NPU_DEVICE=0
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_COMPILE_THREADS=1

run_id="performance-$(date '+%Y%m%dT%H%M%S%z')"
run_dir="/home/z50063656/tmp/t079-performance-results/${unit}-${run_id}"
mkdir -p "${run_dir}/workers"
order=(off:1 on:1 on:2 off:2 off:3 on:3)
for item in "${order[@]}"; do
    mode="${item%%:*}"
    round="${item##*:}"
    worker_dir="${run_dir}/workers/${mode}${round}"
    mkdir -p "${worker_dir}"
    export TORCH_COMPILE_DEBUG=1
    export TORCH_COMPILE_DEBUG_DIR="${worker_dir}/debug"
    export TORCHINDUCTOR_CACHE_DIR="${worker_dir}/inductor-cache"
    export TRITON_CACHE_DIR="${worker_dir}/triton-cache"
    echo "START unit=${unit} mode=${mode} round=${round}"
    python "${repo_root}/runners/t079_performance_worker.py" \
        --unit "${unit}" \
        --mode "${mode}" \
        --round "${round}" \
        --warmup 10 \
        --runs 100 \
        --output "${worker_dir}/result.json" \
        >"${worker_dir}/stdout.log" \
        2>"${worker_dir}/stderr.log"
    echo "END unit=${unit} mode=${mode} round=${round}"
done
python "${repo_root}/runners/aggregate_t079_performance.py" --run-dir "${run_dir}"
ln -sfn "$(basename "${run_dir}")" \
    "/home/z50063656/tmp/t079-performance-results/latest-${unit}"
echo "artifacts=${run_dir}"
