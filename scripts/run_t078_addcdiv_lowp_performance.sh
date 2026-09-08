#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
npu_id="${1:-}"

if [[ "$(pwd -P)" != "/home/z50063656/tmp" ]]; then
    echo "错误：必须从 /home/z50063656/tmp 启动" >&2
    exit 2
fi
if [[ ! "${npu_id}" =~ ^[0-9]+$ ]]; then
    echo "用法：bash scripts/run_t078_addcdiv_lowp_performance.sh NPU_ID" >&2
    exit 2
fi

run_id="$(date '+bf16-performance-%Y%m%dT%H%M%S%z')"
run_dir="/home/z50063656/tmp/t078-performance-results/${run_id}/addcdiv-bfloat16"
mkdir -p "${run_dir}/workers"

export ASCEND_RT_VISIBLE_DEVICES="${npu_id}"
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCH_COMPILE_DEBUG=1
export PYTHONPATH="${repo_root}/runners/t078_source_overlay${PYTHONPATH:+:${PYTHONPATH}}"

order=(off:1 on:1 on:2 off:2 off:3 on:3)
for item in "${order[@]}"; do
    mode="${item%%:*}"
    round="${item##*:}"
    worker_dir="${run_dir}/workers/${mode}${round}"
    mkdir -p "${worker_dir}"
    export TORCHINDUCTOR_CACHE_DIR="${worker_dir}/inductor-cache"
    export TRITON_CACHE_DIR="${worker_dir}/triton-cache"
    export TORCH_COMPILE_DEBUG_DIR="${worker_dir}/debug"
    echo "START dtype=bfloat16 mode=${mode} round=${round}"
    python "${repo_root}/runners/t078_performance_worker.py" \
        --unit addcdiv \
        --addcdiv-dtype bfloat16 \
        --mode "${mode}" \
        --round "${round}" \
        --warmup 10 \
        --runs 100 \
        --output "${worker_dir}/result.json" \
        >"${worker_dir}/stdout.log" \
        2>"${worker_dir}/stderr.log"
    echo "END dtype=bfloat16 mode=${mode} round=${round}"
done

python "${repo_root}/runners/aggregate_t078_performance.py" \
    --unit-dir "${run_dir}"

echo "artifacts=${run_dir}"
