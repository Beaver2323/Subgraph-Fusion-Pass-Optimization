#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
unit="${1:-}"
npu_id="${2:-0}"
case "${unit}" in
    constructors) ;;
    scatter)
        echo "错误：Scatter 已因完整社区图性能/显存回退进入默认产品门禁；不得绕过门禁重跑候选 ON" >&2
        echo "正式结果：${repo_root}/results/current/T-080/performance_summary.json" >&2
        exit 4
        ;;
    *)
        echo "用法：bash scripts/run_t080_performance.sh constructors [NPU_ID]" >&2
        exit 2
        ;;
esac
if [[ "$(pwd -P)" != "/home/z50063656/tmp" ]]; then
    echo "错误：必须从 /home/z50063656/tmp 启动" >&2
    exit 2
fi

# shellcheck disable=SC1091
set +e
source /home/z50063656/Pass/activate_pass.sh >/tmp/t080-performance-activate.log
activate_status=$?
set -e
if ((activate_status != 0)); then
    echo "错误：Pass 环境激活失败" >&2
    exit "${activate_status}"
fi
export ASCEND_RT_VISIBLE_DEVICES="${npu_id}"
export SET_NPU_DEVICE=0
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH="${repo_root}/runners/t078_source_overlay${PYTHONPATH:+:${PYTHONPATH}}"

run_id="performance-$(date '+%Y%m%dT%H%M%S%z')"
run_dir="/home/z50063656/tmp/t080-performance-results/${unit}-${run_id}"
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
    python "${repo_root}/runners/t080_performance_worker.py" \
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
python "${repo_root}/runners/aggregate_t080_performance.py" --run-dir "${run_dir}"
ln -sfn "$(basename "${run_dir}")" \
    "/home/z50063656/tmp/t080-performance-results/latest-${unit}"
echo "artifacts=${run_dir}"
