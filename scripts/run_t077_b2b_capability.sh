#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
npu_id=""
scope="all"
validate_only=0

usage() {
    cat <<'EOF'
用法：
  bash scripts/run_t077_b2b_capability.sh --npu 0
  bash scripts/run_t077_b2b_capability.sh --npu 0 --scope functional
  bash scripts/run_t077_b2b_capability.sh --validate-only

说明：
  --scope functional|grid|all，默认 all。
  所有测试固定使用 triton_experimental；只做测试态 device/benchmarker capability 适配，
  不修改 torch_npu 产品源码。请从 /home/z50063656/tmp 启动实际测试。
EOF
}

while (($#)); do
    case "$1" in
        --npu) npu_id="${2:-}"; shift 2 ;;
        --scope) scope="${2:-}"; shift 2 ;;
        --validate-only) validate_only=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "错误：未知参数 $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${scope}" in functional|grid|all) ;; *) echo "错误：--scope 必须是 functional、grid 或 all" >&2; exit 2 ;; esac
set +e
# shellcheck disable=SC1091
source /home/z50063656/Pass/activate_pass.sh >/tmp/t077-b2b-activate.log 2>&1
activate_status=$?
set -e
if ((activate_status != 0)); then
    echo "错误：Pass 环境激活失败，详见 /tmp/t077-b2b-activate.log" >&2
    exit "${activate_status}"
fi
python -m py_compile \
    "${repo_root}/issues/REF-b2b-gemm-native/capability_probe.py" \
    "${repo_root}/runners/aggregate_b2b_capability.py"
if ((validate_only)); then
    echo "t077_b2b_capability_validation=OK scope=${scope} backend=triton_experimental"
    exit 0
fi
if [[ ! "${npu_id}" =~ ^[0-9]+$ ]]; then
    echo "错误：实际运行必须通过 --npu ID 指定单个物理 NPU" >&2
    exit 2
fi
if [[ "$(pwd -P)" != "/home/z50063656/tmp" ]]; then
    echo "错误：请从 /home/z50063656/tmp 启动本脚本" >&2
    exit 2
fi
if npu-smi info | awk -v id="${npu_id}" '$1 == "|" && $2 == id && $5 ~ /^[0-9]+$/ {found=1} END {exit found ? 0 : 1}'; then
    echo "错误：NPU ${npu_id} 已存在计算进程，拒绝启动" >&2
    exit 3
fi

run_id="capability-$(date '+%Y%m%dT%H%M%S%z')"
run_dir="/home/z50063656/tmp/t077-b2b-capability-results/${run_id}"
mkdir -p "${run_dir}"
export ASCEND_RT_VISIBLE_DEVICES="${npu_id}"
export SET_NPU_DEVICE=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
export TORCHINDUCTOR_COMPILE_THREADS=1
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python3.11/site-packages/torch_npu/lib:${LD_LIBRARY_PATH:-}"

if [[ "${scope}" == "functional" || "${scope}" == "all" ]]; then
    for mode in baseline candidate; do
        worker_dir="${run_dir}/functional/${mode}"
        mkdir -p "${worker_dir}"
        export TORCHINDUCTOR_CACHE_DIR="${worker_dir}/inductor-cache"
        export TRITON_CACHE_DIR="${worker_dir}/triton-cache"
        echo "START functional mode=${mode}"
        python "${repo_root}/issues/REF-b2b-gemm-native/capability_probe.py" \
            --mode "${mode}" --case all --artifact-dir "${worker_dir}" \
            >"${worker_dir}/stdout.log" 2>"${worker_dir}/stderr.log"
        echo "END functional mode=${mode}"
    done
fi

if [[ "${scope}" == "grid" || "${scope}" == "all" ]]; then
    points=(
        "trivial-left-positive 128 16" "trivial-left-positive 256 32"
        "trivial-left-positive 512 16" "trivial-left-positive 512 64"
        "left-gelu-positive 128 16" "left-gelu-positive 256 32"
        "left-gelu-positive 512 16" "left-gelu-positive 512 64"
        "gelu-mlp-performance-probe 128 16" "gelu-mlp-performance-probe 128 64"
        "gelu-mlp-performance-probe 256 32" "gelu-mlp-performance-probe 512 64"
    )
    for spec in "${points[@]}"; do
        read -r case_id matrix_m matrix_n <<<"${spec}"
        worker_dir="${run_dir}/community-grid/${case_id}-m${matrix_m}-n${matrix_n}"
        mkdir -p "${worker_dir}"
        export TORCHINDUCTOR_CACHE_DIR="${worker_dir}/inductor-cache"
        export TRITON_CACHE_DIR="${worker_dir}/triton-cache"
        echo "START grid case=${case_id} M=${matrix_m} N=${matrix_n}"
        python "${repo_root}/issues/REF-b2b-gemm-native/capability_probe.py" \
            --mode candidate --case "${case_id}" \
            --matrix-m "${matrix_m}" --matrix-n "${matrix_n}" \
            --allow-positive-unmatched --artifact-dir "${worker_dir}" \
            >"${worker_dir}/stdout.log" 2>"${worker_dir}/stderr.log"
        echo "END grid case=${case_id} M=${matrix_m} N=${matrix_n}"
    done
fi

python "${repo_root}/runners/aggregate_b2b_capability.py" --run-dir "${run_dir}"
ln -sfn "${run_id}" /home/z50063656/tmp/t077-b2b-capability-results/latest
echo "artifacts=${run_dir}"
