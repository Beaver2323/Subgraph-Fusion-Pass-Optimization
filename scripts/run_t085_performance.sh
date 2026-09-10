#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
work_dir="/home/z50063656/tmp"
unit=""
npu_ids=""
phase="benchmark"
validate_only=0

usage() {
  cat <<'EOF'
用法：
  bash scripts/run_t085_performance.sh --validate-only
  cd /home/z50063656/tmp
  bash /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t085_performance.sh \
    --unit pointless-cumsum --npu 0
  bash /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t085_performance.sh \
    --unit overlap-device-put --npu 0,1
  bash /home/z50063656/Pass/Subgraph-Fusion-Pass-Optimization/scripts/run_t085_performance.sh \
    --unit partitioned-scatter --npu 0 --phase functional
EOF
}

while (($#)); do
  case "$1" in
    --unit) unit="${2:-}"; shift 2 ;;
    --npu) npu_ids="${2:-}"; shift 2 ;;
    --phase) phase="${2:-}"; shift 2 ;;
    --validate-only) validate_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "错误：未知参数$1" >&2; usage >&2; exit 2 ;;
  esac
done

if ((validate_only)); then
  exec python "${repo_root}/scripts/run_t085_performance.py" --validate-only
fi
[[ "$(pwd -P)" == "${work_dir}" ]] || { echo "错误：必须从${work_dir}启动" >&2; exit 2; }
[[ "${phase}" == "functional" || "${phase}" == "benchmark" ]] || {
  echo "错误：--phase只能是functional或benchmark" >&2
  exit 2
}

case "${unit}" in
  pointless-cumsum|partitioned-scatter)
    [[ "${npu_ids}" =~ ^[0-9]+$ ]] || { echo "错误：pointless-cumsum需要单张NPU，例如--npu 0" >&2; exit 2; }
    ;;
  overlap-device-put)
    [[ "${npu_ids}" =~ ^[0-9]+,[0-9]+$ ]] || { echo "错误：overlap需要两张NPU，例如--npu 0,1" >&2; exit 2; }
    [[ "${npu_ids%%,*}" != "${npu_ids##*,}" ]] || { echo "错误：两张NPU不能相同" >&2; exit 2; }
    ;;
  *) echo "错误：必须指定可运行--unit" >&2; usage >&2; exit 2 ;;
esac

exec 9>"${work_dir}/pass-tracker-npu-performance.lock"
if ! flock -n 9; then
  echo "错误：已有tracker NPU性能任务运行" >&2
  exit 3
fi

set +e
# shellcheck disable=SC1091
source /home/z50063656/Pass/activate_pass.sh >/tmp/t085-performance-activate.log
set -e
if [[ "${CONDA_DEFAULT_ENV:-}" != "Pass" ]] || \
   [[ "$(command -v python)" != "/home/z50063656/envs/Pass/bin/python" ]]; then
  echo "错误：Pass环境未正确激活，详见/tmp/t085-performance-activate.log" >&2
  exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES="${npu_ids}"
export SET_NPU_DEVICE=0
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export PASS_TRACKER_WORK_DIR="${work_dir}"
export PYTHONPATH="${repo_root}/runners/t078_source_overlay${PYTHONPATH:+:${PYTHONPATH}}"

command=(
  python
  "${repo_root}/scripts/run_t085_performance.py"
  --unit "${unit}"
  --device npu
  --phase "${phase}"
)
if [[ "${phase}" == "benchmark" ]]; then
  gate="${repo_root}/results/current/T-085/performance_gates/${unit}.json"
  [[ -f "${gate}" ]] || {
    echo "错误：缺少人工签署gate ${gate}" >&2
    exit 2
  }
  command+=(--gate "${gate}")
fi

exec "${command[@]}"
