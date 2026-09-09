#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
tracker_root="${TRACKER_ROOT:-${repo_root}}"
data_root="${PASS_GPU_DATA_ROOT:-/data/z50063656}"
gpu_ids=""
validate_only=0
case_ids=()
handoff_limit="${PASS_GPU_HANDOFF_SINGLE_FILE_BYTES:-98304}"
wait_gpu=0
wait_timeout=0
poll_interval=1
execution_mode=shared
min_free_memory=1024

usage() {
  cat <<'EOF'
用法：
  bash scripts/run_t085_gpu_all.sh --gpus 2,3
  bash scripts/run_t085_gpu_all.sh --gpus 2,3 --wait-gpu
  bash scripts/run_t085_gpu_all.sh --gpus 2,3 --case REF-pointless-cumsum-native
  bash scripts/run_t085_gpu_all.sh --validate-only

T-085完整suite包含真实2-rank NCCL例，完整运行必须指定两张不同的物理GPU。
--case可重复；即使只选单卡case，仍保持两张卡合同，避免误把子集当完整suite。
默认 shared，允许已有计算进程；--exclusive 要求两卡均无计算进程。
EOF
}

while (($#)); do
  case "$1" in
    --gpus)
      [[ $# -ge 2 ]] || { echo "错误：--gpus缺少值" >&2; exit 2; }
      gpu_ids="$2"
      shift 2
      ;;
    --case)
      [[ $# -ge 2 ]] || { echo "错误：--case缺少值" >&2; exit 2; }
      case_ids+=("$2")
      shift 2
      ;;
    --wait-gpu)
      wait_gpu=1
      shift
      ;;
    --exclusive)
      execution_mode=exclusive
      shift
      ;;
    --wait-timeout)
      [[ $# -ge 2 ]] || { echo "错误：--wait-timeout缺少值" >&2; exit 2; }
      wait_timeout="$2"
      shift 2
      ;;
    --poll-interval)
      [[ $# -ge 2 ]] || { echo "错误：--poll-interval缺少值" >&2; exit 2; }
      poll_interval="$2"
      shift 2
      ;;
    --min-free-memory-mib)
      [[ $# -ge 2 ]] || { echo "错误：--min-free-memory-mib缺少值" >&2; exit 2; }
      min_free_memory="$2"
      shift 2
      ;;
    --validate-only)
      validate_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "错误：未知参数 $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

pytorch_root="${PYTORCH_ROOT:-${data_root}/src/pytorch}"
work_dir="${PASS_TRACKER_WORK_DIR:-${data_root}/tmp}"
result_root="${data_root}/tmp/t085-reference-results"

validate_args=(
  --pytorch-root "${pytorch_root}"
  --validate-only
)
for case_id in "${case_ids[@]}"; do
  validate_args+=(--case "${case_id}")
done

cd "${work_dir}"
bash "${tracker_root}/scripts/run_t085_reference_all.sh" "${validate_args[@]}"

if ((validate_only)); then
  echo "gpu_task_validation=OK task=T-085 gpu_executed=0"
  exit 0
fi

if [[ ! "${gpu_ids}" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "错误：完整运行必须通过--gpus指定两张卡，例如2,3。" >&2
  exit 2
fi
IFS=, read -r gpu_a gpu_b <<<"${gpu_ids}"
if [[ "${gpu_a}" == "${gpu_b}" ]]; then
  echo "错误：两张物理GPU不能相同。" >&2
  exit 2
fi
gpu_a=$((10#${gpu_a}))
gpu_b=$((10#${gpu_b}))
if ((gpu_a > gpu_b)); then
  swap_gpu="${gpu_a}"
  gpu_a="${gpu_b}"
  gpu_b="${swap_gpu}"
fi

# shellcheck disable=SC1090
source "${tracker_root}/scripts/gpu_wait.sh"
tracker_gpu_validate_wait_options \
  "${wait_gpu}" "${wait_timeout}" "${poll_interval}" "${min_free_memory}"
tracker_gpu_acquire \
  "${gpu_a}" "${wait_gpu}" "${wait_timeout}" "${poll_interval}" \
  "${data_root}/tmp/tracker-gpu-locks" "${execution_mode}" "${min_free_memory}"
tracker_gpu_acquire \
  "${gpu_b}" "${wait_gpu}" "${wait_timeout}" "${poll_interval}" \
  "${data_root}/tmp/tracker-gpu-locks" "${execution_mode}" "${min_free_memory}"

venv_root="${PASS_GPU_VENV:-${data_root}/envs/PassGPURef}"
cuda_root="${CUDA_HOME:-${data_root}/cuda-12.6}"
for required in \
  "${venv_root}/bin/activate" \
  "${pytorch_root}/torch/__init__.py" \
  "${cuda_root}/bin/nvcc"; do
  [[ -e "${required}" ]] || { echo "错误：缺少${required}" >&2; exit 2; }
done

mkdir -p \
  "${work_dir}" \
  "${result_root}" \
  "${data_root}/pip-cache" \
  "${data_root}/cache" \
  "${data_root}/triton-cache" \
  "${data_root}/inductor-cache" \
  "${data_root}/torch-extensions"

export HOME="${data_root}"
export TMPDIR="${work_dir}"
export PIP_CACHE_DIR="${data_root}/pip-cache"
export XDG_CACHE_HOME="${data_root}/cache"
export TRITON_CACHE_DIR="${data_root}/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="${data_root}/inductor-cache"
export TORCH_EXTENSIONS_DIR="${data_root}/torch-extensions"

# shellcheck disable=SC1090
source "${venv_root}/bin/activate"
export PYTHON="${venv_root}/bin/python"
export CUDA_HOME="${cuda_root}"
export PATH="${CUDA_HOME}/bin:${venv_root}/bin:${PATH}"
export CUDNN_ROOT="${venv_root}/lib/python3.12/site-packages/nvidia/cudnn"
export CUDNN_INCLUDE_DIR="${CUDNN_ROOT}/include"
export LD_LIBRARY_PATH="${CUDNN_ROOT}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PASS_TRACKER_WORK_DIR="${work_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_a},${gpu_b}"
unset CUDA_COMPAT_DIR CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH

run_args=(
  --pytorch-root "${pytorch_root}"
  --output-root "${result_root}"
)
for case_id in "${case_ids[@]}"; do
  run_args+=(--case "${case_id}")
done

launcher_log="$(mktemp "${work_dir}/t085-gpu-launch.XXXXXX.log")"
set +e
bash "${tracker_root}/scripts/run_t085_reference_all.sh" "${run_args[@]}" 2>&1 | tee "${launcher_log}"
runner_status=${PIPESTATUS[0]}
set -e

run_dir="$(sed -n 's/^artifacts=//p' "${launcher_log}" | tail -n 1)"
if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
  echo "错误：未得到有效artifacts目录；日志=${launcher_log}" >&2
  exit "${runner_status:-1}"
fi

text_handoff="${run_dir}/text-handoff.json"
parts_dir="${run_dir}/text-handoff-parts"
set +e
"${PYTHON}" "${tracker_root}/scripts/export_reference_text.py" \
  --run-dir "${run_dir}" \
  --profile review \
  --compact \
  --allow-derived-output \
  --output "${text_handoff}" \
  --split-output-dir "${parts_dir}" \
  --auto-split-over-bytes "${handoff_limit}"
export_status=$?
set -e

"${PYTHON}" "${tracker_root}/scripts/publish_reference_latest.py" \
  --result-root "${result_root}" \
  --run-dir "${run_dir}" \
  --runner-status "${runner_status}" \
  --export-status "${export_status}"

echo "gpu_task_status=${runner_status}"
echo "run_dir=${run_dir}"
echo "latest_text_handoff=${result_root}/latest-text-handoff.json"
if [[ -f "${parts_dir}/manifest.json" ]]; then
  echo "handoff_upload_mode=split"
  echo "handoff_upload_input=${result_root}/latest/text-handoff-parts/manifest.json"
else
  echo "handoff_upload_mode=single-file"
  echo "handoff_upload_input=${result_root}/latest-text-handoff.json"
fi

((runner_status == 0)) || exit "${runner_status}"
exit "${export_status}"
