#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
tracker_root="${TRACKER_ROOT:-${repo_root}}"
data_root="${PASS_GPU_DATA_ROOT:-/data/z50063656}"
task_id=""
gpu_id="${CUDA_VISIBLE_DEVICES:-}"
case_id=""
validate_only=0

usage() {
    cat <<'EOF'
用法：
  bash scripts/run_gpu_reference_task.sh --task T-078 --gpu 2
  bash scripts/run_gpu_reference_task.sh --task T-079 --gpu 2
  bash scripts/run_gpu_reference_task.sh --task T-080 --gpu 2
  bash scripts/run_gpu_reference_task.sh --task T-076 --gpu 2 --case REF-mm-plus-mm-native
  bash scripts/run_gpu_reference_task.sh --task T-078 --validate-only

参数：
  --task T-076|T-077|T-078|T-079|T-080   必填，选择已登记的 GPU reference 任务
  --gpu ID             实际运行时使用的物理 GPU 编号；也可预先设置 CUDA_VISIBLE_DEVICES
  --case CASE_ID       可选，只运行指定 case
  --validate-only      只做零设备静态校验
  -h, --help           显示帮助

路径可通过环境变量覆盖：
  PASS_GPU_DATA_ROOT、TRACKER_ROOT、PYTORCH_ROOT、PASS_GPU_VENV、CUDA_HOME
EOF
}

while (($#)); do
    case "$1" in
        --task)
            task_id="${2:-}"
            shift 2
            ;;
        --gpu)
            gpu_id="${2:-}"
            shift 2
            ;;
        --case)
            case_id="${2:-}"
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

task_id="${task_id^^}"
case "${task_id}" in
    T-076|T076)
        task_id="T-076"
        task_runner="${tracker_root}/scripts/run_reference_all.sh"
        result_root="${data_root}/tmp/t076-reference-results"
        ;;
    T-077|T077)
        task_id="T-077"
        task_runner="${tracker_root}/scripts/run_t077_reference_all.sh"
        result_root="${data_root}/tmp/t077-reference-results"
        ;;
    T-078|T078)
        task_id="T-078"
        task_runner="${tracker_root}/scripts/run_t078_reference_all.sh"
        result_root="${data_root}/tmp/t078-reference-results"
        ;;
    T-079|T079)
        task_id="T-079"
        task_runner="${tracker_root}/scripts/run_t079_reference_all.sh"
        result_root="${data_root}/tmp/t079-reference-results"
        ;;
    T-080|T080)
        task_id="T-080"
        task_runner="${tracker_root}/scripts/run_t080_reference_all.sh"
        result_root="${data_root}/tmp/t080-reference-results"
        ;;
    *)
        echo "错误：--task 必须是 T-076、T-077、T-078、T-079 或 T-080。" >&2
        usage >&2
        exit 2
        ;;
esac

pytorch_root="${PYTORCH_ROOT:-${data_root}/src/pytorch}"
venv_root="${PASS_GPU_VENV:-${data_root}/envs/PassGPURef}"
cuda_root="${CUDA_HOME:-${data_root}/cuda-12.6}"
work_dir="${PASS_TRACKER_WORK_DIR:-${data_root}/tmp}"

for required in \
    "${venv_root}/bin/activate" \
    "${pytorch_root}/torch/__init__.py" \
    "${cuda_root}/bin/nvcc" \
    "${task_runner}"; do
    if [[ ! -e "${required}" ]]; then
        echo "错误：缺少必需路径 ${required}" >&2
        exit 2
    fi
done

mkdir -p \
    "${work_dir}" \
    "${data_root}/pip-cache" \
    "${data_root}/cache" \
    "${data_root}/triton-cache" \
    "${data_root}/inductor-cache" \
    "${data_root}/torch-extensions" \
    "${result_root}"

export HOME="${data_root}"
export TMPDIR="${work_dir}"
export PIP_CACHE_DIR="${data_root}/pip-cache"
export XDG_CACHE_HOME="${data_root}/cache"
export TRITON_CACHE_DIR="${data_root}/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="${data_root}/inductor-cache"
export TORCH_EXTENSIONS_DIR="${data_root}/torch-extensions"

# shellcheck disable=SC1090
source "${venv_root}/bin/activate"
export CUDA_HOME="${cuda_root}"
export PATH="${CUDA_HOME}/bin:${venv_root}/bin:${PATH}"
export CUDNN_ROOT="${venv_root}/lib/python3.12/site-packages/nvidia/cudnn"
export CUDNN_INCLUDE_DIR="${CUDNN_ROOT}/include"
export LD_LIBRARY_PATH="${CUDNN_ROOT}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
unset CUDA_COMPAT_DIR CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH

for required in "${CUDNN_INCLUDE_DIR}/cudnn.h" "${CUDNN_ROOT}/lib/libcudnn.so.9"; do
    if [[ ! -e "${required}" ]]; then
        echo "错误：缺少 cuDNN 文件 ${required}" >&2
        exit 2
    fi
done

export PYTHON="${venv_root}/bin/python"
export PASS_TRACKER_WORK_DIR="${work_dir}"

echo "task=${task_id}"
echo "tracker_root=${tracker_root}"
echo "pytorch_root=${pytorch_root}"
echo "result_root=${result_root}"
echo "python=${PYTHON}"

cd "${work_dir}"

validate_args=(--pytorch-root "${pytorch_root}" --validate-only)
if [[ -n "${case_id}" ]]; then
    validate_args+=(--case "${case_id}")
fi
bash "${task_runner}" "${validate_args[@]}"

if ((validate_only)); then
    echo "gpu_task_validation=OK task=${task_id}"
    exit 0
fi

if [[ -z "${gpu_id}" ]]; then
    echo "错误：实际运行必须通过 --gpu ID 或 CUDA_VISIBLE_DEVICES 指定 GPU。" >&2
    exit 2
fi
if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "错误：一键入口只接受单个物理 GPU 编号，当前值为 ${gpu_id}。" >&2
    exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "错误：找不到 nvidia-smi，不能验证 GPU 是否空闲。" >&2
    exit 2
fi
if ! gpu_processes="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>&1)"; then
    echo "错误：无法查询 GPU ${gpu_id}：${gpu_processes}" >&2
    exit 2
fi
if [[ -n "${gpu_processes//[[:space:]]/}" ]]; then
    echo "错误：GPU ${gpu_id} 已存在计算进程，拒绝启动：${gpu_processes//$'\n'/,}" >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo "gpu_preflight=free physical_gpu=${gpu_id}"

launcher_log="$(mktemp "${work_dir}/${task_id,,}-gpu-launch.XXXXXX.log")"
run_args=(--pytorch-root "${pytorch_root}" --output-root "${result_root}")
if [[ -n "${case_id}" ]]; then
    run_args+=(--case "${case_id}")
fi

set +e
bash "${task_runner}" "${run_args[@]}" 2>&1 | tee "${launcher_log}"
runner_status=${PIPESTATUS[0]}
set -e

run_dir="$(sed -n 's/^artifacts=//p' "${launcher_log}" | tail -n 1)"
if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "错误：runner 未返回有效 artifacts 目录；启动日志：${launcher_log}" >&2
    exit "${runner_status:-1}"
fi

text_handoff="${run_dir}-text-handoff.json"
set +e
"${PYTHON}" "${tracker_root}/scripts/export_reference_text.py" \
    --run-dir "${run_dir}" \
    --output "${text_handoff}"
export_status=$?
set -e

ln -sfn "$(basename "${run_dir}")" "${result_root}/latest"
if ((export_status == 0)); then
    ln -sfn "$(basename "${text_handoff}")" "${result_root}/latest-text-handoff.json"
fi

echo "gpu_task_status=${runner_status}"
echo "run_dir=${run_dir}"
echo "latest_run=${result_root}/latest"
if ((export_status == 0)); then
    echo "text_handoff=${text_handoff}"
    echo "latest_text_handoff=${result_root}/latest-text-handoff.json"
    sha256sum "${text_handoff}"
else
    echo "警告：本轮 artifacts 不完整，文本 handoff 导出失败（状态 ${export_status}）。" >&2
fi

if ((runner_status != 0)); then
    exit "${runner_status}"
fi
exit "${export_status}"
