#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
tracker_root="${TRACKER_ROOT:-${repo_root}}"
data_root="${PASS_GPU_DATA_ROOT:-/data/z50063656}"
task_id=""
gpu_id="${CUDA_VISIBLE_DEVICES:-}"
gpu_ids=""
gpu_option_set=0
case_ids=()
validate_only=0
wait_gpu=0
wait_timeout=0
poll_interval=1
wait_options_set=0
execution_mode=shared
min_free_memory=1024
handoff_single_file_bytes="${PASS_GPU_HANDOFF_SINGLE_FILE_BYTES:-98304}"

usage() {
    cat <<'EOF'
用法：
  bash scripts/run_gpu_reference_task.sh --task T-078 --gpu 2
  bash scripts/run_gpu_reference_task.sh --task T-078 --gpu 2 --wait-gpu
  bash scripts/run_gpu_reference_task.sh --task T-078 --gpu 2 --wait-gpu --wait-timeout 7200
  bash scripts/run_gpu_reference_task.sh --task T-078 --gpu 2 --exclusive --wait-gpu
  bash scripts/run_gpu_reference_task.sh --task T-079 --gpu 2
  bash scripts/run_gpu_reference_task.sh --task T-080 --gpu 2
  bash scripts/run_gpu_reference_task.sh --task T-085 --gpus 2,3 --wait-gpu
  bash scripts/run_gpu_reference_task.sh --task T-076 --gpu 2 --case REF-mm-plus-mm-native
  bash scripts/run_gpu_reference_task.sh --task T-078 --validate-only

参数：
  --task T-076|...|T-113   必填；零GPU-ready批次仅允许--validate-only
  T-083 原生功能例使用 world_size=1，只需1卡；不证明跨rank通信收益。
  T-085 完整原生 suite 含真实2-rank NCCL，必须用 --gpus 指定两张卡。
  --gpu ID             实际运行时使用的物理 GPU 编号；也可预先设置 CUDA_VISIBLE_DEVICES
  --gpus ID1,ID2       T-085 实际运行的两张物理 GPU
  --case CASE_ID       可选，只运行指定 case；可重复指定多个 case
  --validate-only      只做零设备静态校验
  --exclusive          可选独占策略：启动时无计算进程；默认 shared，允许已有进程
  --min-free-memory-mib MIB  最低空闲显存，默认 1024 MiB；0 取消显存门槛，不保证无 OOM
  --wait-gpu           指定卡不满足当前策略时轮询等待；不指定则立即退出
  --wait-timeout SEC   等卡最大秒数，默认 0 表示一直等；需 --wait-gpu
  --poll-interval SEC  检查间隔，1～60 秒，默认 1；需 --wait-gpu
  -h, --help           显示帮助

路径可通过环境变量覆盖：
  PASS_GPU_DATA_ROOT、TRACKER_ROOT、PYTORCH_ROOT、PASS_GPU_VENV、CUDA_HOME
EOF
}

while (($#)); do
    case "$1" in
        --task|--gpu|--gpus|--case|--wait-timeout|--poll-interval|--min-free-memory-mib)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                echo "错误：$1 缺少参数值。" >&2
                exit 2
            fi
            ;;
    esac
    case "$1" in
        --task)
            task_id="${2:-}"
            shift 2
            ;;
        --gpu)
            gpu_id="${2:-}"
            gpu_option_set=1
            shift 2
            ;;
        --gpus)
            gpu_ids="${2:-}"
            shift 2
            ;;
        --case)
            case_ids+=("${2:-}")
            shift 2
            ;;
        --validate-only)
            validate_only=1
            shift
            ;;
        --wait-gpu)
            wait_gpu=1
            shift
            ;;
        --exclusive)
            execution_mode=exclusive
            shift
            ;;
        --min-free-memory-mib)
            min_free_memory="$2"
            shift 2
            ;;
        --wait-timeout)
            wait_timeout="$2"
            wait_options_set=1
            shift 2
            ;;
        --poll-interval)
            poll_interval="$2"
            wait_options_set=1
            shift 2
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
        task_plan="${tracker_root}/upstream/reference_plan.yaml"
        result_root="${data_root}/tmp/t076-reference-results"
        ;;
    T-077|T077)
        task_id="T-077"
        task_runner="${tracker_root}/scripts/run_t077_reference_all.sh"
        task_plan="${tracker_root}/upstream/t077_reference_plan.yaml"
        result_root="${data_root}/tmp/t077-reference-results"
        ;;
    T-078|T078)
        task_id="T-078"
        task_runner="${tracker_root}/scripts/run_t078_reference_all.sh"
        task_plan="${tracker_root}/upstream/t078_reference_plan.yaml"
        result_root="${data_root}/tmp/t078-reference-results"
        ;;
    T-079|T079)
        task_id="T-079"
        task_runner="${tracker_root}/scripts/run_t079_reference_all.sh"
        task_plan="${tracker_root}/upstream/t079_reference_plan.yaml"
        result_root="${data_root}/tmp/t079-reference-results"
        ;;
    T-080|T080)
        task_id="T-080"
        task_runner="${tracker_root}/scripts/run_t080_reference_all.sh"
        task_plan="${tracker_root}/upstream/t080_reference_plan.yaml"
        result_root="${data_root}/tmp/t080-reference-results"
        ;;
    T-081|T081|T-082|T082|T-083|T083|T-084|T084|T-085|T085|T-086|T086|T-087|T087|T-088|T088|T-089|T089|T-090|T090|T-091|T091|T-092|T092|T-093|T093|T-094|T094|T-095|T095|T-096|T096|T-097|T097|T-098|T098|T-099|T099|T-100|T100|T-101|T101|T-102|T102|T-103|T103|T-104|T104|T-105|T105|T-106|T106|T-107|T107|T-108|T108|T-109|T109|T-110|T110|T-111|T111|T-112|T112|T-113|T113)
        task_id="T-${task_id//[!0-9]/}"
        task_suffix="${task_id,,}"
        task_suffix="${task_suffix//-/}"
        task_runner="${tracker_root}/scripts/run_${task_suffix}_reference_all.sh"
        task_plan="${tracker_root}/upstream/${task_suffix}_reference_plan.yaml"
        result_root="${data_root}/tmp/${task_suffix}-reference-results"
        ;;
    *)
        echo "错误：--task 必须是 T-076～T-113。" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ "${task_id}" == "T-085" ]]; then
    dispatch_args=()
    if ((validate_only)); then
        dispatch_args+=(--validate-only)
    else
        if ((gpu_option_set)); then
            echo "错误：T-085 完整 suite 需要 --gpus ID1,ID2，不能使用 --gpu。" >&2
            exit 2
        fi
        dispatch_args+=(--gpus "${gpu_ids}")
    fi
    for case_id in "${case_ids[@]}"; do dispatch_args+=(--case "${case_id}"); done
    if ((wait_gpu)); then dispatch_args+=(--wait-gpu); fi
    if [[ "${execution_mode}" == exclusive ]]; then dispatch_args+=(--exclusive); fi
    dispatch_args+=(
        --wait-timeout "${wait_timeout}"
        --poll-interval "${poll_interval}"
        --min-free-memory-mib "${min_free_memory}"
    )
    exec bash "${tracker_root}/scripts/run_t085_gpu_all.sh" "${dispatch_args[@]}"
fi

if [[ -n "${gpu_ids}" ]]; then
    echo "错误：只有 T-085 接受 --gpus；其他任务请使用 --gpu。" >&2
    exit 2
fi

# shellcheck disable=SC1090
source "${tracker_root}/scripts/gpu_wait.sh"
tracker_gpu_validate_wait_options "${wait_gpu}" "${wait_timeout}" "${poll_interval}" "${min_free_memory}"
if ((wait_options_set && !wait_gpu)); then
    echo "错误：--wait-timeout/--poll-interval 需要同时指定 --wait-gpu。" >&2
    exit 2
fi

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

if [[ ! "${handoff_single_file_bytes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：PASS_GPU_HANDOFF_SINGLE_FILE_BYTES 必须是正整数。" >&2
    exit 2
fi

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
for case_id in "${case_ids[@]}"; do
    validate_args+=(--case "${case_id}")
done
bash "${task_runner}" "${validate_args[@]}"

if ((validate_only)); then
    echo "gpu_task_validation=OK task=${task_id}"
    exit 0
fi

task_number="${task_id#T-}"
if ((10#${task_number} >= 91)); then
    "${PYTHON}" "${tracker_root}/scripts/check_gpu_task_runnable.py" \
        --plan "${task_plan}"
fi

if [[ -z "${gpu_id}" ]]; then
    echo "错误：实际运行必须通过 --gpu ID 或 CUDA_VISIBLE_DEVICES 指定 GPU。" >&2
    exit 2
fi
if [[ ! "${gpu_id}" =~ ^[0-9]{1,9}$ ]]; then
    echo "错误：一键入口只接受单个物理 GPU 编号，当前值为 ${gpu_id}。" >&2
    exit 2
fi

gpu_id=$((10#${gpu_id}))
tracker_gpu_acquire "${gpu_id}" "${wait_gpu}" "${wait_timeout}" "${poll_interval}" "${data_root}/tmp/tracker-gpu-locks" "${execution_mode}" "${min_free_memory}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

launcher_log="$(mktemp "${work_dir}/${task_id,,}-gpu-launch.XXXXXX.log")"
run_args=(--pytorch-root "${pytorch_root}" --output-root "${result_root}")
for case_id in "${case_ids[@]}"; do
    run_args+=(--case "${case_id}")
done

set +e
bash "${task_runner}" "${run_args[@]}" 2>&1 | tee "${launcher_log}"
runner_status=${PIPESTATUS[0]}
set -e

run_dir="$(sed -n 's/^artifacts=//p' "${launcher_log}" | tail -n 1)"
if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "错误：runner 未返回有效 artifacts 目录；启动日志：${launcher_log}" >&2
    if ((runner_status == 0)); then runner_status=1; fi
    "${PYTHON}" "${tracker_root}/scripts/publish_reference_latest.py" \
        --result-root "${result_root}" --runner-status "${runner_status}" --export-status 1
    exit "${runner_status}"
fi

text_handoff="${run_dir}/text-handoff.json"
text_handoff_parts="${run_dir}/text-handoff-parts"
set +e
"${PYTHON}" "${tracker_root}/scripts/export_reference_text.py" \
    --run-dir "${run_dir}" \
    --profile review \
    --bundle-raw-text \
    --compact \
    --allow-derived-output \
    --output "${text_handoff}" \
    --split-output-dir "${text_handoff_parts}" \
    --auto-split-over-bytes "${handoff_single_file_bytes}"
export_status=$?
set -e

"${PYTHON}" "${tracker_root}/scripts/publish_reference_latest.py" \
    --result-root "${result_root}" --run-dir "${run_dir}" \
    --runner-status "${runner_status}" --export-status "${export_status}"

echo "gpu_task_status=${runner_status}"
echo "run_dir=${run_dir}"
echo "latest_run=${result_root}/latest"
if ((export_status == 0)); then
    echo "text_handoff=${text_handoff}"
    echo "latest_text_handoff=${result_root}/latest-text-handoff.json"
    if [[ -f "${text_handoff_parts}/manifest.json" ]]; then
        echo "handoff_upload_mode=split"
        echo "handoff_upload_input=${result_root}/latest/text-handoff-parts/manifest.json"
        echo "text_handoff_parts=${text_handoff_parts}"
    else
        echo "handoff_upload_mode=single-file"
        echo "handoff_upload_input=${result_root}/latest-text-handoff.json"
    fi
    sha256sum "${text_handoff}"
else
    echo "警告：本轮 artifacts 不完整，文本 handoff 导出失败（状态 ${export_status}）。" >&2
fi

if ((runner_status != 0)); then
    exit "${runner_status}"
fi
exit "${export_status}"
