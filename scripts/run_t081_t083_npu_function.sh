#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
work_dir="/home/z50063656/tmp"
task=""
npu_ids=""

usage() {
    echo "用法：bash scripts/run_t081_t083_npu_function.sh --task T-081|T-082|T-083 --npu 0或0,1" >&2
}

while (($#)); do
    case "$1" in
        --task)
            task="${2:-}"
            shift 2
            ;;
        --npu)
            npu_ids="${2:-}"
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ "${task}" != "T-081" && "${task}" != "T-082" && "${task}" != "T-083" ]]; then
    usage
    exit 2
fi
if [[ "$(pwd -P)" != "${work_dir}" ]]; then
    echo "错误：必须从 ${work_dir} 启动" >&2
    exit 2
fi
if [[ "${task}" == "T-083" ]]; then
    if [[ ! "${npu_ids}" =~ ^[0-9]+,[0-9]+$ ]]; then
        echo "错误：T-083要求两个不同NPU，例如--npu 0,1" >&2
        exit 2
    fi
    if [[ "${npu_ids%%,*}" == "${npu_ids##*,}" ]]; then
        echo "错误：T-083的两个NPU不能相同" >&2
        exit 2
    fi
else
    if [[ ! "${npu_ids}" =~ ^[0-9]+$ ]]; then
        echo "错误：T-081/T-082要求单个NPU，例如--npu 0" >&2
        exit 2
    fi
fi

# activate_pass.sh 末尾会尝试清理一组可选 alias；alias 不存在时 source 的
# 最终返回值是 1，但在此之前 Pass conda 环境已经成功激活。不能让 set -e 把
# 这个非致命返回值误判成环境失败，随后用实际环境状态做强校验。
# shellcheck disable=SC1091
set +e
source /home/z50063656/Pass/activate_pass.sh >/tmp/t081-t083-npu-activate.log
set -e
if [[ "${CONDA_DEFAULT_ENV:-}" != "Pass" ]] || \
   [[ "$(command -v python)" != "/home/z50063656/envs/Pass/bin/python" ]]; then
    echo "错误：Pass 环境未正确激活，详见 /tmp/t081-t083-npu-activate.log" >&2
    exit 1
fi
export ASCEND_RT_VISIBLE_DEVICES="${npu_ids}"
export SET_NPU_DEVICE=0
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export PASS_TRACKER_WORK_DIR="${work_dir}"

case "${task}" in
    T-081) units=(constant-fold convert) ;;
    T-082) units=(permute view) ;;
    T-083) units=(all-gather all-reduce reduce-scatter) ;;
esac

task_lower="${task,,}"
task_compact="${task_lower/-/}"
timestamp="$(date +%Y%m%dT%H%M%S%z)"
run_root="${work_dir}/${task_compact}-npu-functional-results/functional-${timestamp}"
mkdir -p "${run_root}"

for unit in "${units[@]}"; do
    for mode in off on; do
        output="${run_root}/${unit}/${mode}"
        mkdir -p "${output}"
        echo "START task=${task} unit=${unit} mode=${mode} backend=triton_experimental"
        if [[ "${task}" == "T-083" ]]; then
            python -m torch.distributed.run \
                --standalone \
                --nproc-per-node=2 \
                "${repo_root}/runners/t081_t083_performance_worker.py" \
                --phase functional \
                --unit "${unit}" \
                --mode "${mode}" \
                --device npu \
                --output "${output}"
        else
            python "${repo_root}/runners/t081_t083_performance_worker.py" \
                --phase functional \
                --unit "${unit}" \
                --mode "${mode}" \
                --device npu \
                --output "${output}"
        fi
        echo "END task=${task} unit=${unit} mode=${mode}"
    done
done

ln -sfn "${run_root}" "${work_dir}/${task_compact}-npu-functional-results/latest"
echo "npu_functional_run=${run_root}"
echo "latest=${work_dir}/${task_compact}-npu-functional-results/latest"
