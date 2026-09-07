#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
work_dir="/home/z50063656/tmp"
task=""
unit=""
npu_ids=""

usage() {
    cat <<'EOF'
用法：
  bash scripts/run_t081_t083_performance.sh --task T-081 --unit constant-fold --npu 0
  bash scripts/run_t081_t083_performance.sh --task T-082 --unit view --npu 0
  bash scripts/run_t081_t083_performance.sh --task T-083 --unit all-reduce --npu 0,1

单元：
  T-081: constant-fold, convert
  T-082: permute, view
  T-083: all-gather, all-reduce, reduce-scatter（必须两张不同NPU）
EOF
}

while (($#)); do
    case "$1" in
        --task) task="${2:-}"; shift 2 ;;
        --unit) unit="${2:-}"; shift 2 ;;
        --npu) npu_ids="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "错误：未知参数 $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$(pwd -P)" != "${work_dir}" ]]; then
    echo "错误：必须从 ${work_dir} 启动" >&2
    exit 2
fi

# 性能测量禁止同一 tracker 在多卡上并发；即使物理卡不重叠，编译线程、
# 主机调度和 HCCL 仍共享资源，会污染 host/Event 尾延迟。
exec 9>"${work_dir}/pass-tracker-npu-performance.lock"
if ! flock -n 9; then
    echo "错误：已有 tracker NPU 性能任务运行；请等待其完成后重试" >&2
    exit 3
fi

case "${task}:${unit}" in
    T-081:constant-fold|T-081:convert|T-082:permute|T-082:view) world_size=1 ;;
    T-083:all-gather|T-083:all-reduce|T-083:reduce-scatter) world_size=2 ;;
    *) echo "错误：task与unit组合无效" >&2; usage >&2; exit 2 ;;
esac

if ((world_size == 1)); then
    [[ "${npu_ids}" =~ ^[0-9]+$ ]] || {
        echo "错误：T-081/T-082要求单个NPU" >&2
        exit 2
    }
else
    [[ "${npu_ids}" =~ ^[0-9]+,[0-9]+$ ]] || {
        echo "错误：T-083要求两个NPU，例如0,1" >&2
        exit 2
    }
    [[ "${npu_ids%%,*}" != "${npu_ids##*,}" ]] || {
        echo "错误：T-083的两个NPU不能相同" >&2
        exit 2
    }
fi

# activate_pass.sh 可能因清理不存在的alias返回非零；忽略该非致命返回值，
# 再以环境名和解释器绝对路径确认激活结果。
# shellcheck disable=SC1091
set +e
source /home/z50063656/Pass/activate_pass.sh >/tmp/t081-t083-performance-activate.log
set -e
if [[ "${CONDA_DEFAULT_ENV:-}" != "Pass" ]] || \
   [[ "$(command -v python)" != "/home/z50063656/envs/Pass/bin/python" ]]; then
    echo "错误：Pass环境未正确激活，详见 /tmp/t081-t083-performance-activate.log" >&2
    exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES="${npu_ids}"
export SET_NPU_DEVICE=0
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export PASS_TRACKER_WORK_DIR="${work_dir}"

gate="${repo_root}/results/current/${task}/performance_gates/${unit}.json"
if [[ ! -f "${gate}" ]]; then
    echo "错误：缺少已复核性能门禁：${gate}" >&2
    exit 2
fi

exec python "${repo_root}/scripts/run_prepared_performance.py" \
    --task "${task}" \
    --unit "${unit}" \
    --device npu \
    --gate "${gate}"
