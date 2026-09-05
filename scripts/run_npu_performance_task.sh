#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
task_id=""
npu_id=""
unit="gumbel"
validate_only=0

usage() {
    cat <<'EOF'
用法：
  bash scripts/run_npu_performance_task.sh --task T-077 --unit gumbel --npu 0
  bash scripts/run_npu_performance_task.sh --task T-077 --unit decompose-mm --npu 0
  bash scripts/run_npu_performance_task.sh --task T-076 --validate-only
  bash scripts/run_npu_performance_task.sh --task T-078 --validate-only

说明：
  T-076 的性能阶段复用仓库内同冻结版本的既有三轮 A/B 证据；明确关闭的三类 pad 免测。
  T-077 的 --unit 支持 gumbel、decompose-bmm、decompose-mm、decompose-addmm。
  decompose 三项只做测试态最小 capability 适配，不修改产品源码；B2B 使用独立 capability 探针。
  T-078～T-080 仅完成性能方案，worker 尚未实现；当前只开放 --validate-only。
  后续必须先实现并验证 worker，再凭 GPU reference 与 NPU 功能/命中证据开放实测。
EOF
}

while (($#)); do
    case "$1" in
        --task) task_id="${2:-}"; shift 2 ;;
        --npu) npu_id="${2:-}"; shift 2 ;;
        --unit) unit="${2:-}"; shift 2 ;;
        --validate-only) validate_only=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "错误：未知参数 $1" >&2; usage >&2; exit 2 ;;
    esac
done

task_id="${task_id^^}"
case "${task_id}" in
    T-076|T076)
        for path in \
            report/p0_sweep_performance_20260820.md \
            report/t025_t026_pad_family_20260821.md \
            report/t058_experimental_addmm_gate_20260826.md; do
            test -s "${repo_root}/${path}" || { echo "错误：缺少 ${path}" >&2; exit 2; }
        done
        echo "performance_task=T-076 status=disposition-complete-two-backend-aligned-measurements-three-explicitly-disabled-exemptions"
        echo "plan=${repo_root}/upstream/t076_performance_plan.yaml"
        exit 0
        ;;
    T-077|T077)
        task_id="T-077"
        ;;
    T-078|T078|T-079|T079|T-080|T080)
        case "${task_id}" in
            T078) task_id="T-078" ;;
            T079) task_id="T-079" ;;
            T080) task_id="T-080" ;;
        esac
        python "${repo_root}/scripts/validate_prepared_tasks.py" --task "${task_id}"
        if ((validate_only)); then
            echo "performance_task_validation=OK task=${task_id} status=plan-only worker=not-implemented"
            exit 0
        fi
        echo "错误：${task_id} 性能 worker 尚未实现（不仅是等待解锁）；实现并验证后，仍需 GPU reference 与 NPU triton_experimental 功能/命中门禁。" >&2
        exit 4
        ;;
    *) echo "错误：--task 必须是 T-076～T-080" >&2; exit 2 ;;
esac

# 参数解析后 $# 已归零，再激活环境，避免 CANN set_env.sh 误收本脚本参数。
# shellcheck disable=SC1091
set +e
source /home/z50063656/Pass/activate_pass.sh >/tmp/pass-performance-activate.log
activate_status=$?
set -e
if ((activate_status != 0)); then
    echo "错误：Pass 环境激活失败，详见 /tmp/pass-performance-activate.log" >&2
    exit "${activate_status}"
fi

export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/pass-python-cache"
python -m py_compile \
    "${repo_root}/runners/gumbel_performance_worker.py" \
    "${repo_root}/runners/aggregate_performance.py" \
    "${repo_root}/runners/decompose_performance_worker.py" \
    "${repo_root}/runners/aggregate_decompose_performance.py" \
    "${repo_root}/issues/REF-b2b-gemm-native/capability_probe.py"
python -m json.tool "${repo_root}/upstream/t077_performance_plan.yaml" >/dev/null
if ((validate_only)); then
    echo "performance_task_validation=OK task=T-077 measured=4 capability_assessed=1 pending_capability=0"
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

run_id="performance-$(date '+%Y%m%dT%H%M%S%z')"
order=(off:1 on:1 on:2 off:2 off:3 on:3)
export ASCEND_RT_VISIBLE_DEVICES="${npu_id}"
export SET_NPU_DEVICE=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python3.11/site-packages/torch_npu/lib:${LD_LIBRARY_PATH:-}"

case "${unit}" in
    gumbel) ;;
    decompose-bmm) worker_unit="bmm"; warmup=10; runs=100 ;;
    decompose-mm) worker_unit="mm"; warmup=10; runs=100 ;;
    decompose-addmm) worker_unit="addmm"; warmup=5; runs=30 ;;
    *)
        echo "错误：--unit 必须是 gumbel、decompose-bmm、decompose-mm 或 decompose-addmm" >&2
        exit 2
        ;;
esac

if [[ "${unit}" != "gumbel" ]]; then
    run_dir="/home/z50063656/tmp/t077-performance-results/${unit}-${run_id}"
    mkdir -p "${run_dir}/workers"
    for item in "${order[@]}"; do
        mode="${item%%:*}"
        round="${item##*:}"
        worker_dir="${run_dir}/workers/${mode}${round}"
        mkdir -p "${worker_dir}"
        export TORCHINDUCTOR_CACHE_DIR="${worker_dir}/inductor-cache"
        export TRITON_CACHE_DIR="${worker_dir}/triton-cache"
        echo "START unit=${unit} mode=${mode} round=${round}"
        python "${repo_root}/runners/decompose_performance_worker.py" \
            --unit "${worker_unit}" \
            --mode "${mode}" \
            --round "${round}" \
            --warmup "${warmup}" \
            --runs "${runs}" \
            --output "${worker_dir}/result.json" \
            >"${worker_dir}/stdout.log" 2>"${worker_dir}/stderr.log"
        echo "END unit=${unit} mode=${mode} round=${round}"
    done
    python "${repo_root}/runners/aggregate_decompose_performance.py" \
        --run-dir "${run_dir}"
    ln -sfn "$(basename "${run_dir}")" \
        "/home/z50063656/tmp/t077-performance-results/latest-${unit}"
    echo "artifacts=${run_dir}"
    exit 0
fi

run_dir="/home/z50063656/tmp/t077-performance-results/${run_id}"
mkdir -p "${run_dir}/workers"
for item in "${order[@]}"; do
    mode="${item%%:*}"
    round="${item##*:}"
    worker_dir="${run_dir}/workers/${mode}${round}"
    mkdir -p "${worker_dir}"
    export TORCHINDUCTOR_CACHE_DIR="${worker_dir}/inductor-cache"
    export TRITON_CACHE_DIR="${worker_dir}/triton-cache"
    echo "START mode=${mode} round=${round}"
    python "${repo_root}/runners/gumbel_performance_worker.py" \
        --mode "${mode}" \
        --round "${round}" \
        --output "${worker_dir}/result.json" \
        >"${worker_dir}/stdout.log" 2>"${worker_dir}/stderr.log"
    echo "END mode=${mode} round=${round}"
done

python "${repo_root}/runners/aggregate_performance.py" --run-dir "${run_dir}"
ln -sfn "${run_id}" /home/z50063656/tmp/t077-performance-results/latest
echo "artifacts=${run_dir}"
