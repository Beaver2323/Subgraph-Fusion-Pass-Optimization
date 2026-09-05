#!/usr/bin/env bash
# GPU 一键入口的等卡函数；只查询模式/显存/进程，不占用显存、不修改 GPU 配置。

tracker_gpu_validate_wait_options() {
    local wait_enabled="$1" wait_timeout="$2" poll_interval="$3" min_free_memory="${4:-1024}"
    if [[ ! "${wait_timeout}" =~ ^[0-9]{1,9}$ ]] ||
       [[ ! "${poll_interval}" =~ ^[0-9]{1,2}$ ]]; then
        echo "错误：--wait-timeout 须为 0～999999999 的整数秒；--poll-interval 须为 1～60。" >&2
        return 2
    fi
    if ((10#${poll_interval} < 1 || 10#${poll_interval} > 60)); then
        echo "错误：--poll-interval 须为 1～60 的整数秒。" >&2
        return 2
    fi
    if [[ "${wait_enabled}" != 0 && "${wait_enabled}" != 1 ]]; then
        echo "错误：无效的等卡开关。" >&2
        return 2
    fi
    if [[ ! "${min_free_memory}" =~ ^[0-9]{1,9}$ ]]; then
        echo "错误：--min-free-memory-mib 须为 0～999999999 的整数。" >&2
        return 2
    fi
}

tracker_gpu_wait_cancel() {
    local status="$1" wait_child
    # 排队阶段本 shell 的后台作业只有自己的 sleep。用作业表避免信号落在
    # sleep 已启动但 $! 尚未赋给变量的窗口，留下持有输出管道的孤儿进程。
    # 不查询/终止 nvidia-smi 返回的 GPU 用户进程。
    for wait_child in $(jobs -pr); do
        kill "${wait_child}" 2>/dev/null || true
        wait "${wait_child}" 2>/dev/null || true
    done
    echo "gpu_wait=cancelled exit_code=${status}" >&2
    exit "${status}"
}

tracker_gpu_wait_sleep() {
    sleep "$1" &
    tracker_gpu_wait_sleep_pid=$!
    wait "${tracker_gpu_wait_sleep_pid}"
    tracker_gpu_wait_sleep_pid=""
}

tracker_gpu_acquire() {
    local gpu_id="$1" wait_enabled="$2" wait_timeout="$3" poll_interval="$4" lock_dir="$5"
    local execution_mode="${6:-shared}" min_free_memory="${7:-1024}"
    tracker_gpu_validate_wait_options "${wait_enabled}" "${wait_timeout}" "${poll_interval}" "${min_free_memory}" || return 2
    if [[ "${execution_mode}" != shared && "${execution_mode}" != exclusive ]]; then
        echo "错误：GPU 运行策略只能是 shared 或 exclusive。" >&2
        return 2
    fi
    if [[ ! "${gpu_id}" =~ ^[0-9]{1,9}$ ]]; then
        echo "错误：--gpu 须为单个物理 GPU 的整数编号。" >&2
        return 2
    fi
    gpu_id=$((10#${gpu_id}))
    wait_timeout=$((10#${wait_timeout}))
    poll_interval=$((10#${poll_interval}))
    min_free_memory=$((10#${min_free_memory}))
    local required
    for required in nvidia-smi flock timeout; do
        if ! command -v "${required}" >/dev/null 2>&1; then
            echo "错误：找不到 ${required}，不能安全检查 GPU。" >&2
            return 2
        fi
    done
    local lock_file="${lock_dir}/gpu-${gpu_id}.lock"
    if [[ -L "${lock_dir}" || -L "${lock_file}" ]] ||
       [[ -e "${lock_file}" && ! -f "${lock_file}" ]]; then
        echo "错误：GPU 协作锁目录/文件不得是软链接或特殊文件。" >&2
        return 2
    fi
    mkdir -p "${lock_dir}" || return 2
    # 文件保留，但锁由内核随进程结束释放；不能删除锁文件制造另一个锁 inode。
    exec {tracker_gpu_lock_fd}>>"${lock_file}" || return 2
    local started=${SECONDS} elapsed remaining delay query_limit processes lock_status status=0 reason
    local gpu_info compute_mode free_memory lock_kind=-s
    if [[ "${execution_mode}" == exclusive ]]; then lock_kind=-x; fi
    tracker_gpu_wait_sleep_pid=""
    trap 'tracker_gpu_wait_cancel 130' INT
    trap 'tracker_gpu_wait_cancel 143' TERM
    while true; do
        elapsed=$((SECONDS - started))
        if ((wait_enabled && wait_timeout > 0 && elapsed >= wait_timeout)); then
            echo "gpu_wait=timed-out physical_gpu=${gpu_id} elapsed_seconds=${elapsed}" >&2
            status=124
            break
        fi
        if flock "${lock_kind}" -n -E 3 "${tracker_gpu_lock_fd}"; then
            query_limit=10
            if ((wait_enabled && wait_timeout > 0)); then
                remaining=$((wait_timeout - elapsed))
                if ((remaining < query_limit)); then query_limit=${remaining}; fi
            fi
            if ! gpu_info="$(timeout -k 1 "${query_limit}" nvidia-smi -i "${gpu_id}" --query-gpu=compute_mode,memory.free --format=csv,noheader,nounits 2>&1)"; then
                echo "错误：GPU ${gpu_id} 查询失败或超时，停止排队：${gpu_info}" >&2
                status=2
                break
            fi
            if [[ ! "${gpu_info}" =~ ^[A-Za-z_[:blank:]]+,[[:blank:]]*[0-9]{1,9}[[:blank:]]*$ ]]; then
                echo "错误：GPU ${gpu_id} 模式/显存查询返回无法识别的内容：${gpu_info}" >&2
                status=2
                break
            fi
            IFS=, read -r compute_mode free_memory <<<"${gpu_info}"
            compute_mode="${compute_mode//[[:space:]]/}"
            compute_mode="${compute_mode^^}"
            free_memory="${free_memory//[[:space:]]/}"
            free_memory=$((10#${free_memory}))
            if [[ "${compute_mode}" != DEFAULT && "${compute_mode}" != EXCLUSIVE_PROCESS ]]; then
                echo "错误：GPU ${gpu_id} 计算模式 ${compute_mode} 不支持此入口；不会修改机器配置。" >&2
                status=2
                break
            fi
            reason=""
            if ((free_memory < min_free_memory)); then
                reason="insufficient-memory"
            elif [[ "${execution_mode}" == exclusive || "${compute_mode}" == EXCLUSIVE_PROCESS ]]; then
                # 共享模式在 DEFAULT 下不要求计算进程退出；硬件独占模式仍需查占用。
                elapsed=$((SECONDS - started))
                if ((wait_enabled && wait_timeout > 0)); then
                    remaining=$((wait_timeout - elapsed))
                    if ((remaining <= 0)); then
                        status=124
                        echo "gpu_wait=timed-out physical_gpu=${gpu_id} elapsed_seconds=${elapsed}" >&2
                        break
                    fi
                    if ((remaining < query_limit)); then query_limit=${remaining}; fi
                fi
                if ! processes="$(timeout -k 1 "${query_limit}" nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>&1)"; then
                    echo "错误：GPU ${gpu_id} 进程查询失败或超时：${processes}" >&2
                    status=2
                    break
                fi
                if [[ -n "${processes//[[:space:]]/}" ]]; then
                    if [[ ! "${processes}" =~ ^[0-9[:space:]]+$ ]]; then
                        echo "错误：GPU ${gpu_id} 进程查询返回无法识别的内容：${processes}" >&2
                        status=2
                        break
                    fi
                    reason="compute-processes"
                fi
            fi
            elapsed=$((SECONDS - started))
            if ((wait_enabled && wait_timeout > 0 && elapsed >= wait_timeout)); then
                echo "gpu_wait=timed-out physical_gpu=${gpu_id} elapsed_seconds=${elapsed}" >&2
                status=124
                break
            fi
            if [[ -z "${reason}" ]]; then
                export PASS_GPU_EXECUTION_MODE="${execution_mode}"
                export PASS_GPU_MIN_FREE_MEMORY_MIB="${min_free_memory}"
                export PASS_GPU_PREFLIGHT_FREE_MEMORY_MIB="${free_memory}"
                export PASS_GPU_COMPUTE_MODE="${compute_mode}"
                echo "gpu_preflight=ready physical_gpu=${gpu_id} execution_mode=${execution_mode} free_memory_mib=${free_memory} waited_seconds=${elapsed} cooperative_lock=held"
                break
            fi
            flock -u "${tracker_gpu_lock_fd}" || { status=2; break; }
        else
            lock_status=$?
            if ((lock_status != 3)); then
                echo "错误：GPU 协作锁操作失败（${lock_status}），停止排队。" >&2
                status=2
                break
            fi
            reason="tracker-lock"
        fi
        if ((!wait_enabled)); then
            echo "错误：GPU ${gpu_id} 忙（${reason}），拒绝启动；可加 --wait-gpu 等待。" >&2
            status=3
            break
        fi
        delay=${poll_interval}
        if ((wait_timeout > 0)); then
            remaining=$((wait_timeout - elapsed))
            if ((remaining < delay)); then delay=${remaining}; fi
        fi
        echo "gpu_wait=queued physical_gpu=${gpu_id} reason=${reason} elapsed_seconds=${elapsed} next_check_seconds=${delay} timeout_seconds=${wait_timeout}"
        tracker_gpu_wait_sleep "${delay}"
    done
    trap - INT TERM
    if ((status != 0)); then
        exec {tracker_gpu_lock_fd}>&-
        unset tracker_gpu_lock_fd
    fi
    # 成功时故意保留 FD，调用方整个运行/导出结束前都持锁。
    return "${status}"
}
