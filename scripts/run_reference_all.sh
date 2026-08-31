#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
work_dir="${PASS_TRACKER_WORK_DIR:-/home/z50063656/tmp}"
actual_dir="$(pwd -P)"

if [[ "${actual_dir}" != "$(cd "${work_dir}" && pwd -P)" ]]; then
    echo "错误：请先 cd ${work_dir}，所有测试必须从该目录发起。" >&2
    exit 2
fi

exec "${PYTHON:-python}" \
    "${repo_root}/runners/reference_runner.py" \
    --repo-root "${repo_root}" \
    --work-dir "${work_dir}" \
    "$@"
