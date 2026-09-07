#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$(pwd -P)" != "/home/z50063656/tmp" ]]; then
    echo "错误：必须从 /home/z50063656/tmp 启动" >&2
    exit 2
fi

npu_id="${1:-0}"
if [[ ! "${npu_id}" =~ ^[0-9]+$ ]]; then
    echo "用法：bash scripts/run_t080_scatter.sh [NPU_ID]" >&2
    exit 2
fi

# 兼容旧入口。Scatter 候选已因同后端性能与显存回退进入产品默认门禁，
# 因此这里不再绕过门禁重放“ON 命中”候选套件，只验证最终产品状态。
echo "INFO T-080 Scatter 候选已默认关闭；转入最终产品门禁双臂验证。"
exec bash "${repo_root}/scripts/verify_t080_scatter_gate.sh" "${npu_id}"
