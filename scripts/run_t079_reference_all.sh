#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

python "${repo_root}/scripts/validate_prepared_tasks.py" --task T-079

exec "${repo_root}/scripts/run_reference_all.sh" \
    --manifest-path upstream/t079_manifest.yaml \
    --plan-path upstream/t079_reference_plan.yaml \
    "$@"
