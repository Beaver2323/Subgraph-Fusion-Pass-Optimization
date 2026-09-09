#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
"${PYTHON:-python}" "${repo_root}/scripts/validate_prepared_tasks.py" --task T-084
exec bash "${repo_root}/scripts/run_reference_all.sh" \
  --manifest-path upstream/t084_manifest.yaml \
  --plan-path upstream/t084_reference_plan.yaml \
  "$@"
