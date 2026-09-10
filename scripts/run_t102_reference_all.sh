#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec bash "${repo_root}/scripts/run_reference_all.sh" \
  --manifest-path upstream/t102_manifest.yaml \
  --plan-path upstream/t102_reference_plan.yaml \
  "$@"
