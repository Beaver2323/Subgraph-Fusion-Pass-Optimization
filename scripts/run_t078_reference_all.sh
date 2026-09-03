#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

exec "${repo_root}/scripts/run_reference_all.sh" \
    --manifest-path upstream/t078_manifest.yaml \
    --plan-path upstream/t078_reference_plan.yaml \
    "$@"
