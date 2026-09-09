#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
pytorch_root=""
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  if [[ "${arguments[index]}" == "--pytorch-root" ]] && \
     ((index + 1 < ${#arguments[@]})); then
    pytorch_root="${arguments[index + 1]}"
    break
  fi
done

validator=("${PYTHON:-python}" "${repo_root}/scripts/validate_t085_preparation.py")
if [[ -n "${pytorch_root}" ]]; then
  validator+=(--pytorch-root "${pytorch_root}")
fi
"${validator[@]}"

exec bash "${repo_root}/scripts/run_reference_all.sh" \
  --manifest-path upstream/t085_manifest.yaml \
  --plan-path upstream/t085_reference_plan.yaml \
  "$@"
