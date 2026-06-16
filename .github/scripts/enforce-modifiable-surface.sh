#!/usr/bin/env bash
# Rejects diffs that touch files outside benchmark.json editablePaths.
# The allowlist is read from the BASE commit so a PR cannot grant itself access.
# Usage: BASE_SHA=<sha> HEAD_SHA=<sha> enforce-modifiable-surface.sh
set -euo pipefail

: "${BASE_SHA:?BASE_SHA is required}"
: "${HEAD_SHA:?HEAD_SHA is required}"

allowed="$(git show "${BASE_SHA}:benchmark.json" \
  | python3 -c 'import json,sys; print("\n".join(json.load(sys.stdin)["editablePaths"]))')"
changed="$(git diff --name-only "${BASE_SHA}" "${HEAD_SHA}")"

bad=0
while IFS= read -r f; do
  [[ -z "${f}" ]] && continue
  ok=0
  while IFS= read -r allowed_path; do
    [[ -z "${allowed_path}" ]] && continue
    # Exact match OR file is inside an allowed directory prefix.
    if [[ "${f}" == "${allowed_path}" || "${f}" == "${allowed_path}/"* ]]; then
      ok=1
      break
    fi
  done <<<"${allowed}"
  if [[ "${ok}" == "0" ]]; then
    echo "::error file=${f}::${f} is outside the modifiable surface (see editablePaths in benchmark.json)"
    bad=1
  fi
done <<<"${changed}"
exit "${bad}"
