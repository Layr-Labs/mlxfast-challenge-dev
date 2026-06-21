#!/usr/bin/env bash
# Overlay only benchmark.json editablePaths from an untrusted submission checkout.
set -euo pipefail

: "${SUBMISSION_WORKTREE:?SUBMISSION_WORKTREE is required}"

CONTRACT_PATH="${CONTRACT_PATH:-benchmark.json}"

if [[ ! -d "${SUBMISSION_WORKTREE}" ]]; then
  echo "::error::submission worktree not found at ${SUBMISSION_WORKTREE}" >&2
  exit 1
fi
if [[ ! -f "${CONTRACT_PATH}" ]]; then
  echo "::error::trusted benchmark contract missing at ${CONTRACT_PATH}" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "::error::jq is required to read editablePaths from ${CONTRACT_PATH}" >&2
  exit 1
fi

validate_contract_path() {
  local path="$1"
  if [[ -z "${path}" || "${path}" == /* || "${path}" == *\\* ]]; then
    echo "::error::invalid editable path '${path}' in ${CONTRACT_PATH}" >&2
    exit 1
  fi
  case "/${path}/" in
    *"/../"*|*"/./"*)
      echo "::error::invalid editable path '${path}' in ${CONTRACT_PATH}" >&2
      exit 1
      ;;
  esac
}

while IFS= read -r editable_path; do
  validate_contract_path "${editable_path}"

  source_path="${SUBMISSION_WORKTREE}/${editable_path}"
  target_path="${editable_path}"

  if [[ ! -e "${source_path}" ]]; then
    echo "::error file=${editable_path}::submitted editable path is missing" >&2
    exit 1
  fi
  if find "${source_path}" -type l -print -quit | grep -q .; then
    echo "::error file=${editable_path}::submitted editable paths must not contain symlinks" >&2
    exit 1
  fi

  rm -rf "${target_path}"
  mkdir -p "$(dirname "${target_path}")"
  if [[ -d "${source_path}" ]]; then
    mkdir -p "${target_path}"
    (cd "${source_path}" && tar -cf - .) | (cd "${target_path}" && tar -xf -)
  else
    cp "${source_path}" "${target_path}"
  fi

  echo "benchmark: overlaid editable path ${editable_path}"
done < <(jq -r '.editablePaths[]' "${CONTRACT_PATH}")

echo "benchmark: trusted harness retained; submitted editable paths overlaid"
