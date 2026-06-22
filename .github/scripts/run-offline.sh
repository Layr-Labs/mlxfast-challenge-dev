#!/usr/bin/env bash
# Run a command under the macOS no-network Seatbelt profile.
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: run-offline.sh COMMAND [ARG...]" >&2
  exit 2
fi

if ! command -v sandbox-exec >/dev/null 2>&1; then
  echo "run-offline.sh: sandbox-exec not found; offline command execution requires macOS" >&2
  exit 1
fi

sandbox_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

absolute_path() {
  local path="$1"
  local dir
  local base
  dir="$(dirname "${path}")"
  base="$(basename "${path}")"
  if [[ "${dir}" = "." ]]; then
    printf '%s/%s\n' "${PWD}" "${base}"
  else
    (cd "${dir}" 2>/dev/null && printf '%s/%s\n' "${PWD}" "${base}") || printf '%s\n' "${path}"
  fi
}

absolute_executable() {
  local executable="$1"
  local dir
  local base
  if [[ "${executable}" == */* ]]; then
    dir="$(dirname "${executable}")"
    base="$(basename "${executable}")"
    (cd "${dir}" 2>/dev/null && printf '%s/%s\n' "${PWD}" "${base}") || return 1
  else
    command -v "${executable}"
  fi
}

write_allowed_writes() {
  local raw="${MLXFAST_OFFLINE_WRITABLE_PATHS:-}"
  local old_ifs
  local path
  if [[ -z "${raw}" ]]; then
    return 0
  fi

  old_ifs="${IFS}"
  IFS=:
  for path in ${raw}; do
    if [[ -n "${path}" ]]; then
      printf '(allow file-write* (subpath "%s"))\n' "$(sandbox_escape "$(absolute_path "${path}")")"
    fi
  done
  IFS="${old_ifs}"
}

write_strict_profile() {
  local executable="$1"
  local profile
  profile="$(mktemp "${TMPDIR:-/tmp}/mlxfast-offline.XXXXXX")"
  {
    cat <<EOF
(version 1)
(allow default)
(deny network*)
(deny process-fork)
(deny process-exec*)
(allow process-exec (literal "$(sandbox_escape "${executable}")"))
(deny file-write*)
EOF
    write_allowed_writes
  } > "${profile}"
  printf '%s\n' "${profile}"
}

curl_executable="$(absolute_executable curl)"
curl_profile="$(write_strict_profile "${curl_executable}")"
if sandbox-exec -f "${curl_profile}" \
    "${curl_executable}" -fsS --max-time 10 https://example.com -o /dev/null 2>/dev/null; then
  echo "run-offline.sh: sandbox profile did not block network access; refusing to run" >&2
  exit 1
fi

command_executable="$(absolute_executable "$1")"
shift
command_profile="$(write_strict_profile "${command_executable}")"

echo "run-offline.sh: network egress and child process execution are blocked; running: ${command_executable} $*"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9
exec sandbox-exec -f "${command_profile}" "${command_executable}" "$@"
