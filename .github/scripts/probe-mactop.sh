#!/usr/bin/env bash
# Diagnose whether a macOS runner exposes mactop/IOReport counters.
set -euo pipefail

seconds="${MLXFAST_MACTOP_PROBE_SECONDS:-3}"
interval_ms="${MLXFAST_MACTOP_PROBE_INTERVAL_MS:-100}"

if ! command -v mactop >/dev/null 2>&1; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "probe-mactop: Homebrew is required to install mactop" >&2
    exit 1
  fi
  echo "probe-mactop: installing mactop"
  brew install mactop
fi

mactop_bin="$(command -v mactop)"
echo "probe-mactop: mactop=${mactop_bin}"
echo "probe-mactop: user=$(id)"
echo "probe-mactop: sw_vers"
sw_vers || true
echo "probe-mactop: uname=$(uname -a)"
echo "probe-mactop: cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"

run_mactop_probe() {
  local label="$1"
  shift
  local stdout_path
  local stderr_path
  local status
  stdout_path="$(mktemp "${TMPDIR:-/tmp}/mlxfast-mactop-${label}.out.XXXXXX")"
  stderr_path="$(mktemp "${TMPDIR:-/tmp}/mlxfast-mactop-${label}.err.XXXXXX")"

  echo "probe-mactop: ${label} start command=$*"
  set +e
  "$@" --headless --interval "${interval_ms}" --format json >"${stdout_path}" 2>"${stderr_path}" &
  local pid="$!"
  sleep "${seconds}"
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null
  fi
  wait "${pid}"
  status="$?"
  set -e

  echo "probe-mactop: ${label} status=${status}"
  echo "probe-mactop: ${label} stdout_bytes=$(wc -c < "${stdout_path}" | tr -d ' ')"
  echo "probe-mactop: ${label} stderr_bytes=$(wc -c < "${stderr_path}" | tr -d ' ')"
  echo "probe-mactop: ${label} stdout_head"
  sed -n '1,5p' "${stdout_path}" || true
  echo "probe-mactop: ${label} stderr"
  cat "${stderr_path}" || true
}

run_mactop_probe "plain" "${mactop_bin}"

if sudo -n true 2>/dev/null; then
  run_mactop_probe "sudo" sudo -n "${mactop_bin}"
else
  echo "probe-mactop: sudo passwordless unavailable"
fi

if command -v sandbox-exec >/dev/null 2>&1 && [[ -f tools/deny-network.sb ]]; then
  run_mactop_probe "deny-network-sandbox" sandbox-exec -f tools/deny-network.sb "${mactop_bin}"
else
  echo "probe-mactop: sandbox-exec or tools/deny-network.sb unavailable"
fi

