#!/usr/bin/env bash
# Run a command under the macOS no-network Seatbelt profile.
set -euo pipefail

SANDBOX_PROFILE="${MLXFAST_SANDBOX_PROFILE:-tools/deny-network.sb}"

if [[ "$#" -eq 0 ]]; then
  echo "usage: run-offline.sh COMMAND [ARG...]" >&2
  exit 2
fi

if ! command -v sandbox-exec >/dev/null 2>&1; then
  echo "run-offline.sh: sandbox-exec not found; offline command execution requires macOS" >&2
  exit 1
fi

if [[ ! -f "${SANDBOX_PROFILE}" ]]; then
  echo "run-offline.sh: sandbox profile not found at ${SANDBOX_PROFILE}" >&2
  exit 1
fi

if sandbox-exec -f "${SANDBOX_PROFILE}" \
    curl -fsS --max-time 10 https://example.com -o /dev/null 2>/dev/null; then
  echo "run-offline.sh: sandbox profile did not block network access; refusing to run" >&2
  exit 1
fi

echo "run-offline.sh: network egress is blocked; running: $*"
exec sandbox-exec -f "${SANDBOX_PROFILE}" env \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 \
  HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
  "$@"
