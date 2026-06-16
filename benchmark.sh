#!/usr/bin/env bash
# Run a source-faithful benchmark and emit the benchmark.json scorePath.
set -euo pipefail

# The benchmark always runs offline. Unless already inside the sandbox, prove
# the Seatbelt profile blocks egress, then re-exec under sandbox-exec (no sudo
# needed) so transform.py and the harness never see the network — locally or
# in CI. The proxy vars point at a closed local port so anything that ignores
# the profile fails fast instead of hanging.
SANDBOX_PROFILE="${MLXFAST_SANDBOX_PROFILE:-tools/deny-network.sb}"

if [[ "${MLXFAST_IN_SANDBOX:-0}" != "1" && "${MLXFAST_NO_SANDBOX:-0}" != "1" ]]; then
  if ! command -v sandbox-exec >/dev/null 2>&1; then
    echo "benchmark.sh: sandbox-exec not found (the benchmark requires macOS)." >&2
    echo "Set MLXFAST_NO_SANDBOX=1 to skip the offline sandbox; scores" >&2
    echo "produced that way are not comparable to sandboxed runs." >&2
    exit 1
  fi
  if sandbox-exec -f "${SANDBOX_PROFILE}" \
      curl -fsS --max-time 10 https://example.com -o /dev/null 2>/dev/null; then
    echo "benchmark.sh: sandbox-exec did not block network access; refusing to run" >&2
    exit 1
  fi
  echo "benchmark.sh: network egress is blocked; re-running inside the sandbox"
  exec sandbox-exec -f "${SANDBOX_PROFILE}" env \
    MLXFAST_IN_SANDBOX=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 \
    HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
    "$0" "$@"
fi

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-${VENV_DIR}/bin/python}"
SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
REFERENCE_DIR="mlxfast/reference_weights/DeepSeek-V4-Flash-4bit"
SOURCE_HASH_PATH="weights/.benchmark-source.sha256"
BENCHMARK_HELPER="tools/benchmark_contract.py"

resolve_python() {
  if [[ "${PYTHON}" == */* ]]; then
    [[ -x "${PYTHON}" ]] && printf '%s\n' "${PYTHON}"
    return 0
  fi

  command -v "${PYTHON}" 2>/dev/null || true
}

resolved_python="$(resolve_python)"
if [[ -z "${resolved_python}" ]]; then
  echo "benchmark.sh: ${PYTHON} not found; run ./setup.sh first" >&2
  exit 1
fi
PYTHON="${resolved_python}"

if [[ ! -f "${BENCHMARK_HELPER}" ]]; then
  echo "benchmark.sh: missing ${BENCHMARK_HELPER}" >&2
  exit 1
fi

mkdir -p weights
wanted_hash="$("${PYTHON}" "${BENCHMARK_HELPER}" source-hash)"
current_hash="$(cat "${SOURCE_HASH_PATH}" 2>/dev/null || true)"

if [[ "${MLXFAST_FORCE_TRANSFORM:-0}" == "1" || ! -f weights/config.json || "${current_hash}" != "${wanted_hash}" ]]; then
  # Only a (re)transform needs the reference checkpoint. When weights/ is
  # already present with a matching source hash -- e.g. restored from cache on
  # the benchmark runner in the split CI pipeline -- we skip this branch
  # entirely and never require the reference.
  if [[ ! -f "${REFERENCE_DIR}/config.json" ]]; then
    cat >&2 <<EOF
benchmark.sh: reference weights not found at ${REFERENCE_DIR}, needed to (re)run transform.py.
Run ./setup.sh, or set MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 only after placing the reference checkpoint there.
(If you expected cached weights/, the source hash did not match -- transform.py changed since the cache was built.)
EOF
    exit 1
  fi
  echo "benchmark.sh: regenerating weights from transform.py"
  find weights -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
  "${PYTHON}" transform.py
  if [[ ! -f weights/config.json ]]; then
    echo "benchmark.sh: transform.py did not produce weights/config.json" >&2
    exit 1
  fi
  printf '%s\n' "${wanted_hash}" > "${SOURCE_HASH_PATH}"
else
  echo "benchmark.sh: reusing weights/ for unchanged participant source"
fi

rm -f "${SCORE_PATH}"

run_args=(run --skip-transform-verify)
if [[ "$#" -eq 0 ]]; then
  run_args+=(--note "${MLXFAST_NOTE:-benchmark.json run}")
else
  run_args+=("$@")
fi
run_args+=(--score-path "${SCORE_PATH}")

"${PYTHON}" -m mlxfast.cli "${run_args[@]}"

if [[ ! -s "${SCORE_PATH}" ]]; then
  echo "benchmark.sh: benchmark did not produce ${SCORE_PATH}" >&2
  exit 1
fi
