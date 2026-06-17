#!/usr/bin/env bash
# Run the Swift benchmark and emit the benchmark.json scorePath.
set -euo pipefail

# The benchmark always runs offline. Unless already inside the sandbox, prove
# the Seatbelt profile blocks egress, then re-exec under sandbox-exec (no sudo
# needed) so transform/runtime code never sees the network — locally or
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

SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-weights}"
GOLDEN_PATH="${MLXFAST_CORRECTNESS_GOLDEN_PATH:-correctness_golden.json}"
REFERENCE_PATH="${MLXFAST_REFERENCE_DIR:-reference_weights/DeepSeek-V4-Flash-4bit}"
SWIFT_BIN="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
MLX_METALLIB="${MLXFAST_MLX_METALLIB:-$(dirname "${SWIFT_BIN}")/mlx.metallib}"

if [[ ! -x "${SWIFT_BIN}" ]]; then
  echo "benchmark.sh: Swift release binary missing; building"
  mkdir -p .build/clang-module-cache
  export CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-${PWD}/.build/clang-module-cache}"
  swift build -c release
fi

if [[ ! -f "${MLX_METALLIB}" ]]; then
  echo "benchmark.sh: MLX metallib missing at ${MLX_METALLIB}; run ./setup.sh before ranked benchmark runs" >&2
fi

if [[ "${MLXFAST_FORCE_TRANSFORM:-0}" == "1" || ! -f "${WEIGHTS_PATH}/config.json" ]]; then
  if [[ -f "${REFERENCE_PATH}/config.json" ]]; then
    echo "benchmark.sh: regenerating weights with Swift transform"
    if ! "${SWIFT_BIN}" transform --reference "${REFERENCE_PATH}" --output "${WEIGHTS_PATH}"; then
      echo "benchmark.sh: Swift transform failed; benchmark will emit a failed score" >&2
    fi
  else
    echo "benchmark.sh: reference weights missing at ${REFERENCE_PATH}; benchmark will emit a failed score" >&2
  fi
else
  echo "benchmark.sh: reusing ${WEIGHTS_PATH}/"
fi

rm -f "${SCORE_PATH}"

"${SWIFT_BIN}" benchmark \
  --weights "${WEIGHTS_PATH}" \
  --golden "${GOLDEN_PATH}" \
  --score-path "${SCORE_PATH}" \
  "$@"

if [[ ! -s "${SCORE_PATH}" ]]; then
  echo "benchmark.sh: benchmark did not produce ${SCORE_PATH}" >&2
  exit 1
fi
