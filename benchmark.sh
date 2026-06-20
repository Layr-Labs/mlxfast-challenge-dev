#!/usr/bin/env bash
# Run the Swift benchmark and emit the benchmark.json scorePath.
set -euo pipefail

SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-weights}"
GOLDEN_PATH="${MLXFAST_CORRECTNESS_GOLDEN_PATH:-correctness_golden.json}"
REFERENCE_PATH="${MLXFAST_REFERENCE_DIR:-reference_weights/DeepSeek-V4-Flash-4bit}"
SWIFT_BIN="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
MLX_METALLIB="${MLXFAST_MLX_METALLIB:-$(dirname "${SWIFT_BIN}")/mlx.metallib}"
SANDBOX_PROFILE="${MLXFAST_SANDBOX_PROFILE:-tools/deny-network.sb}"
SOURCE_HASH_PATH="${WEIGHTS_PATH}/.benchmark-source.sha256"

source_hash() {
  local paths=(
    "Package.swift"
    "Package.resolved"
    "Sources/MLXFastCore"
    "Sources/MLXFastTransform"
  )

  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git ls-files -z "${paths[@]}" | while IFS= read -r -d '' path; do
      if [[ -f "${path}" ]]; then
        printf '%s\0' "${path}"
        shasum -a 256 "${path}"
      else
        printf '%s\0MISSING\0' "${path}"
      fi
    done | shasum -a 256 | awk '{print $1}'
    return 0
  fi

  find "${paths[@]}" -type f 2>/dev/null | LC_ALL=C sort | while IFS= read -r path; do
    printf '%s\0' "${path}"
    shasum -a 256 "${path}"
  done | shasum -a 256 | awk '{print $1}'
}

clear_weights_dir() {
  case "${WEIGHTS_PATH}" in
    ""|"/")
      echo "benchmark.sh: refusing to clear unsafe weights path '${WEIGHTS_PATH}'" >&2
      exit 1
      ;;
  esac
  mkdir -p "${WEIGHTS_PATH}"
  find "${WEIGHTS_PATH}" -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
}

if [[ "${MLXFAST_IN_SANDBOX:-0}" != "1" && ! -x "${SWIFT_BIN}" ]]; then
  echo "benchmark.sh: Swift release binary missing; building"
  mkdir -p .build/clang-module-cache
  export CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-${PWD}/.build/clang-module-cache}"
  swift build -c release
fi

# The benchmark runtime always runs offline. Unless already inside the sandbox,
# prove the Seatbelt profile blocks egress, then re-exec under sandbox-exec (no
# sudo needed) so transform/runtime code never sees the network — locally or in
# CI. The proxy vars point at a closed local port so anything that ignores the
# profile fails fast instead of hanging.
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

if [[ ! -x "${SWIFT_BIN}" ]]; then
  echo "benchmark.sh: Swift release binary missing at ${SWIFT_BIN}" >&2
  exit 1
fi

if [[ ! -f "${MLX_METALLIB}" ]]; then
  echo "benchmark.sh: MLX metallib missing at ${MLX_METALLIB}; run ./setup.sh before ranked benchmark runs" >&2
fi

mkdir -p "${WEIGHTS_PATH}"
wanted_hash="$(source_hash)"
current_hash="$(cat "${SOURCE_HASH_PATH}" 2>/dev/null || true)"

if [[ "${MLXFAST_FORCE_TRANSFORM:-0}" == "1" || ! -f "${WEIGHTS_PATH}/config.json" || "${current_hash}" != "${wanted_hash}" ]]; then
  if [[ -f "${REFERENCE_PATH}/config.json" ]]; then
    echo "benchmark.sh: regenerating weights with Swift transform"
    clear_weights_dir
    "${SWIFT_BIN}" transform --reference "${REFERENCE_PATH}" --output "${WEIGHTS_PATH}"
    if [[ ! -f "${WEIGHTS_PATH}/config.json" ]]; then
      echo "benchmark.sh: Swift transform did not produce ${WEIGHTS_PATH}/config.json" >&2
      exit 1
    fi
    printf '%s\n' "${wanted_hash}" > "${SOURCE_HASH_PATH}"
  else
    cat >&2 <<EOF
benchmark.sh: reference weights not found at ${REFERENCE_PATH}, needed to regenerate weights/.
Run ./setup.sh, or set MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 only after placing the reference checkpoint there.
(If you expected cached weights/, the transform source hash did not match.)
EOF
    exit 1
  fi
else
  echo "benchmark.sh: reusing ${WEIGHTS_PATH}/ for unchanged transform source"
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
