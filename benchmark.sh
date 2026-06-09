#!/usr/bin/env bash
# Run a source-faithful benchmark and emit the benchmark.json scorePath.
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-${VENV_DIR}/bin/python}"
SCORE_PATH="${QUANTIZATIONFAIL_SCORE_PATH:-score.json}"
REFERENCE_DIR="quantizationfail/reference_weights/gemma-4-26B-A4B-it-qat-4bit"
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

if [[ ! -f "${REFERENCE_DIR}/config.json" ]]; then
  cat >&2 <<EOF
benchmark.sh: reference weights not found at ${REFERENCE_DIR}.
Run ./setup.sh, or set QUANTIZATIONFAIL_SKIP_WEIGHTS_DOWNLOAD=1 only after
placing the reference checkpoint there.
EOF
  exit 1
fi

mkdir -p weights
wanted_hash="$("${PYTHON}" "${BENCHMARK_HELPER}" source-hash)"
current_hash="$(cat "${SOURCE_HASH_PATH}" 2>/dev/null || true)"

if [[ "${QUANTIZATIONFAIL_FORCE_TRANSFORM:-0}" == "1" || ! -f weights/config.json || "${current_hash}" != "${wanted_hash}" ]]; then
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
  run_args+=(--note "${QUANTIZATIONFAIL_NOTE:-benchmark.json run}")
else
  run_args+=("$@")
fi
run_args+=(--score-path "${SCORE_PATH}")

"${PYTHON}" -m quantizationfail.cli "${run_args[@]}"

if [[ ! -s "${SCORE_PATH}" ]]; then
  echo "benchmark.sh: benchmark did not produce ${SCORE_PATH}" >&2
  exit 1
fi
