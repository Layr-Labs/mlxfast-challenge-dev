#!/usr/bin/env bash
# Validate score and integrity artifacts before upload.
set -euo pipefail

SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
INTEGRITY_PATH="${MLXFAST_INTEGRITY_PATH:-benchmark-integrity.json}"
GOLDEN_PATH="${MLXFAST_CORRECTNESS_GOLDEN_PATH:-correctness_golden.json}"
: "${MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256:?MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256 is required}"
: "${MLXFAST_EXPECTED_CORRECTNESS_STEPS:?MLXFAST_EXPECTED_CORRECTNESS_STEPS is required}"
: "${MLXFAST_EXPECTED_CORRECTNESS_CASES:?MLXFAST_EXPECTED_CORRECTNESS_CASES is required}"

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "::error file=${path}::required benchmark artifact is missing or empty" >&2
    exit 1
  fi
}

require_file "${SCORE_PATH}"
require_file "${SCORE_PATH}.sha256"
require_file "${INTEGRITY_PATH}"
require_file "${GOLDEN_PATH}.sha256"
require_file "${GOLDEN_PATH}.bytes"

shasum -a 256 -c "${SCORE_PATH}.sha256"

score_hash="$(shasum -a 256 "${SCORE_PATH}" | awk '{print $1}')"
integrity_score_hash="$(jq -r '.score_sha256 // empty' "${INTEGRITY_PATH}")"
if [[ "${integrity_score_hash}" != "${score_hash}" ]]; then
  echo "::error file=${INTEGRITY_PATH}::integrity score hash does not match ${SCORE_PATH}" >&2
  exit 1
fi

golden_hash="$(awk '{print $1}' "${GOLDEN_PATH}.sha256")"
if [[ "${golden_hash}" != "${MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256}" ]]; then
  echo "::error file=${GOLDEN_PATH}.sha256::golden hash mismatch" >&2
  exit 1
fi

jq -e \
  --arg golden_hash "${MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256}" \
  --argjson correctness_steps "${MLXFAST_EXPECTED_CORRECTNESS_STEPS}" \
  --argjson correctness_cases "${MLXFAST_EXPECTED_CORRECTNESS_CASES}" \
  '
  .passed == true
  and (.score | type == "number")
  and (.score >= 0)
  and (.metrics.passed_correctness == true)
  and (.metrics.checked_steps == $correctness_steps)
  and (.metrics.case_count == $correctness_cases)
  and (.metrics.num_layers == 43)
  and (.metrics.golden_hash == $golden_hash)
  and (.metrics.decode_seconds_per_token | type == "number")
  and (.metrics.decode_seconds_per_token > 0)
  and (.metrics.prefill_seconds_per_token | type == "number")
  and (.metrics.prefill_seconds_per_token > 0)
  and (.metrics.correctness_seconds | type == "number")
  and (.metrics.correctness_seconds > 0)
  and (.metrics.timed_benchmark_seconds | type == "number")
  and (.metrics.timed_benchmark_seconds > 0)
  and (.metrics.benchmark_wall_seconds | type == "number")
  and (.metrics.benchmark_wall_seconds >= .metrics.timed_benchmark_seconds)
  and (.metrics.weights_hash | test("^[0-9a-f]{64}$"))
  and (.metrics.weights_file_count | type == "number")
  and (.metrics.weights_file_count > 0)
  and (.metrics.weights_byte_count | type == "number")
  and (.metrics.weights_byte_count > 0)
  and (.metrics.bandwidth_source | type == "string")
  and (.metrics.bandwidth_source | length > 0)
  and (.metrics.runtime == "swift")
  ' "${SCORE_PATH}" >/dev/null

echo "benchmark: validated score and integrity artifacts"
