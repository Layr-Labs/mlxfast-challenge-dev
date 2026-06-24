#!/usr/bin/env bash
# Validate score and integrity artifacts before upload.
set -euo pipefail

SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
INTEGRITY_PATH="${MLXFAST_INTEGRITY_PATH:-benchmark-integrity.json}"
GOLDEN_PATH="${MLXFAST_CORRECTNESS_GOLDEN_PATH:-correctness_golden.json}"

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

: "${MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256:?MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256 is required}"
: "${MLXFAST_EXPECTED_CORRECTNESS_STEPS:?MLXFAST_EXPECTED_CORRECTNESS_STEPS is required}"
: "${MLXFAST_EXPECTED_CORRECTNESS_CASES:?MLXFAST_EXPECTED_CORRECTNESS_CASES is required}"
: "${MLXFAST_EXPECTED_CORRECTNESS_CHECKED_STEPS:?MLXFAST_EXPECTED_CORRECTNESS_CHECKED_STEPS is required}"

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
  --argjson checked_steps "${MLXFAST_EXPECTED_CORRECTNESS_CHECKED_STEPS}" \
  --argjson correctness_cases "${MLXFAST_EXPECTED_CORRECTNESS_CASES}" \
  '
  def same_keys($expected):
    (keys_unsorted | sort) == ($expected | sort);

  same_keys(["metrics", "passed", "score"])
  and (.metrics | same_keys([
    "actual_token",
    "bandwidth_gb_per_token",
    "bandwidth_source",
    "benchmark_wall_seconds",
    "case_count",
    "checked_steps",
    "commit",
    "correctness_seconds",
    "decode_seconds_per_token",
    "error",
    "expected_token",
    "expert_bytes_read",
    "expert_cache_evictions",
    "expert_cache_hits",
    "expert_cache_misses",
    "expert_hit_rate",
    "expert_peak_cached_tensors",
    "expert_read_seconds",
    "first_failing_case",
    "first_failing_layer",
    "first_failing_step",
    "golden_hash",
    "harness_hash",
    "max_abs_diff",
    "num_layers",
    "passed_correctness",
    "peak_ram_gb",
    "prefill_seconds_per_token",
    "preflight_seconds",
    "process_resident_memory_gb",
    "runtime",
    "timed_benchmark_seconds",
    "timestamp",
    "weights_byte_count",
    "weights_file_count",
    "weights_hash"
  ]))
  and .passed == true
  and (.score | type == "number")
  and (.score >= 0)
  and (.metrics.passed_correctness == true)
  and (.metrics.checked_steps == $checked_steps)
  and (.metrics.case_count == $correctness_cases)
  and (.metrics.num_layers == 43)
  and (.metrics.golden_hash == $golden_hash)
  and (.metrics.first_failing_case == null)
  and (.metrics.first_failing_layer == null)
  and (.metrics.first_failing_step == null)
  and (.metrics.expected_token == null)
  and (.metrics.actual_token == null)
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
  and (.metrics.bandwidth_source == "expert_streaming_reads" or .metrics.bandwidth_source == "mactop_hardware")
  and (.metrics.error == "")
  and (.metrics.commit | test("^[0-9a-f]{7,40}$"))
  and (.metrics.harness_hash | test("^[0-9a-f]{64}$"))
  and (.metrics.timestamp | test("^[0-9]{4}-[0-9]{2}-[0-9]{2}T"))
  and (.metrics.runtime == "swift")
  ' "${SCORE_PATH}" >/dev/null

case "${MLXFAST_REQUIRE_MACTOP_BANDWIDTH:-0}" in
  1|true|TRUE|yes|YES)
    bandwidth_source="$(jq -r '.metrics.bandwidth_source' "${SCORE_PATH}")"
    if [[ "${bandwidth_source}" != "mactop_hardware" ]]; then
      echo "::error file=${SCORE_PATH}::mactop hardware bandwidth was required, got ${bandwidth_source}" >&2
      exit 1
    fi
    ;;
esac

jq -e '
  def same_keys($expected):
    (keys_unsorted | sort) == ($expected | sort);

  same_keys([
    "golden_path",
    "golden_sha256",
    "score_path",
    "score_sha256",
    "transform_source_sha256",
    "weights_byte_count",
    "weights_file_count",
    "weights_path",
    "weights_sha256"
  ])
  and (.score_sha256 | test("^[0-9a-f]{64}$"))
  and (.golden_sha256 | test("^[0-9a-f]{64}$"))
  and (.weights_sha256 | test("^[0-9a-f]{64}$"))
  and (.transform_source_sha256 | test("^[0-9a-f]{64}$"))
  and (.weights_file_count | type == "number")
  and (.weights_file_count > 0)
  and (.weights_byte_count | type == "number")
  and (.weights_byte_count > 0)
  ' "${INTEGRITY_PATH}" >/dev/null

echo "benchmark: validated score and integrity artifacts"
