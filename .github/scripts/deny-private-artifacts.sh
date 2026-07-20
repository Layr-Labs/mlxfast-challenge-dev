#!/usr/bin/env bash
# Refuse artifact/cache uploads that could leak private prompts, golden tokens,
# model shards, symlinks, or unexpectedly large files.
set -euo pipefail

MAX_BYTES="${MLXFAST_MAX_ARTIFACT_BYTES:-1048576}"
ARTIFACT_PROFILE="${MLXFAST_ARTIFACT_PROFILE:-default}"

if [[ "$#" -eq 0 ]]; then
  echo "usage: deny-private-artifacts.sh PATH..." >&2
  exit 2
fi

fail() {
  local path="$1"
  local message="$2"
  if [[ "${MLXFAST_GITHUB_ANNOTATIONS:-1}" == "0" ]]; then
    echo "${path}: ${message}" >&2
  else
    echo "::error file=${path}::${message}" >&2
  fi
  exit 1
}

validate_single_json_document() {
  local path="$1"
  jq -s -e 'length == 1' "${path}" >/dev/null \
    || fail "${path}" "artifact JSON must contain exactly one document"
}

validate_mtp_raw_paths() {
  local path="$1"
  local profile="$2"
  local stream
  local duplicate

  stream="$(jq --stream -c 'select(length == 2) | .[0]' "${path}")" \
    || fail "${path}" "artifact JSON stream could not be parsed"
  while IFS= read -r json_path; do
    [[ -n "${json_path}" ]] || continue
    case "${profile}:${json_path}" in
      mtp-correctness:'["schema_version"]' \
        |mtp-correctness:'["track_id"]' \
        |mtp-correctness:'["gate_passed"]' \
        |mtp-correctness:'["token_count"]' \
        |mtp-score:'["score"]' \
        |mtp-score:'["passed"]' \
        |mtp-score:'["track_id"]' \
        |mtp-score:'["metrics","mode"]' \
        |mtp-score:'["metrics","aggregation"]' \
        |mtp-score:'["metrics","decode_tokens"]' \
        |mtp-score:'["metrics","decode_speedup_floor"]' \
        |mtp-score:'["metrics","accepted_pair_count"]' \
        |mtp-score:'["metrics","target_pair_count"]' \
        |mtp-score:'["metrics","parity_all_ok"]' \
        |mtp-score:'["metrics","mtp_decode_speedup_ratio_of_means"]')
        ;;
      *)
        fail "${path}" "artifact JSON contains a non-allowlisted raw key path: ${json_path}"
        ;;
    esac
  done <<< "${stream}"

  duplicate="$(
    printf '%s\n' "${stream}" \
      | LC_ALL=C sort \
      | uniq -d \
      | awk 'NR == 1 { print; exit }'
  )"
  [[ -z "${duplicate}" ]] \
    || fail "${path}" "artifact JSON contains a duplicate raw key path: ${duplicate}"
}

validate_mtp_correctness_verdict() {
  local path="$1"
  local track="${MLXFAST_EXPECTED_TRACK_ID:-}"
  local token_count="${MLXFAST_EXPECTED_TOKEN_COUNT:-}"
  [[ -n "${track}" && "${token_count}" =~ ^[0-9]+$ ]] \
    || fail "${path}" "MTP correctness verdict validation requires trusted track/token constants"
  validate_single_json_document "${path}"
  validate_mtp_raw_paths "${path}" mtp-correctness
  jq -e \
    --arg track "${track}" \
    --argjson token_count "${token_count}" \
    '
    type == "object"
    and ((keys | sort) == ["gate_passed", "schema_version", "token_count", "track_id"])
    and .schema_version == 1
    and .track_id == $track
    and .gate_passed == true
    and .token_count == $token_count
    ' "${path}" >/dev/null \
    || fail "${path}" "MTP correctness verdict must match the fixed trusted schema"
}

validate_mtp_score() {
  local path="$1"
  local track="${MLXFAST_EXPECTED_TRACK_ID:-}"
  local token_count="${MLXFAST_EXPECTED_TOKEN_COUNT:-}"
  local floor="${MLXFAST_EXPECTED_SCORE_FLOOR:-}"
  local min_pairs="${MLXFAST_EXPECTED_MIN_PAIRS:-}"
  local target_pairs="${MLXFAST_EXPECTED_TARGET_PAIRS:-}"
  [[ -n "${track}" \
      && "${token_count}" =~ ^[0-9]+$ \
      && "${floor}" =~ ^[0-9]+([.][0-9]+)?$ \
      && "${min_pairs}" =~ ^[0-9]+$ \
      && "${target_pairs}" =~ ^[0-9]+$ ]] \
    || fail "${path}" "MTP score validation requires trusted track/scoring constants"
  validate_single_json_document "${path}"
  validate_mtp_raw_paths "${path}" mtp-score
  jq -e \
    --arg track "${track}" \
    --argjson token_count "${token_count}" \
    --argjson floor "${floor}" \
    --argjson min_pairs "${min_pairs}" \
    --argjson target_pairs "${target_pairs}" \
    '
    type == "object"
    and ((keys | sort) == ["metrics", "passed", "score", "track_id"])
    and (.score | type) == "number"
    and (.score | isnan | not)
    and (.score | isinfinite | not)
    and .score >= $floor
    and .passed == true
    and .track_id == $track
    and (.metrics | type) == "object"
    and ((.metrics | keys | sort) == [
      "accepted_pair_count",
      "aggregation",
      "decode_speedup_floor",
      "decode_tokens",
      "mode",
      "mtp_decode_speedup_ratio_of_means",
      "parity_all_ok",
      "target_pair_count"
    ])
    and .metrics.mode == "mtp-paired-decode-only"
    and .metrics.aggregation == "ratio_of_means"
    and .metrics.decode_tokens == $token_count
    and .metrics.decode_speedup_floor == $floor
    and (.metrics.accepted_pair_count | type) == "number"
    and (.metrics.accepted_pair_count | floor) == .metrics.accepted_pair_count
    and .metrics.accepted_pair_count >= $min_pairs
    and .metrics.accepted_pair_count <= $target_pairs
    and .metrics.target_pair_count == $target_pairs
    and .metrics.parity_all_ok == true
    and (.metrics.mtp_decode_speedup_ratio_of_means | isnan | not)
    and (.metrics.mtp_decode_speedup_ratio_of_means | isinfinite | not)
    and .metrics.mtp_decode_speedup_ratio_of_means == .score
    ' "${path}" >/dev/null \
    || fail "${path}" "MTP score must match the minimized trusted schema"
}

check_file() {
  local path="$1"
  local base
  local size

  [[ -e "${path}" ]] || return 0

  if [[ -L "${path}" ]]; then
    fail "${path}" "artifact candidate must not be a symlink"
  fi
  if [[ -d "${path}" ]]; then
    while IFS= read -r -d '' child; do
      check_file "${child}"
    done < <(find "${path}" -mindepth 1 -print0)
    return 0
  fi
  if [[ ! -f "${path}" ]]; then
    fail "${path}" "artifact candidate must be a regular file"
  fi

  case "${path}" in
    */mlxfast-mtp-measure-*/results.json|*/mlxfast-mtp-measure-*/verdict.txt)
      fail "${path}" "sealed MTP measure outputs must remain runner-private"
      ;;
  esac

  base="$(basename "${path}")"
  case "${base}" in
    *correctness_golden*.json|*golden_prompt*.json|*golden_prompt*.txt|*private_prompts*.json|*gpqa_reference_cases*.json|*gpqa_ttft_results*.json|*.safetensors|*.bin|*.gguf)
      fail "${path}" "private prompt, golden, or model files must not be uploaded or cached"
      ;;
    mtp-gates-report.json|mtp-gates-stdout.raw|mtp-gates-private.log|mtp-paired-results.json|mtp-measure-verdict.txt)
      fail "${path}" "raw MTP reports, diagnostics, and measure verdicts must remain runner-private"
      ;;
  esac

  case "${ARTIFACT_PROFILE}" in
    default)
      ;;
    mtp-correctness)
      [[ "${base}" == "mtp-correctness-verdict.json" ]] \
        || fail "${path}" "MTP correctness artifacts allow only the fixed verdict"
      validate_mtp_correctness_verdict "${path}"
      ;;
    mtp-score)
      [[ "${base}" == "score.json" ]] \
        || fail "${path}" "MTP benchmark artifacts allow only the minimized score"
      validate_mtp_score "${path}"
      ;;
    *)
      fail "${path}" "unknown MLXFAST_ARTIFACT_PROFILE=${ARTIFACT_PROFILE}"
      ;;
  esac

  size="$(wc -c < "${path}" | tr -d ' ')"
  if [[ ! "${size}" =~ ^[0-9]+$ ]]; then
    fail "${path}" "could not determine artifact candidate size"
  fi
  if (( size > MAX_BYTES )); then
    fail "${path}" "artifact candidate is ${size} bytes, above MLXFAST_MAX_ARTIFACT_BYTES=${MAX_BYTES}"
  fi
}

for path in "$@"; do
  check_file "${path}"
done
