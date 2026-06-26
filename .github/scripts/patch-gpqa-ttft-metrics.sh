#!/usr/bin/env bash
# Patch aggregate hidden-GPQA TTFT metrics into score.json after benchmark timing.
set -euo pipefail

RESULTS_PATH="${MLXFAST_GPQA_TTFT_RESULTS_PATH:?MLXFAST_GPQA_TTFT_RESULTS_PATH is required}"
SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
INTEGRITY_PATH="${MLXFAST_INTEGRITY_PATH:-benchmark-integrity.json}"

if [[ ! -s "${RESULTS_PATH}" ]]; then
  echo "::error file=${RESULTS_PATH}::GPQA TTFT result file is missing or empty" >&2
  exit 1
fi
if [[ ! -s "${SCORE_PATH}" ]]; then
  echo "::error file=${SCORE_PATH}::score file is missing or empty" >&2
  exit 1
fi
if [[ ! -s "${INTEGRITY_PATH}" ]]; then
  echo "::error file=${INTEGRITY_PATH}::benchmark integrity file is missing or empty" >&2
  exit 1
fi

jq -e '
  (.passed == true)
  and (.source == "hidden_gpqa_first_token")
  and (.case_count | type == "number")
  and (.case_count > 0)
  and (.pass_count == .case_count)
  and (.mean_seconds | type == "number")
  and (.mean_seconds > 0)
  and (.p50_seconds | type == "number")
  and (.p50_seconds > 0)
  and (.max_seconds | type == "number")
  and (.max_seconds >= .p50_seconds)
' "${RESULTS_PATH}" >/dev/null

ttft_passed="$(jq -r '.passed' "${RESULTS_PATH}")"
ttft_pass_count="$(jq -r '.pass_count' "${RESULTS_PATH}")"
ttft_case_count="$(jq -r '.case_count' "${RESULTS_PATH}")"
ttft_seconds="$(jq -r '.mean_seconds' "${RESULTS_PATH}")"
ttft_p50_seconds="$(jq -r '.p50_seconds' "${RESULTS_PATH}")"
ttft_max_seconds="$(jq -r '.max_seconds' "${RESULTS_PATH}")"
ttft_source="$(jq -r '.source' "${RESULTS_PATH}")"

tmp_score="$(mktemp "${SCORE_PATH}.XXXXXX")"
jq \
  --argjson ttft_passed "${ttft_passed}" \
  --argjson ttft_pass_count "${ttft_pass_count}" \
  --argjson ttft_case_count "${ttft_case_count}" \
  --argjson ttft_seconds "${ttft_seconds}" \
  --argjson ttft_p50_seconds "${ttft_p50_seconds}" \
  --argjson ttft_max_seconds "${ttft_max_seconds}" \
  --arg ttft_source "${ttft_source}" \
  '
  .metrics.gpqa_ttft_passed = $ttft_passed
  | .metrics.gpqa_ttft_pass_count = $ttft_pass_count
  | .metrics.gpqa_ttft_case_count = $ttft_case_count
  | .metrics.gpqa_ttft_seconds = $ttft_seconds
  | .metrics.gpqa_ttft_p50_seconds = $ttft_p50_seconds
  | .metrics.gpqa_ttft_max_seconds = $ttft_max_seconds
  | .metrics.gpqa_ttft_source = $ttft_source
  ' "${SCORE_PATH}" > "${tmp_score}"
mv "${tmp_score}" "${SCORE_PATH}"

shasum -a 256 "${SCORE_PATH}" > "${SCORE_PATH}.sha256"
score_hash="$(shasum -a 256 "${SCORE_PATH}" | awk '{print $1}')"
tmp_integrity="$(mktemp "${INTEGRITY_PATH}.XXXXXX")"
jq --arg score_hash "${score_hash}" '.score_sha256 = $score_hash' \
  "${INTEGRITY_PATH}" > "${tmp_integrity}"
mv "${tmp_integrity}" "${INTEGRITY_PATH}"

echo "gpqa-ttft: passed ${ttft_pass_count}/${ttft_case_count} mean_seconds=${ttft_seconds}"
