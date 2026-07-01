#!/usr/bin/env bash
# Combine results from a 4-machine parallel run into one authoritative score.json:
#   machine1/ - ran `mlxfast-swift benchmark` with MLXFAST_BENCHMARK_CORRECTNESS_STEPS=0
#               (GPQA cases + GPQA TTFT + timed prefill/decode; skips its own base
#               teacher-forced correctness case)
#   machine2/ .. machineN/ - each ran `mlxfast-swift correctness --step-range START-END`
#               covering a disjoint slice of the base correctness case's step window
#
# This script is the only place the real, combined correctness verdict is asserted.
# Machine 1's own score.json reports passed_correctness based on nothing but GPQA/TTFT/
# floor checks -- it must never be published as-is. This script ANDs in the base-case
# verdict from every correctness-only machine before anything gets staged or uploaded.
set -euo pipefail

: "${MLXFAST_EXPECTED_CORRECTNESS_STEPS:?MLXFAST_EXPECTED_CORRECTNESS_STEPS is required}"
: "${MLXFAST_COMBINED_SCORE_PATH:=score.combined.json}"

MACHINE1_DIR="${MLXFAST_MACHINE1_DIR:-machine1}"
CORRECTNESS_MACHINE_DIRS=(${MLXFAST_CORRECTNESS_MACHINE_DIRS:-machine2 machine3 machine4})

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "::error file=${path}::required combiner input is missing or empty" >&2
    exit 1
  fi
}

require_file "${MACHINE1_DIR}/score.json"
require_file "${MACHINE1_DIR}/weights.sha256"

expected_weights_hash="$(cat "${MACHINE1_DIR}/weights.sha256")"

# --- Tripwire: every machine must have independently transformed byte-identical
# weights/. Since each machine downloads and transforms on its own (no cross-machine
# artifact handoff), a mismatch here means the checkpoint version, submitted commit, or
# transform itself diverged between machines -- treat that as fatal, not a warning.
for dir in "${MACHINE1_DIR}" "${CORRECTNESS_MACHINE_DIRS[@]}"; do
  require_file "${dir}/weights.sha256"
  actual="$(cat "${dir}/weights.sha256")"
  if [[ "${actual}" != "${expected_weights_hash}" ]]; then
    echo "::error file=${dir}/weights.sha256::weights hash mismatch across machines" >&2
    echo "expected (from ${MACHINE1_DIR})=${expected_weights_hash}" >&2
    echo "actual (from ${dir})=${actual}" >&2
    exit 1
  fi
done
echo "combine-parallel-correctness: weights hash agrees across all machines: ${expected_weights_hash}"

# --- Combine the base correctness case's step-range slices.
total_checked_steps=0
base_case_passed=true
first_failing_step=""
first_failing_dir=""

for dir in "${CORRECTNESS_MACHINE_DIRS[@]}"; do
  require_file "${dir}/correctness-report.json"
  report="${dir}/correctness-report.json"

  passed="$(jq -r '.passed' "${report}")"
  checked_steps="$(jq -r '.checked_steps' "${report}")"
  failing_step="$(jq -r '.first_failing_step // empty' "${report}")"

  total_checked_steps=$((total_checked_steps + checked_steps))

  if [[ "${passed}" != "true" ]]; then
    base_case_passed=false
    if [[ -n "${failing_step}" ]] \
      && { [[ -z "${first_failing_step}" ]] || (( failing_step < first_failing_step )); }; then
      first_failing_step="${failing_step}"
      first_failing_dir="${dir}"
    fi
    echo "combine-parallel-correctness: ${dir} FAILED at step ${failing_step:-unknown}" >&2
  else
    echo "combine-parallel-correctness: ${dir} passed, checked_steps=${checked_steps}"
  fi
done

# --- Coverage check: when every machine reports its own slice passed, those slices
# must still partition [0, EXPECTED) with no gaps or overlaps -- otherwise a
# misconfigured range assignment (e.g. a 9-step gap nobody was assigned to check) would
# report "all passed" despite part of the window never having been verified at all.
# Only enforced on the all-passed path: a real failure legitimately stops early and
# reports fewer checked_steps than its assigned slice, which is expected, not a
# misconfiguration, and must not be confused with a coverage gap.
if [[ "${base_case_passed}" == "true" && "${total_checked_steps}" -ne "${MLXFAST_EXPECTED_CORRECTNESS_STEPS}" ]]; then
  echo "::error::all machines reported passed, but combined checked_steps=${total_checked_steps} does not equal expected ${MLXFAST_EXPECTED_CORRECTNESS_STEPS}" >&2
  echo "this means the per-machine --step-range assignments overlap or leave a gap; fix the range assignment, do not adjust this check" >&2
  base_case_passed=false
  first_failing_dir="range-coverage-check"
fi

# --- Assemble the final score.json: take machine1's payload (score, GPQA, TTFT, speedup
# floors) and override the correctness fields with the real, combined verdict.
# machine1's own metrics.checked_steps only covers what it actually checked itself
# (anchors/free-run/behavior -- it skipped the base case), so the combined total adds
# the base-case step count on top of that rather than replacing it.
jq -e \
  --argjson base_case_passed "${base_case_passed}" \
  --argjson base_case_checked_steps "${total_checked_steps}" \
  --arg first_failing_step "${first_failing_step}" \
  --arg first_failing_dir "${first_failing_dir}" \
  '
  (if $first_failing_step == "" then null else ($first_failing_step | tonumber) end) as $ffs
  | (.metrics.passed_correctness and $base_case_passed) as $combined_passed_correctness
  | .metrics.passed_correctness = $combined_passed_correctness
  | .metrics.checked_steps = (.metrics.checked_steps + $base_case_checked_steps)
  | .metrics.first_failing_case = (if $base_case_passed then .metrics.first_failing_case else "base_correctness_case (\($first_failing_dir))" end)
  | .metrics.first_failing_step = (if $base_case_passed then .metrics.first_failing_step else $ffs end)
  | .passed = (.passed and $combined_passed_correctness)
  | if .passed then . else (.score = null) end
  ' "${MACHINE1_DIR}/score.json" > "${MLXFAST_COMBINED_SCORE_PATH}"

combined_passed="$(jq -r '.passed' "${MLXFAST_COMBINED_SCORE_PATH}")"
echo "combine-parallel-correctness: wrote ${MLXFAST_COMBINED_SCORE_PATH} passed=${combined_passed}"

if [[ "${combined_passed}" != "true" ]]; then
  exit 1
fi
