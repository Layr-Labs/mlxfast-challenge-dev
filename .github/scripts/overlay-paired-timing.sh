#!/usr/bin/env bash
# Overlay the measure-job paired timing into the sealed gates score -- the
# single-machine replacement for the old combine job's "Merge gates and timing
# into machine1" step plus the timing machine's harness-side floor check.
#
# Inputs (all runner-owned, produced earlier in the same job):
#   MLXFAST_SCORE_PATH             sealed gates+semantic score.json (base case
#                                  + gates verdicts; timing fields are the
#                                  skip-timed baseline placeholders)
#   MLXFAST_MEASURE_RESULTS_PATH   measure-job results.json (paired mode:
#                                  candidate + pinned-baseline seconds/token,
#                                  speedups, paired_score, telemetry stats)
#   MLXFAST_CANDIDATE_SCORE_PATH   measure-job's sealed candidate score.json
#                                  (harness-authored timing diagnostics:
#                                  timed_benchmark_seconds, wall seconds,
#                                  bandwidth fields -- already coarsened by the
#                                  harness where applicable)
#   MLXFAST_INTEGRITY_PATH         harness integrity record; its score_sha256
#                                  is re-anchored to the merged score
#   MLXFAST_DECODE_SPEEDUP_FLOOR / MLXFAST_PREFILL_SPEEDUP_FLOOR
#                                  keep in sync with MLXFastConstants.{score
#                                  DecodeSpeedupFloor,scorePrefillSpeedupFloor}
#   MLXFAST_CHAMPION_DECODE_SPEEDUP / MLXFAST_CHAMPION_PREFILL_SPEEDUP
#                                  OPTIONAL advancing acceptance-band reference
#                                  (see docs/benchmark-window-freeze.md,
#                                  "Advancing acceptance band"): the CHAMPION's
#                                  (current best accepted submission's) paired
#                                  speedups vs the same pinned baseline,
#                                  recorded by the maintainer from its accepted
#                                  run. Because candidate and champion speedups
#                                  both reference the pinned baseline measured
#                                  in-session behind the thermal gate, the
#                                  candidate/champion seconds ratio
#                                  (= champion_speedup / candidate_speedup)
#                                  cancels common-mode host drift -- the
#                                  single-machine equivalent of the old
#                                  fleet pipeline's live champion re-measure.
#                                  Both-or-neither: a half-set pair fails fast
#                                  (champion-advance protocol sets both
#                                  together). Unset -> the band gate here is
#                                  skipped and the harness's in-process band
#                                  vs the frozen reference remains the only
#                                  band (pre-champion behavior).
#   MLXFAST_DECODE_BAND_UP_TOLERANCE / MLXFAST_DECODE_BAND_DOWN_TOLERANCE /
#   MLXFAST_PREFILL_BAND_UP_TOLERANCE / MLXFAST_PREFILL_BAND_DOWN_TOLERANCE
#                                  required with the champion pair; keep in
#                                  sync with MLXFastConstants.{decode,prefill}
#                                  Band{Up,Down}Tolerance
#
# Verdict semantics (identical to the old pipeline):
#   .passed = gates_score.passed AND both speedup floors AND (when a champion
#             is recorded) both acceptance bands vs the champion
#   .score  = decode_speedup^0.75 * prefill_speedup^0.25 when passed, null
#             otherwise (with metrics.error set to the harness's own
#             "performance floor failed" / "acceptance band failed" prefixes
#             so redact-benchmark-failure.sh categorizes it identically).
# Exits nonzero when the merged verdict is a failure, so the workflow job
# fails exactly where the old timing machine's benchmark.sh did.
set -euo pipefail

SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
RESULTS_PATH="${MLXFAST_MEASURE_RESULTS_PATH:?MLXFAST_MEASURE_RESULTS_PATH is required}"
CANDIDATE_SCORE_PATH="${MLXFAST_CANDIDATE_SCORE_PATH:?MLXFAST_CANDIDATE_SCORE_PATH is required}"
INTEGRITY_PATH="${MLXFAST_INTEGRITY_PATH:-benchmark-integrity.json}"
DECODE_FLOOR="${MLXFAST_DECODE_SPEEDUP_FLOOR:?MLXFAST_DECODE_SPEEDUP_FLOOR is required}"
PREFILL_FLOOR="${MLXFAST_PREFILL_SPEEDUP_FLOOR:?MLXFAST_PREFILL_SPEEDUP_FLOOR is required}"

# Advancing acceptance-band reference (the champion). Both-or-neither: the
# champion-advance protocol records both axes together, and silently gating
# against a half-set reference would misprice one axis, so fail fast instead.
CHAMPION_DECODE_SPEEDUP="${MLXFAST_CHAMPION_DECODE_SPEEDUP:-}"
CHAMPION_PREFILL_SPEEDUP="${MLXFAST_CHAMPION_PREFILL_SPEEDUP:-}"
if [[ -n "${CHAMPION_DECODE_SPEEDUP}" || -n "${CHAMPION_PREFILL_SPEEDUP}" ]]; then
  if [[ -z "${CHAMPION_DECODE_SPEEDUP}" || -z "${CHAMPION_PREFILL_SPEEDUP}" ]]; then
    echo "::error::MLXFAST_CHAMPION_DECODE_SPEEDUP and MLXFAST_CHAMPION_PREFILL_SPEEDUP must be set together (champion-advance protocol, docs/benchmark-window-freeze.md); refusing to gate against a half-set champion reference" >&2
    exit 1
  fi
  for value in "${CHAMPION_DECODE_SPEEDUP}" "${CHAMPION_PREFILL_SPEEDUP}"; do
    jq -ne --arg v "${value}" '($v | tonumber) as $n | $n > 0 and ($n | isinfinite | not) and ($n | isnan | not)' >/dev/null 2>&1 || {
      echo "::error::champion speedup '${value}' is not a finite positive number" >&2
      exit 1
    }
  done
  DECODE_BAND_UP="${MLXFAST_DECODE_BAND_UP_TOLERANCE:?MLXFAST_DECODE_BAND_UP_TOLERANCE is required with a champion reference}"
  DECODE_BAND_DOWN="${MLXFAST_DECODE_BAND_DOWN_TOLERANCE:?MLXFAST_DECODE_BAND_DOWN_TOLERANCE is required with a champion reference}"
  PREFILL_BAND_UP="${MLXFAST_PREFILL_BAND_UP_TOLERANCE:?MLXFAST_PREFILL_BAND_UP_TOLERANCE is required with a champion reference}"
  PREFILL_BAND_DOWN="${MLXFAST_PREFILL_BAND_DOWN_TOLERANCE:?MLXFAST_PREFILL_BAND_DOWN_TOLERANCE is required with a champion reference}"
else
  DECODE_BAND_UP="0"
  DECODE_BAND_DOWN="0"
  PREFILL_BAND_UP="0"
  PREFILL_BAND_DOWN="0"
fi

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "::error file=$1::required overlay input is missing or empty" >&2
    exit 1
  fi
}
require_file "${SCORE_PATH}"
require_file "${RESULTS_PATH}"
require_file "${CANDIDATE_SCORE_PATH}"
require_file "${INTEGRITY_PATH}"

# The paired results must be from a real paired run that measure-job ACCEPTed
# on both sides (its exit code already guaranteed that; assert the shape).
jq -e '
  .mode == "paired"
  and (.paired.decode_speedup | type == "number") and (.paired.decode_speedup > 0)
  and (.paired.prefill_speedup | type == "number") and (.paired.prefill_speedup > 0)
  and (.candidate.decode_seconds_per_token > 0)
  and (.candidate.prefill_seconds_per_token > 0)
  and (.baseline.decode_seconds_per_token > 0)
  and (.baseline.prefill_seconds_per_token > 0)
  and (.candidate.verdict == "ACCEPT")
  and (.baseline.verdict == "ACCEPT")
' "${RESULTS_PATH}" >/dev/null || {
  echo "::error file=${RESULTS_PATH}::paired results are not a clean ACCEPTed paired measurement" >&2
  exit 1
}

# Defense-in-depth on the candidate's sealed timing score, mirroring the old
# combine job's pre-merge leak-field and expert-zero checks (the dense
# RAM-resident runtime must report structural zeros; a nonzero value is a
# schema regression that must fail loudly, not merge silently).
jq -e '
  (.metrics.first_failing_case == null)
  and (.metrics.first_failing_step == null)
  and (.metrics.expected_token == null)
  and (.metrics.actual_token == null)
  and (.metrics.expert_bytes_read == 0)
  and (.metrics.expert_cache_hits == 0)
  and (.metrics.expert_cache_misses == 0)
  and (.metrics.expert_cache_evictions == 0)
  and (.metrics.expert_read_seconds == 0)
  and (.metrics.expert_peak_cached_tensors == 0)
  and (.metrics.bandwidth_source == "ram_resident_model")
  and (.metrics.timed_benchmark_seconds > 0)
' "${CANDIDATE_SCORE_PATH}" >/dev/null || {
  echo "::error file=${CANDIDATE_SCORE_PATH}::candidate timing score failed the pre-merge checks" >&2
  exit 1
}

# Base the merged score on the gates score ($g) and overlay only the
# timing-derived fields, with the same provenance rules the old combine used:
#   - decode/prefill seconds per token: the candidate's measured timing.
#   - baseline_*: the pinned baseline's MEASURED same-session timing (the old
#     pipeline injected these via MLXFAST_PAIRED_BASELINE_*; the harness then
#     published them in exactly these fields).
#   - speedups/score: the paired ratio (ranking fields, full precision).
#   - floors: applied here in the trusted shell; the old timing machine's
#     harness applied the identical constants in-process.
#   - timed_benchmark_seconds / bandwidth_*: candidate harness diagnostics
#     (already coarsened by the harness).
#   - benchmark_wall_seconds, peak_ram_gb, process_resident_memory_gb: max()
#     across the gates pass and the timed pass (single serial process
#     semantics; diagnostic, not scored).
#   - expert_* counters: summed (structural zeros, asserted above).
merged_tmp="$(mktemp)"
trap 'rm -f "${merged_tmp}"' EXIT
jq -n \
  --slurpfile gates "${SCORE_PATH}" \
  --slurpfile timing "${CANDIDATE_SCORE_PATH}" \
  --slurpfile results "${RESULTS_PATH}" \
  --argjson decode_floor "${DECODE_FLOOR}" \
  --argjson prefill_floor "${PREFILL_FLOOR}" \
  --arg champion_decode "${CHAMPION_DECODE_SPEEDUP}" \
  --arg champion_prefill "${CHAMPION_PREFILL_SPEEDUP}" \
  --argjson decode_band_up "${DECODE_BAND_UP}" \
  --argjson decode_band_down "${DECODE_BAND_DOWN}" \
  --argjson prefill_band_up "${PREFILL_BAND_UP}" \
  --argjson prefill_band_down "${PREFILL_BAND_DOWN}" \
  '
  ($gates[0]) as $g | ($timing[0]) as $t | ($results[0]) as $r
  | ($r.paired.decode_speedup) as $ds
  | ($r.paired.prefill_speedup) as $ps
  | ($ds >= $decode_floor) as $decode_ok
  | ($ps >= $prefill_floor) as $prefill_ok
  # Advancing acceptance band vs the champion (when recorded): the
  # candidate/champion seconds ratio per axis is champion_speedup /
  # candidate_speedup, since both speedups reference the same in-session
  # pinned-baseline measurement (common-mode drift cancels in the ratio).
  # > 1 + up = slower than the champion (regression); < 1 - down = a
  # per-submission gain over the champion beyond the step cap.
  | ($champion_decode | length > 0) as $has_champion
  | (if $has_champion then (($champion_decode | tonumber) / $ds) else 1 end) as $decode_vs_champion
  | (if $has_champion then (($champion_prefill | tonumber) / $ps) else 1 end) as $prefill_vs_champion
  | (($has_champion | not) or ($decode_vs_champion <= (1 + $decode_band_up) and $decode_vs_champion >= (1 - $decode_band_down))) as $decode_band_ok
  | (($has_champion | not) or ($prefill_vs_champion <= (1 + $prefill_band_up) and $prefill_vs_champion >= (1 - $prefill_band_down))) as $prefill_band_ok
  | ($g.passed and $decode_ok and $prefill_ok and $decode_band_ok and $prefill_band_ok) as $merged_passed
  | $g
  | .metrics.decode_seconds_per_token = $r.candidate.decode_seconds_per_token
  | .metrics.prefill_seconds_per_token = $r.candidate.prefill_seconds_per_token
  | .metrics.baseline_decode_seconds_per_token = $r.baseline.decode_seconds_per_token
  | .metrics.baseline_prefill_seconds_per_token = $r.baseline.prefill_seconds_per_token
  | .metrics.decode_speedup = $ds
  | .metrics.prefill_speedup = $ps
  | .metrics.decode_speedup_floor = $decode_floor
  | .metrics.prefill_speedup_floor = $prefill_floor
  | .metrics.passed_decode_speedup_floor = $decode_ok
  | .metrics.passed_prefill_speedup_floor = $prefill_ok
  | .metrics.timed_benchmark_seconds = $t.metrics.timed_benchmark_seconds
  | .metrics.bandwidth_gb_per_token = $t.metrics.bandwidth_gb_per_token
  | .metrics.bandwidth_source = $t.metrics.bandwidth_source
  | .metrics.benchmark_wall_seconds = ([.metrics.benchmark_wall_seconds, $t.metrics.benchmark_wall_seconds] | max)
  | .metrics.expert_bytes_read = (.metrics.expert_bytes_read + $t.metrics.expert_bytes_read)
  | .metrics.expert_cache_hits = (.metrics.expert_cache_hits + $t.metrics.expert_cache_hits)
  | .metrics.expert_cache_misses = (.metrics.expert_cache_misses + $t.metrics.expert_cache_misses)
  | .metrics.expert_cache_evictions = (.metrics.expert_cache_evictions + $t.metrics.expert_cache_evictions)
  | .metrics.expert_read_seconds = (.metrics.expert_read_seconds + $t.metrics.expert_read_seconds)
  | .metrics.expert_peak_cached_tensors = ([.metrics.expert_peak_cached_tensors, $t.metrics.expert_peak_cached_tensors] | max)
  | .metrics.peak_ram_gb = ([.metrics.peak_ram_gb, $t.metrics.peak_ram_gb] | max)
  | .metrics.process_resident_memory_gb = ([.metrics.process_resident_memory_gb, $t.metrics.process_resident_memory_gb] | max)
  | .metrics.partial_result = false
  | .passed = $merged_passed
  | if $merged_passed then
      .score = (pow($ds; 0.75) * pow($ps; 0.25))
    else
      .score = null
      # Error precedence mirrors the harness benchmark path: speedup floors
      # first, then the acceptance band (prefill reported before decode). Both
      # prefixes below are fixed harness-authored strings that
      # redact-benchmark-failure.sh categorizes.
      | (if (($decode_ok and $prefill_ok) | not) then
          .metrics.error = "performance floor failed (paired decode_speedup=\($ds) prefill_speedup=\($ps))"
        elif ($prefill_band_ok | not) then
          .metrics.error = (
            if $prefill_vs_champion > (1 + $prefill_band_up) then
              "acceptance band failed: prefill candidate/champion ratio \($prefill_vs_champion) exceeds +\($prefill_band_up * 100)% of the champion reference: slowdown/regression beyond tolerance"
            else
              "acceptance band failed: prefill candidate/champion ratio \($prefill_vs_champion) improves on the champion reference by more than \($prefill_band_down * 100)%: a single submission gain is capped -- please stage it across submissions in smaller batches (or this is a suspiciously lucky reading)"
            end)
        elif ($decode_band_ok | not) then
          .metrics.error = (
            if $decode_vs_champion > (1 + $decode_band_up) then
              "acceptance band failed: decode candidate/champion ratio \($decode_vs_champion) exceeds +\($decode_band_up * 100)% of the champion reference: slowdown/regression beyond tolerance"
            else
              "acceptance band failed: decode candidate/champion ratio \($decode_vs_champion) improves on the champion reference by more than \($decode_band_down * 100)%: a single submission gain is capped -- please stage it across submissions in smaller batches (or this is a suspiciously lucky reading)"
            end)
        else . end)
    end
  ' > "${merged_tmp}"

mv "${merged_tmp}" "${SCORE_PATH}"
trap - EXIT

# Re-anchor the published-integrity pair: the sidecar must name the artifact
# filename (score.json) so `shasum -c` works on the downloaded artifact, and
# the harness integrity record must reference the merged bytes.
score_hash="$(shasum -a 256 "${SCORE_PATH}" | awk '{print $1}')"
printf '%s  score.json\n' "${score_hash}" > "${SCORE_PATH}.sha256"
integrity_tmp="$(mktemp)"
jq --arg hash "${score_hash}" '.score_sha256 = $hash' "${INTEGRITY_PATH}" > "${integrity_tmp}"
mv "${integrity_tmp}" "${INTEGRITY_PATH}"

merged_passed="$(jq -r '.passed' "${SCORE_PATH}")"
merged_score="$(jq -r '.score' "${SCORE_PATH}")"
echo "overlay-paired-timing: passed=${merged_passed} score=${merged_score} decode_speedup=$(jq -r '.metrics.decode_speedup' "${SCORE_PATH}") prefill_speedup=$(jq -r '.metrics.prefill_speedup' "${SCORE_PATH}")"
if [[ -n "${CHAMPION_DECODE_SPEEDUP}" ]]; then
  echo "overlay-paired-timing: acceptance band gated against champion decode_speedup=${CHAMPION_DECODE_SPEEDUP} prefill_speedup=${CHAMPION_PREFILL_SPEEDUP} (tolerances decode +${DECODE_BAND_UP}/-${DECODE_BAND_DOWN}, prefill +${PREFILL_BAND_UP}/-${PREFILL_BAND_DOWN})"
else
  echo "overlay-paired-timing: no champion recorded; acceptance band left to the harness in-process gate vs the frozen reference"
fi

if [[ "${merged_passed}" != "true" ]]; then
  echo "::error::merged benchmark verdict is a failure (speedup floor, acceptance band, or gates)" >&2
  exit 1
fi
