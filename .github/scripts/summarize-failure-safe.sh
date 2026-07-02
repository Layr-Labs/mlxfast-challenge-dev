#!/usr/bin/env bash
# Emit a participant-safe, human-readable reason a benchmark score failed.
#
# LEAK CONTRACT: this script reads ONLY non-sensitive score.json fields --
# speedup ratios, their floors, gate pass/fail booleans, and pass/case COUNTS.
# It never reads (and never emits) any field that can carry hidden-fixture
# content: .metrics.error (free text -- may quote golden token ids),
# .metrics.first_failing_case / .first_failing_step / .first_failing_layer
# (position/identity in the hidden oracle), or .metrics.expected_token /
# .actual_token. Those stay withheld exactly as the fail-closed upload gate
# leaves them; this only translates the SAFE fields into a sentence.
#
# Usage: summarize-failure-safe.sh <score.json>
# Prints one human-readable line to stdout. Exit 0 always (it is a reporter,
# not a gate); callers decide what to do with a still-failing run.
set -euo pipefail

score_path="${1:?usage: summarize-failure-safe.sh <score.json>}"
if [[ ! -s "${score_path}" ]]; then
  echo "benchmark failure reason: score.json is missing or empty (the run likely crashed or timed out before writing a score)."
  exit 0
fi

# One jq pass over the allowlisted safe fields only. Building the message
# inside jq keeps the field access auditable in one place -- the set of
# .metrics.* keys referenced below IS the complete list of what can leak, and
# every one is a number, boolean, or count, never fixture-derived text.
jq -r '
  def r($x): (($x * 1000) | floor) / 1000;
  .passed as $passed
  | .metrics as $m
  | ($m.prefill_speedup) as $ps
  | ($m.decode_speedup) as $ds
  | ($m.prefill_speedup_floor) as $pf
  | ($m.decode_speedup_floor) as $df
  | if $passed == true then
      "benchmark passed: prefill speedup \(r($ps)), decode speedup \(r($ds))."
    elif ($m.passed_prefill_speedup_floor == false or $m.passed_decode_speedup_floor == false) then
      ([ if $m.passed_prefill_speedup_floor == false
         then "prefill speedup \(r($ps)) is below the required floor \($pf)"
         else empty end,
         if $m.passed_decode_speedup_floor == false
         then "decode speedup \(r($ds)) is below the required floor \($df)"
         else empty end
       ] | join(" and ")) as $which
      | "benchmark failed: performance floor not met -- \($which). The submission produced correct output but was not fast enough on at least one axis (measured on the official runner): prefill speedup \(r($ps)) (floor \($pf)), decode speedup \(r($ds)) (floor \($df))."
    elif ($m.passed_correctness == false) then
      "benchmark failed: correctness gate -- the model output diverged from the hidden reference. Location and token specifics are withheld to protect the hidden fixtures; reproduce locally against the public correctness golden to debug."
    elif (($m.gpqa_ttft_case_count // 0) > 0 and $m.gpqa_ttft_passed == false) then
      "benchmark failed: hidden GPQA time-to-first-token gate -- first-token latency exceeded the guardrail (p50 \(r($m.gpqa_ttft_p50_seconds // 0))s, max \(r($m.gpqa_ttft_max_seconds // 0))s over \($m.gpqa_ttft_case_count) cases)."
    elif ($m.semantic_gpqa_passed == false) then
      "benchmark failed: semantic GPQA gate -- only \($m.semantic_gpqa_pass_count // 0) of \($m.semantic_gpqa_case_count // 0) answers were judged correct (below the required minimum)."
    else
      "benchmark failed: the run did not pass, but no scored gate (performance floor, correctness, GPQA TTFT, or semantic GPQA) reported the specific cause in the safe fields. See operator-only logs for the withheld detail."
    end
' "${score_path}"
