#!/bin/bash
# =============================================================================
# STAGED OPERATOR REFERENCE -- NEVER RUN FROM THIS REPO.
# Version-controlled copy of the m5-bench runner's measurement wrapper,
# mirroring how ops/m5-bench/m5-bench-timing.yml is staged. To deploy, an
# operator copies this file to /opt/bench-runner/measure-job.sh on the runner
# box and re-signs the protected-surface integrity manifest (see RUNBOOK);
# the janitor quarantines the box on unsigned drift. Nothing in this repo
# executes this file.
# =============================================================================
# measure-job.sh -- Workstream B end-to-end measurement wrapper for the
# m5-bench self-hosted runner. Invoked by the m5-bench-timing workflow job
# (as the `runner` user) or manually by an operator.
#
# Pipeline:
#   preflight (quarantine flag, quiescence)          [trusted, runner]
#   -> thermal gate (GPU <= 40C)                     [trusted, runner]
#   -> per-phase re-quiescence (bench residue reap   [trusted, runner]
#      + load/util settle) before every timed attempt
#   -> ensure per-implementation benchmark oracle    [generation untrusted via
#      (self-generated, cached per binary hash)       bench-exec; assembly trusted]
#   -> timed official benchmark, candidate           [untrusted via bench-exec]
#      with 2 Hz macmon telemetry                    [telemetry trusted, runner]
#   -> acceptance check w/ one gated retry           [trusted]
#   -> timed official benchmark, pinned baseline     [same mechanics]
#   -> baseline sanity band vs calibration           [trusted]
#   -> paired ratio + sealed results.json            [trusted]
#
# Isolation contract with Workstream A (bench-exec.sh):
#   sudo -u bench /opt/bench/bench-exec.sh <workspace-abs-path> <command> [args...]
#     - drops to the `bench` uid, cwd = workspace
#     - isolates the parent via uid + PF egress block + workspace-write
#       restriction (NOT via a wrapping sandbox-exec: macOS refuses nested
#       sandbox-exec and the harness applies a Seatbelt profile to its OWN
#       worker child)
#     - exports MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE=<merged hardened worker
#       profile> and MLXFAST_USE_RUNTIME_WORKER=1, then execs the command
#       directly. We therefore invoke ./benchmark.sh --official NORMALLY:
#       no sandbox-exec of our own, no MLXFAST_NO_SANDBOX.
#
# Score sealing: the benchmark process's stdout is captured in THIS trusted
# shell; the score payload is emitted as the exec'd process's final stdout act
# and validated here (exactly one JSON object, passed:boolean, score, metrics)
# before being copied to a runner-owned sealed file -- mirroring how
# benchmark.sh seals score.json from the Swift process stdout. Any later
# modification of on-disk files inside the bench-writable workspace is
# discarded.
#
# Acceptance verdicts (from prior measurement work on this box):
#   TOLERATE  metrics.error == "" | "acceptance band failed*" | "performance
#             floor failed*"  (M4-calibrated reference constants are
#             meaningless on this M5; the measurement is still valid)
#   REJECT    token mismatches (first_failing_step/case != null), crashes /
#             other harness errors, missing metrics, GPU clock throttling
#             (any steady loaded sample < 1600 MHz; the first sample of each
#             loaded stretch is a macmon ramp artifact and is excluded from
#             the floor only), missing telemetry.
#
# Exit codes:
#   0 accepted   2 thermal gate timeout   3 preflight quiescence failed
#   4 quarantine flag present             5 candidate rejected
#   6 baseline rejected / drift           7 oracle generation failed
#   8 prerequisite missing                9 usage error
set -u

BR_HOME="$(cd "$(dirname "$0")" && pwd)"
STATE_DIR="${MEASURE_STATE_DIR:-${BR_HOME}/state}"

# Move off any cwd the invoking context may hold that this process's euid cannot
# stat (e.g. a runner-owned job dir entered as root, or a bench-only workspace).
# All paths below are absolute; a stable, always-accessible cwd avoids spurious
# "getcwd: cannot access parent directories" failures from cp/find/etc.
cd / 2>/dev/null || true

# --- Tunables (env-overridable) ---------------------------------------------
MEASURE_EXEC_MODE="${MEASURE_EXEC_MODE:-bench-exec}"   # bench-exec | direct
BENCH_EXEC="${BENCH_EXEC:-/opt/bench/bench-exec.sh}"
BENCH_USER="${BENCH_USER:-bench}"
QUARANTINE_FLAG="${QUARANTINE_FLAG:-/opt/bench/quarantine.flag}"

# --- Per-box single-runner lock ---------------------------------------------
# Concurrency across boxes is fine (per-run workflow group), but TWO measure-job
# runs on the SAME box would corrupt each other's baseline clone, oracle cache,
# and bench reset. Take an atomic per-box lock (mkdir is atomic on APFS); refuse
# to start if a LIVE measure-job already holds it. A stale lock (recorded pid
# gone) is reaped. The EXIT trap that releases it is registered together with
# the baseline-clone cleanup below so both always run.
# The lock lives under the runner-writable STATE_DIR: the previous default
# (/opt/bench-runner/measure-job.lock) sat in the root-owned 0755
# /opt/bench-runner, where the runner uid can never mkdir, so every ranked
# measure step died at acquisition with exit 9 ("could not acquire box lock")
# while root-run operator tests passed. Same serialization contract, in a
# directory the ranked invoker actually owns.
BOX_LOCK="${BOX_LOCK:-${STATE_DIR}/measure-job.lock}"
mkdir -p "${STATE_DIR}" 2>/dev/null || true
if mkdir "${BOX_LOCK}" 2>/dev/null; then
  echo "$$" > "${BOX_LOCK}/pid" 2>/dev/null || true
else
  _lock_holder="$(cat "${BOX_LOCK}/pid" 2>/dev/null || true)"
  if [ -n "${_lock_holder}" ] && kill -0 "${_lock_holder}" 2>/dev/null; then
    echo "measure-job: another measure-job (pid ${_lock_holder}) holds the box lock ${BOX_LOCK}; refusing to run two on one box" >&2
    exit 9
  fi
  echo "measure-job: reaping stale box lock (dead pid ${_lock_holder:-unknown})" >&2
  rm -rf "${BOX_LOCK}" 2>/dev/null || true
  mkdir "${BOX_LOCK}" 2>/dev/null || { echo "measure-job: could not acquire box lock ${BOX_LOCK}" >&2; exit 9; }
  echo "$$" > "${BOX_LOCK}/pid" 2>/dev/null || true
fi
# Early release trap for the window before the combined trap below is set.
trap 'rm -rf "${BOX_LOCK}" 2>/dev/null || true' EXIT

# STABILITY CONTRACT -- FIXED, NOT OVERRIDABLE. These define the thermal
# regime that gives the validated CV ~0.04% on this box; letting a job/env
# weaken them (e.g. a warm-start gate or a short cooldown ceiling) would
# silently degrade every measurement. Assigned literally (env ignored) and
# marked readonly so nothing downstream -- workflow env, supervisor.env,
# sudoers env_keep, or submitted code -- can move them.
readonly GATE_TEMP=40                       # start a run only below this GPU temp (C)
readonly COOL_TIMEOUT=900                   # max seconds to wait for the gate
# Throttle floor: reject if any STEADY loaded sample is below this (MHz).
# "Steady" = a loaded sample whose immediately preceding telemetry sample was
# ALSO loaded. The first sample of each loaded stretch is excluded from the
# floor (never from sample counts): macmon's 500 ms interval averages the
# idle->load clock ramp into that first sample, which reads 1570-1595 MHz at
# >0.95 util on this box while the very next sample is at full clock
# (1607-1620). That ramp artifact false-rejected the PINNED BASELINE tree
# itself (ranked runs 29193101539/29197772216, operator variance runs
# 2026-07-11) at 51-55C -- far below any real thermal throttle. Genuine
# sustained throttling spans consecutive loaded samples and still rejects on
# the 2nd+ sample of the stretch. The 1600 value itself is unchanged and
# remains fixed, readonly, non-overridable.
readonly MIN_FREQ=1600                      # reject if any steady loaded sample below (MHz)
TELEM_INTERVAL_MS="${TELEM_INTERVAL_MS:-500}"   # 2 Hz telemetry
# The GPU-loaded portion of an official run is only ~6 s (512-token prefill +
# 128-step decode; the rest is per-phase model load), i.e. ~10-12 loaded
# samples at 2 Hz -- validated live 2026-07-08. Keep this well below that so
# a faster candidate is not falsely rejected, but nonzero so a run with dead
# telemetry or a crashed benchmark never passes.
MIN_LOADED_SAMPLES="${MIN_LOADED_SAMPLES:-5}"   # telemetry must show a real run
GPU_LOADED_UTIL="${GPU_LOADED_UTIL:-0.9}"   # sample counts as "loaded" above this

# SCORE PLAUSIBILITY GUARDS -- FIXED, NOT OVERRIDABLE (Chain C hardening).
# The scored decode/prefill seconds-per-token are taken VERBATIM from the
# benchmark process's own stdout; nothing else here previously checked them for
# physical plausibility. macmon telemetry is an INDEPENDENT host-clock witness
# of when the GPU was actually loaded. These guards reject a run whose
# self-reported timing is impossible given the observed GPU-loaded window or is
# faster than any real optimization on this silicon could be. They are
# deliberately LOOSE so a genuinely fast-but-honest candidate never trips them
# (an honest run reports MORE compute than these floors, with headroom); they
# only catch fabrication. Assigned literally + readonly so submitted code,
# workflow env, or sudoers env_keep cannot move them (same posture as the
# thermal contract above). NOTE: re-validate the tolerances on the M5 box
# against real paired variance -- editing this file requires a manifest re-sign.
readonly SCORED_PREFILL_TOKENS=512          # scored prefill prompt length (mirrors ORACLE_PROMPT_TOKENS)
readonly SCORED_DECODE_TOKENS=128           # scored decode steps (ORACLE_MIN_EXPECTED-1: 1 prefill + 128 decode)
# Reported compute = prefill_spt*SCORED_PREFILL_TOKENS + decode_spt*SCORED_DECODE_TOKENS.
# It must be at least this fraction of the telemetry GPU-loaded wall window
# (max-min timestamp over loaded samples). The window is NOT only scored work:
# it legitimately includes eager init/warm-up GPU work (weight prep, kernel
# warming, transform-metadata validation) that the challenge docs explicitly
# encourage, repeated per phase because each timed phase loads the model
# fresh -- so it grows without bound relative to scored compute as candidates
# do more init. Measured honest ratios on the ranked boxes (2026-07-12):
# pinned baseline ~0.53 (window ~12.3 s); legitimate init-heavy candidates
# 0.2364-0.2954 (windows 19-24 s, healthy 1618-1620 MHz clocks, all gates
# passed) -- the old 0.25 cut INSIDE that honest range and false-rejected two
# ranked submissions (runs 29180914015, 29181951062). 0.10 restores headroom
# while still rejecting the fabrication this guard was built for: a claimed
# "instant" score reports an order of magnitude less compute than the observed
# loaded window (the documented 0.001 s/tok decode example lands near 0.077
# against an honest window -- still under the floor, and independently dead on
# MIN_DECODE_SPT below). Fine-grained shave-fabrications are bounded by the
# absolute floors and MAX_PLAUSIBLE_SPEEDUP, not by this fraction: with small
# honest windows (2.9-3.5 s observed) the old 0.25 never constrained them
# either. Keep this a coarse order-of-magnitude witness, or it starts scoring
# init strategy instead of fabrication.
readonly MIN_REPORTED_WINDOW_FRACTION="0.10"
# Absolute per-token floors: no dense 4-bit forward on this silicon runs this
# fast. Baseline decode ~0.044 s/tok, prefill ~0.0016 s/tok; these floors are
# ~9x and ~8x faster than baseline -- far beyond any real optimization, so they
# only ever trip on fabrication, never on an honest run.
readonly MIN_DECODE_SPT="0.005"
readonly MIN_PREFILL_SPT="0.0002"
# Hard ceiling on the paired speedups (baseline/candidate). Honest paired
# speedups here are ~1.0-1.1; even a strong legitimate optimization stays well
# under 5x. A fabricated too-fast candidate inflates the ratio past this and is
# rejected as implausible.
readonly MAX_PLAUSIBLE_SPEEDUP="5.0"

PREFLIGHT_MAX_LOAD="${PREFLIGHT_MAX_LOAD:-2.0}"
PREFLIGHT_MAX_GPU_UTIL="${PREFLIGHT_MAX_GPU_UTIL:-0.10}"
PREFLIGHT_PROC_PATTERN="${PREFLIGHT_PROC_PATTERN:-darkbloom}"
# Per-phase re-quiescence (see phase_quiesce). The reap/assert pattern is
# benchmark.sh's own RESIDENT_MODEL_PROCESS_PATTERN: the argv of every mlxfast
# process that holds (or is loading) the ~17 GB model -- the runtime worker in
# any wrapping (bare or under sandbox-exec; -f matches the full argv) plus the
# in-process model-holding CLI subcommands. Fixed, not env-tunable: it must
# always match what the harness actually spawns.
readonly RESIDENT_MODEL_PATTERN='runtime-worker[[:space:]]+--weights|mlxfast-swift[[:space:]]+(benchmark|correctness|correctness-trace|generate-golden|generate-gpqa-answers|attach-free-run-gate)'
PHASE_QUIESCE_TIMEOUT="${PHASE_QUIESCE_TIMEOUT:-300}"   # max seconds for load/util to settle per phase

ORACLE_CACHE_DIR="${ORACLE_CACHE_DIR:-${STATE_DIR}/oracle-cache}"
ORACLE_STEPS="${ORACLE_STEPS:-200}"
ORACLE_PROMPT="${ORACLE_PROMPT:-correctness_prompts/public_longcopy_gate_english_512.txt}"
ORACLE_PROMPT_TOKENS="${ORACLE_PROMPT_TOKENS:-512}"
ORACLE_MIN_EXPECTED="${ORACLE_MIN_EXPECTED:-129}"   # 1 prefill + 128 decode expectations

BASELINE_WS="${BASELINE_WS:-}"
BASELINE_CALIBRATION="${BASELINE_CALIBRATION:-${STATE_DIR}/baseline-calibration.json}"
BASELINE_BAND_ENFORCE="${BASELINE_BAND_ENFORCE:-1}"
# H2: default ON. The pinned baseline tree is read-only to bench (score-
# denominator tamper protection); each job measures a per-job copy-on-write
# clone the baseline run may write but the candidate could never pre-touch.
BASELINE_CLONE="${BASELINE_CLONE:-1}"       # 1: clone pristine baseline into a throwaway per-job work dir
# Runner-writable and on the SAME volume as the pinned tree (/Users/Shared) so
# clonefile stays copy-on-write. The old /opt/bench/work default was not
# runner-writable, so the clone path never actually worked.
BASELINE_CLONE_PARENT="${BASELINE_CLONE_PARENT:-/Users/Shared/bench-jobs/.baseline-clones}"
BASELINE_CLONE_DIR=""                        # set when a clone is made; cleaned on exit

MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"           # 1 gated retry on rejected run

# Workspace-relative filenames used inside each measured tree.
SCORE_REL="score.mjob.json"
INTEGRITY_REL="benchmark-integrity.mjob.json"
ORACLE_REL="bench_oracle.mjob.json"
ORACLE_SRC_REL="oracle_src.mjob.json"

# M1: the worker Seatbelt profile must deny reading the ACTUAL grading oracle
# this script installs (ORACLE_REL), not bench-exec's unused default filename
# (bench_oracle_golden.json). BENCH_GOLDEN_PATH is in the sudoers env_keep list,
# so exporting it makes each `sudo -n bench-exec` render
# (deny file-read* (literal <ws>/bench_oracle.mjob.json)) for the real oracle.
# Workspace-relative: bench-exec joins it onto the resolved workspace.
export BENCH_GOLDEN_PATH="${BENCH_GOLDEN_PATH:-${ORACLE_REL}}"

# --- macmon autodetect -------------------------------------------------------
if [ -z "${MACMON:-}" ]; then
  for cand in "${BR_HOME}/bin/macmon" "${HOME}/bin/macmon" /opt/bench-runner/bin/macmon; do
    if [ -x "${cand}" ]; then MACMON="${cand}"; break; fi
  done
fi
MACMON="${MACMON:-macmon}"

# --- CLI ----------------------------------------------------------------------
CANDIDATE_WS=""
TAG="mjob-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR=""
PREFLIGHT_ONLY=0
CANDIDATE_ONLY=0

usage() {
  cat >&2 <<EOF
usage: measure-job.sh --candidate <abs-workspace> [options]
  --candidate PATH    candidate workspace (built + transformed; required
                      unless --preflight-only)
  --baseline PATH     pinned prebuilt baseline workspace
                      (default: \$BASELINE_WS)
  --tag TAG           run tag (default: timestamp)
  --out DIR           results directory
                      (default: \$STATE_DIR/results/<tag>)
  --preflight-only    run quiescence/quarantine checks and exit
  --candidate-only    skip baseline pairing (smoke use only)
env: MEASURE_EXEC_MODE=bench-exec|direct
     BASELINE_WS  BASELINE_CALIBRATION  BASELINE_BAND_ENFORCE  BASELINE_CLONE
     ORACLE_CACHE_DIR  MACMON  QUARANTINE_FLAG  (see header comments)
NOTE: the thermal-stability contract (GATE_TEMP=40C, COOL_TIMEOUT=900s,
     MIN_FREQ=1600MHz) is FIXED in this script and cannot be overridden.
EOF
  exit 9
}

while [ $# -gt 0 ]; do
  case "$1" in
    --candidate) CANDIDATE_WS="${2:?}"; shift 2 ;;
    --baseline)  BASELINE_WS="${2:?}"; shift 2 ;;
    --tag)       TAG="${2:?}"; shift 2 ;;
    --out)       OUT_DIR="${2:?}"; shift 2 ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --candidate-only) CANDIDATE_ONLY=1; shift ;;
    -h|--help)   usage ;;
    *) echo "measure-job: unknown argument $1" >&2; usage ;;
  esac
done

OUT_DIR="${OUT_DIR:-${STATE_DIR}/results/${TAG}}"
mkdir -p "${OUT_DIR}" "${ORACLE_CACHE_DIR}"
LOG_FILE="${OUT_DIR}/measure.log"

log() {
  local line
  line="$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
  echo "${line}" | tee -a "${LOG_FILE}" >&2
}

die() {
  local code="$1"; shift
  log "FATAL(code=${code}): $*"
  echo "FAIL code=${code} reason=\"$*\"" > "${OUT_DIR}/verdict.txt"
  exit "${code}"
}

# Remove the throwaway baseline clone (H2) on any exit so per-job copies don't
# accumulate on the Data volume (they are copy-on-write but diverge as the
# baseline run writes score/tmp into them). Never fails the run.
cleanup_baseline_clone() {
  [ -n "${BASELINE_CLONE_DIR}" ] && rm -rf "${BASELINE_CLONE_DIR}" 2>/dev/null || true
}
# Combined EXIT trap: release the per-box lock AND remove the baseline clone.
# Supersedes the early lock-only trap set at acquisition above.
trap 'cleanup_baseline_clone; rm -rf "${BOX_LOCK}" 2>/dev/null || true' EXIT

# ACL granted to the bench uid on a per-job tree so the sandboxed run can write
# its own outputs (score, tmp, rendered profile) inside a runner-owned dir --
# same grant the ranked workflow applies to the candidate workspace. Used for
# the baseline CLONE so the pinned tree itself never needs to be bench-writable.
BENCH_WS_ACL="user:${BENCH_USER} allow list,search,readattr,readextattr,read,execute,add_file,add_subdirectory,delete_child,write,append,writeattr,writeextattr,file_inherit,directory_inherit"

# --- Execution bridge ---------------------------------------------------------
# exec_in_ws <workspace> <command> [args...]
# bench-exec mode: through Workstream A's bridge (uid drop + PF egress block +
# workspace-write restriction; exports the hardened worker profile env).
# direct mode: run as the current user with cwd=workspace (validation only;
# proves gate/telemetry/acceptance/pairing logic without root).
exec_in_ws() {
  local ws="$1"; shift
  if [ "${MEASURE_EXEC_MODE}" = "direct" ]; then
    ( cd "${ws}" && "$@" )
  else
    # ROOT mode: runner is granted `sudo (root) bench-exec.sh` and bench-exec
    # drops to the bench uid itself. This is the only working mode -- the
    # `sudo -u bench` (BENCH) mode cannot execute the 0750 root:wheel bridge
    # (nor read its 0750 config), by design.
    sudo -n "${BENCH_EXEC}" "${ws}" "$@"
  fi
}

# --- Telemetry / gating helpers ------------------------------------------------
macmon_sample() {
  "${MACMON}" pipe -s1 2>/dev/null
}

gpu_temp() {
  macmon_sample | jq -r '.temp.gpu_temp_avg // empty'
}

gpu_util() {
  macmon_sample | jq -r '.gpu_usage[1] // empty'
}

# num_lt / num_gt <a> <b> : float compare (true iff a<b / a>b) that parses
# SCIENTIFIC NOTATION correctly. jq renders a very small (or very large)
# magnitude with an exponent (e.g. 0.00000001 -> "1E-8", 4.4e18 -> "4.4e+18"),
# and `bc` mis-evaluates such a literal (it treats `E`/`e` as a stray token, not
# an exponent), so a bc `< floor` / `> ceiling` test silently returns the WRONG
# answer for exactly the fabricated out-of-range values those guards exist to
# reject. awk parses E-notation natively, so every plausibility comparison on a
# SELF-REPORTED (benchmark-stdout) number goes through these instead of bc. `+0`
# coerces the operands to numbers; a non-numeric operand coerces to 0. Used by
# evaluate_acceptance (metric presence), plausibility_check, and the paired
# ceiling. (The bc calls left in this file compare only runner-owned telemetry /
# sysctl values -- temp, load, freq, baseline ratios -- which are never in
# E-notation and never participant-controlled.)
num_lt() { awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN { exit !((a + 0) < (b + 0)) }'; }
num_gt() { awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN { exit !((a + 0) > (b + 0)) }'; }

telem_loaded_window() {
  # telem_loaded_window <telem-jsonl> -> seconds (float) spanned by the GPU-
  # LOADED samples (max-min of their timestamps), the SAME loaded-util threshold
  # the acceptance stats use. This is the independent host clock the plausibility
  # cross-check compares the self-reported timing against. macmon stamps ISO8601
  # with microseconds and a +00:00 offset; parse to epoch seconds KEEPING the
  # fraction. Prints 0 when there are no loaded samples or telemetry is unusable
  # (fail-open: 0 disables the fraction check, other guards still apply).
  local telem="$1"
  [ -s "${telem}" ] || { echo 0; return; }
  jq -rs --argjson loaded "${GPU_LOADED_UTIL}" '
    def epoch:
      (sub("\\+00:00$"; "Z")) as $z
      | ($z | sub("\\.[0-9]+Z$"; "Z") | fromdateiso8601) as $whole
      | (($z | capture("\\.(?<f>[0-9]+)Z$") | .f) // "0") as $fs
      | $whole + ("0." + $fs | tonumber);
    [ .[] | select((.gpu_usage[1] // 0) > $loaded) | (.timestamp | epoch) ] as $ts
    | if ($ts | length) > 0 then (($ts | max) - ($ts | min)) else 0 end
  ' "${telem}" 2>/dev/null || echo 0
}

thermal_gate() {
  local label="$1" waited=0 t
  while :; do
    t="$(gpu_temp)"
    if [ -n "${t}" ] && [ "$(echo "${t} < ${GATE_TEMP}" | bc -l)" = "1" ]; then
      log "GATE_OK label=${label} gpu_temp=${t} waited=${waited}s"
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
    if [ "${waited}" -ge "${COOL_TIMEOUT}" ]; then
      log "GATE_TIMEOUT label=${label} gpu_temp=${t:-unknown} waited=${waited}s"
      return 1
    fi
  done
}

preflight() {
  # Quarantine flag from Workstream A's janitor: hard stop, nothing runs.
  if [ -e "${QUARANTINE_FLAG}" ]; then
    die 4 "quarantine flag present at ${QUARANTINE_FLAG}; janitor detected drift -- see RUNBOOK"
  fi

  # The darkbloom provider must stay OFF this box (permanent precondition;
  # it was manually unloaded). Any matching process invalidates quiescence.
  local procs
  procs="$(pgrep -ifl "${PREFLIGHT_PROC_PATTERN}" || true)"
  if [ -n "${procs}" ]; then
    die 3 "quiescence: ${PREFLIGHT_PROC_PATTERN}-class processes running: ${procs}"
  fi

  local load
  load="$(sysctl -n vm.loadavg | awk '{print $2}')"
  if [ "$(echo "${load} < ${PREFLIGHT_MAX_LOAD}" | bc -l)" != "1" ]; then
    die 3 "quiescence: 1-min load ${load} >= ${PREFLIGHT_MAX_LOAD}"
  fi

  local util
  util="$(gpu_util)"
  if [ -z "${util}" ]; then
    die 3 "quiescence: macmon returned no GPU sample (MACMON=${MACMON})"
  fi
  if [ "$(echo "${util} < ${PREFLIGHT_MAX_GPU_UTIL}" | bc -l)" != "1" ]; then
    die 3 "quiescence: GPU util ${util} >= ${PREFLIGHT_MAX_GPU_UTIL}"
  fi

  # Fail closed if the bench egress lockdown is NOT actually live. macOS can
  # reset /etc/pf.conf on reboot/update, which silently drops the com.bench
  # anchor (pf-load refuses to wire a missing anchor), leaving the bench uid
  # with OPEN egress. A scored job must never run on a PF-less box. Enforced
  # only on the real bench-exec scored path (direct mode is local validation,
  # no bench uid, no PF dependency). bench uid is pinned to 560 by contract.
  #
  # HOW we check depends on who we are: /dev/pf is root-only, so the original
  # `pfctl | grep` here always counted 0 rules when this script runs as the
  # runner uid (the ranked path) and false-FATALed every scored run (incident
  # 2026-07-12). Structural rule check when pfctl is readable (root/operator
  # runs); otherwise a FUNCTIONAL probe: run curl as the bench uid through
  # bench-exec against a fixed IP (no DNS dependency) and require it to FAIL.
  # bench-exec (root mode) additionally refuses outright (exit 15) when the
  # uid-560 block is missing from the com.bench anchor, so the bridge itself
  # stays fail-closed even when the probe cannot run.
  if [ "${MEASURE_EXEC_MODE}" = "bench-exec" ] && [ "${BENCH_PF_ENFORCE:-1}" = "1" ]; then
    local pf_out="" pf_rc=0 pf_bench_rules=0
    pf_out="$(/sbin/pfctl -a com.bench -sr 2>/dev/null)" || pf_rc=$?
    if [ "${pf_rc}" -eq 0 ]; then
      pf_bench_rules="$(printf '%s\n' "${pf_out}" | grep -c 'user = 560' || true)"
      if [ "${pf_bench_rules:-0}" -lt 1 ]; then
        die 3 "bench PF egress lockdown NOT active (com.bench anchor has no user-560 block); refusing to run a scored job on an unprotected box -- re-wire /etc/pf.conf + run pf-load.sh (see RUNBOOK)"
      fi
      log "PREFLIGHT_PF_OK structural com.bench user-560 block rules=${pf_bench_rules}"
    else
      log "PREFLIGHT_PF pfctl unreadable as uid $(id -u) (rc=${pf_rc}); using functional bench-uid egress probe"
      local probe_ws="${CANDIDATE_WS:-}" probe_rc=0
      if [ -n "${probe_ws}" ] && [ -d "${probe_ws}" ]; then
        sudo -n "${BENCH_EXEC}" "${probe_ws}" \
          /usr/bin/curl -sS --max-time 6 -o /dev/null https://1.1.1.1/ \
          >/dev/null 2>&1 || probe_rc=$?
        if [ "${probe_rc}" -eq 0 ]; then
          die 3 "bench PF egress lockdown NOT active (bench-uid probe REACHED the internet); refusing to run a scored job on an unprotected box -- re-wire /etc/pf.conf + run pf-load.sh (see RUNBOOK)"
        fi
        if [ "${probe_rc}" -eq 15 ]; then
          die 3 "bench PF egress lockdown NOT active (bench-exec PF guard: uid-560 block missing from com.bench anchor); re-wire /etc/pf.conf + run pf-load.sh (see RUNBOOK)"
        fi
        log "PREFLIGHT_PF_OK functional bench-uid egress probe blocked (curl rc=${probe_rc})"
      else
        log "PREFLIGHT_PF probe skipped (no candidate workspace at preflight); bench-exec's own PF guard still fails closed at first use"
      fi
    fi
  fi

  log "PREFLIGHT_OK load=${load} gpu_util=${util} quarantine=absent"
}

# --- Per-phase re-quiescence -----------------------------------------------------
# preflight() checks quiescence ONCE, before anything runs; nothing re-checked
# it between the timed phases. Each timed attempt's benchmark chain
# (sudo -> bench-exec -> benchmark.sh -> mlxfast-swift -> runtime worker) can
# leave live residue behind after the foreground bridge returns: bash defers
# trap handlers until the foreground child exits, and the model-holding worker
# can miss stdin-EOF during its multi-minute model load, so the top PID
# exiting proves nothing about the rest of the tree. A ~17 GB leftover worker
# from a rejected attempt (or the other side of the pair) still loading or
# spinning while the next phase is measured is exactly the asymmetric
# contention the paired ratio cannot cancel. phase_quiesce therefore runs
# before EVERY timed attempt (baseline, candidate, and each gated retry),
# ahead of the existing thermal gate in run_timed:
#   1. reap residual bench processes: TERM the WHOLE bench uid (not one pid),
#      short grace, KILL survivors. The bench uid exists solely to run
#      submitted bench code and the box lock guarantees one measure-job per
#      box, so anything alive there is residue from a prior phase, attempt,
#      or job. The runner uid cannot signal another uid's processes, so the
#      kill goes through the same bench-exec bridge the runs (and the
#      preflight PF probe) already use; it is best-effort remediation -- the
#      assert below is the guarantee. In direct mode (own-uid validation,
#      no bench uid) only model-holding processes are reaped: a blanket
#      same-uid sweep would kill the operator's own session.
#   2. assert ZERO model-holding processes survive (benchmark.sh's resident-
#      model argv pattern); fail closed with the preflight quiescence
#      convention (die 3) if any do.
#   3. wait for the 1-min load and GPU util to settle back under the SAME
#      PREFLIGHT_MAX_LOAD / PREFLIGHT_MAX_GPU_UTIL thresholds preflight
#      enforced -- wait-then-fail like thermal_gate, NOT insta-die like
#      preflight, because right after a phase finishes the 1-min loadavg
#      legitimately needs a minute or two to decay (settling, not
#      contention). bc (not num_lt) is correct here: these are runner-owned
#      telemetry/sysctl values, never E-notation -- see the num_lt note.
# The thermal_gate call in run_timed stays unchanged and runs after this, so
# every attempt on BOTH sides of the pair starts residue-free, quiescent, and
# cool -- identical measurement preconditions.
phase_quiesce() {
  # phase_quiesce <label> <ws>: <ws> is the workspace handed to bench-exec for
  # the reap (same bridge contract as the preflight PF probe); unused in
  # direct mode.
  local label="$1" ws="$2"
  local pids survivors load util waited=0

  if [ "${MEASURE_EXEC_MODE}" = "direct" ]; then
    pids="$(pgrep -U "$(id -u)" -fl -- "${RESIDENT_MODEL_PATTERN}" 2>/dev/null | tr '\n' ';')"
    if [ -n "${pids}" ]; then
      log "PHASE_QUIESCE_REAP label=${label} mode=direct residue=${pids}"
      pkill -TERM -U "$(id -u)" -f -- "${RESIDENT_MODEL_PATTERN}" 2>/dev/null || true
      sleep 2
      pkill -KILL -U "$(id -u)" -f -- "${RESIDENT_MODEL_PATTERN}" 2>/dev/null || true
      sleep 1
    fi
    survivors="$(pgrep -U "$(id -u)" -fl -- "${RESIDENT_MODEL_PATTERN}" 2>/dev/null | tr '\n' ';')"
  else
    pids="$(pgrep -l -U "${BENCH_USER}" 2>/dev/null | tr '\n' ';')"
    if [ -n "${pids}" ]; then
      log "PHASE_QUIESCE_REAP label=${label} uid=${BENCH_USER} residue=${pids}"
      sudo -n "${BENCH_EXEC}" "${ws}" /usr/bin/pkill -TERM -U "${BENCH_USER}" >/dev/null 2>&1 || true
      sleep 2
      if pgrep -U "${BENCH_USER}" >/dev/null 2>&1; then
        sudo -n "${BENCH_EXEC}" "${ws}" /usr/bin/pkill -KILL -U "${BENCH_USER}" >/dev/null 2>&1 || true
        sleep 1
      fi
    fi
    survivors="$(pgrep -U "${BENCH_USER}" -fl -- "${RESIDENT_MODEL_PATTERN}" 2>/dev/null | tr '\n' ';')"
  fi
  if [ -n "${survivors}" ]; then
    die 3 "phase quiescence (${label}): residual model-holding bench processes survived reap: ${survivors}"
  fi

  while :; do
    load="$(sysctl -n vm.loadavg | awk '{print $2}')"
    util="$(gpu_util)"
    if [ -n "${util}" ] \
       && [ "$(echo "${load} < ${PREFLIGHT_MAX_LOAD}" | bc -l)" = "1" ] \
       && [ "$(echo "${util} < ${PREFLIGHT_MAX_GPU_UTIL}" | bc -l)" = "1" ]; then
      log "PHASE_QUIESCE_OK label=${label} load=${load} gpu_util=${util} waited=${waited}s"
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
    if [ "${waited}" -ge "${PHASE_QUIESCE_TIMEOUT}" ]; then
      die 3 "phase quiescence (${label}): load=${load:-unknown} gpu_util=${util:-unknown} still above preflight thresholds after ${waited}s"
    fi
  done
}

# --- Workspace prerequisites ----------------------------------------------------
resolve_binary() {
  # The worker profile's process-exec allowance references the RESOLVED
  # binary path (.build/arm64-apple-macosx/release/mlxfast-swift), not the
  # .build/release symlink; resolve for hashing so the oracle cache key is
  # stable across the symlink.
  local ws="$1"
  if [ -x "${ws}/.build/release/mlxfast-swift" ]; then
    # readlink -f is available on macOS 12.3+
    readlink -f "${ws}/.build/release/mlxfast-swift" 2>/dev/null \
      || echo "${ws}/.build/release/mlxfast-swift"
  fi
}

check_ws_prereqs() {
  local ws="$1" side="$2"
  [ -d "${ws}" ] || die 8 "${side}: workspace ${ws} missing"
  [ -f "${ws}/benchmark.sh" ] || die 8 "${side}: ${ws}/benchmark.sh missing"
  [ -x "${ws}/.build/release/mlxfast-swift" ] || die 8 "${side}: release binary missing (build step must run first)"
  [ -s "${ws}/.build/release/mlx.metallib" ] || die 8 "${side}: mlx.metallib missing (setup step must run first)"
  [ -f "${ws}/weights/config.json" ] || die 8 "${side}: weights/config.json missing (transform step must run first)"
  [ -f "${ws}/${ORACLE_PROMPT}" ] || die 8 "${side}: oracle prompt ${ORACLE_PROMPT} missing"
}

# Reports "<short-sha>[-dirty]" for a workspace. SECURITY: these git calls run
# as the RUNNER uid (the score-sealing ring) but inside the bench-WRITABLE
# workspace, so a job can plant a repo-local .git/config or hooks that execute
# code as runner via config-driven code paths (core.fsmonitor, core.hooksPath,
# core.pager, alias.*, protocol.ext, include/includeIf, ...). Neutralize them:
#   * ignore global/system/XDG/user config entirely (env), and
#   * pin the dangerous knobs to inert values on the COMMAND LINE, where `-c`
#     has higher precedence than any file (repo-local, included, or global) so a
#     workspace-planted .git/config cannot re-enable them, and
#   * detect "dirty" with the `diff`/`ls-files` plumbing under those same inert
#     knobs (no config-reading porcelain, no fsmonitor/hook invocation).
# Fails open to "unknown" (never executes workspace-controlled code) and keeps
# the exact output contract: "<sha12>" or "unknown", with an optional "-dirty".
GIT_HARDENED_ENV=(
  PATH=/usr/bin:/bin:/usr/sbin:/sbin
  GIT_CONFIG_GLOBAL=/dev/null
  GIT_CONFIG_SYSTEM=/dev/null
  GIT_CONFIG_NOSYSTEM=1
  GIT_TERMINAL_PROMPT=0
  GIT_OPTIONAL_LOCKS=0
  HOME=/var/empty
  XDG_CONFIG_HOME=/dev/null
)
GIT_HARDENED_OPTS=(
  -c core.fsmonitor=false
  -c core.hooksPath=/dev/null
  -c core.pager=cat
  -c core.sshCommand=false
  -c protocol.ext.allow=never
  -c "safe.directory=*"
)
git_hardened() {
  # git_hardened <ws> <git-args...> : run git against an UNTRUSTED workspace with
  # every config-driven code path neutralized.
  local ws="$1"; shift
  env -i "${GIT_HARDENED_ENV[@]}" \
    git -C "${ws}" "${GIT_HARDENED_OPTS[@]}" "$@"
}

ws_head() {
  local ws="$1" head dirty=""
  head="$(git_hardened "${ws}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
  # Only probe dirtiness when HEAD resolved; a non-repo stays "unknown" (never
  # "unknown-dirty"), matching the prior `git status`-based contract, since the
  # plumbing below returns nonzero without a HEAD and would otherwise flip it.
  # Dirty = staged change (index vs HEAD) OR stat-modified tracked file OR any
  # untracked non-ignored file. All three use plumbing that is content-clean:
  #   diff-index --cached  compares the index to HEAD (already-hashed blobs, no
  #                        worktree read) -> never applies a clean filter;
  #   ls-files -m / --others  decide from the stat cache only (no content
  #                        materialization) -> never apply a clean filter.
  # So even a planted .gitattributes + repo-local filter.<name>.clean cannot be
  # provoked into executing. (`git status`/`diff-files` WOULD hash stat-dirty
  # worktree files and can run that filter, which is exactly why they are not
  # used here.) Stat-only detection may over-report a touched-but-identical file
  # as dirty; erring toward "-dirty" is safe for this informational field.
  if [ "${head}" != "unknown" ] \
     && { ! git_hardened "${ws}" diff-index --cached --quiet HEAD -- 2>/dev/null \
          || [ -n "$(git_hardened "${ws}" ls-files -m 2>/dev/null | head -1)" ] \
          || [ -n "$(git_hardened "${ws}" ls-files --others --exclude-standard 2>/dev/null | head -1)" ]; }; then
    dirty="-dirty"
  fi
  echo "${head}${dirty}"
}

# --- Oracle (self-generated per implementation, cached per binary hash) ---------
# Checked-in goldens are M4-generated; on this M5 near-tie argmaxes differ, so
# the benchmark oracle must be generated BY the implementation under test.
# Cache key = sha256 of the resolved release binary: same binary + same
# deterministic transform => same greedy continuation. Generation executes
# submitted-surface model code, so it runs through bench-exec; the oracle JSON
# is assembled from the raw generate-golden output here in the trusted shell.
oracle_key() {
  local ws="$1" bin
  bin="$(resolve_binary "${ws}")"
  [ -n "${bin}" ] || return 1
  shasum -a 256 "${bin}" | awk '{print $1}'
}

assemble_oracle() {
  # assemble_oracle <src-json> <dst-json>
  local src="$1" dst="$2"
  jq -e --argjson want_prompt "${ORACLE_PROMPT_TOKENS}" --argjson want_exp "${ORACLE_MIN_EXPECTED}" '
    (.version == 1)
    and (.cases | type == "array" and length >= 1)
    and (.cases[0].prompt_tokens | type == "array" and length == $want_prompt)
    and (.cases[0].expected_tokens | type == "array" and length >= $want_exp)
  ' "${src}" >/dev/null 2>&1 || return 1
  jq '{
    version: 1,
    cases: .cases,
    benchmark: {
      prefill_prompt_tokens: .cases[0].prompt_tokens,
      expected_prefill_token: .cases[0].expected_tokens[0],
      decode_seed_tokens: .cases[0].prompt_tokens,
      expected_decode_seed_token: .cases[0].expected_tokens[0],
      expected_decode_tokens: .cases[0].expected_tokens[1:]
    }
  }' "${src}" > "${dst}" || return 1
  jq -e '.benchmark.expected_decode_tokens | length >= 128' "${dst}" >/dev/null 2>&1
}

ensure_oracle() {
  # ensure_oracle <ws> <side>: leaves a valid oracle at ${ws}/${ORACLE_REL}.
  local ws="$1" side="$2" key cached
  key="$(oracle_key "${ws}")" || die 8 "${side}: cannot hash release binary for oracle key"
  cached="${ORACLE_CACHE_DIR}/${key}.json"

  if [ -s "${cached}" ]; then
    log "ORACLE_CACHE_HIT side=${side} key=${key}"
  else
    log "ORACLE_GENERATE side=${side} key=${key} steps=${ORACLE_STEPS} (untrusted, via ${MEASURE_EXEC_MODE})"
    thermal_gate "oracle-${side}" || die 2 "thermal gate timeout before oracle generation (${side})"
    if ! exec_in_ws "${ws}" /usr/bin/env MJOB_WS="${ws}" /bin/bash -c '
        set -u
        cd "${MJOB_WS}" || exit 97
        exec .build/release/mlxfast-swift generate-golden \
          --prompt-file "$1" --weights weights --tokenizer weights \
          --output "$2" --name benchoracle --steps "$3"
        ' oracle-gen "${ORACLE_PROMPT}" "${ORACLE_SRC_REL}" "${ORACLE_STEPS}" \
        >> "${OUT_DIR}/oracle-${side}.log" 2>&1; then
      die 7 "${side}: generate-golden failed (see oracle-${side}.log)"
    fi
    [ -s "${ws}/${ORACLE_SRC_REL}" ] || die 7 "${side}: generate-golden produced no ${ORACLE_SRC_REL}"
    # Trusted assembly + validation.
    if ! assemble_oracle "${ws}/${ORACLE_SRC_REL}" "${cached}.tmp"; then
      rm -f "${cached}.tmp"
      die 7 "${side}: oracle assembly/validation failed (prompt!=${ORACLE_PROMPT_TOKENS} or expected<${ORACLE_MIN_EXPECTED})"
    fi
    mv "${cached}.tmp" "${cached}"
    log "ORACLE_CACHED side=${side} key=${key} sha256=$(shasum -a 256 "${cached}" | awk '{print $1}')"
  fi

  cp -f "${cached}" "${ws}/${ORACLE_REL}"
  chmod a+r "${ws}/${ORACLE_REL}" 2>/dev/null || true
}

# --- Timed run + telemetry + sealing + acceptance --------------------------------
# The inner command emits the score payload as its FINAL stdout act; all
# benchmark console output goes to stderr. The trusted shell captures stdout
# and seals from it. It cds into the workspace itself (MJOB_WS) rather than
# assuming the bridge sets cwd.
INNER_BENCH='set -u
cd "${MJOB_WS}" || exit 97
./benchmark.sh --official 1>&2
status=$?
if [ -s "${MLXFAST_SCORE_PATH}" ]; then
  cat "${MLXFAST_SCORE_PATH}"
fi
exit "${status}"'

# Globals set by run_timed / evaluate_acceptance:
VERDICT=""
RUN_DEC=""
RUN_PRE=""
RUN_STATS="{}"

seal_score() {
  # seal_score <raw-stdout-file> <sealed-out>: mirror benchmark.sh's sealing --
  # require exactly one JSON object shaped like a score payload; fail closed on
  # empty, non-JSON, or concatenated objects.
  local raw="$1" sealed="$2"
  if [ "$(jq -s 'length' "${raw}" 2>/dev/null)" != "1" ]; then
    return 1
  fi
  jq -e '(.passed | type == "boolean") and has("score") and (.metrics | type == "object")' \
    "${raw}" >/dev/null 2>&1 || return 1
  cp -f "${raw}" "${sealed}"
  shasum -a 256 "${sealed}" | awk '{print $1}' > "${sealed}.sha256"
}

evaluate_acceptance() {
  # evaluate_acceptance <sealed-score> <telem-jsonl>
  # Sets VERDICT, RUN_DEC, RUN_PRE, RUN_STATS.
  local sealed="$1" telem="$2" err mismatch minf n

  # min_freq is computed over STEADY loaded samples (previous sample also
  # loaded): the first sample of each loaded stretch blends the idle->load
  # clock ramp across macmon's 500 ms averaging window and reads 25-90 MHz low
  # at full utilization, which false-tripped the MIN_FREQ floor on the pinned
  # baseline itself (see the MIN_FREQ comment). ramp_min_freq (min over ALL
  # loaded samples, the old semantics) is kept in the stats for audit. n,
  # mean_freq, and max_temp still cover all loaded samples. Fail closed: with
  # no steady sample, min_freq falls back to the strict all-loaded minimum.
  RUN_STATS="$(jq -s --argjson loaded "${GPU_LOADED_UTIL}" '
    . as $all
    | ($all | to_entries | map(select((.value.gpu_usage[1] // 0) > $loaded))) as $loadedE
    | ($loadedE | map(select(.key > 0 and (($all[.key - 1].gpu_usage[1] // 0) > $loaded)))) as $steadyE
    | ($loadedE | length) as $ln
    | { n: $ln,
        min_freq: (if ($steadyE | length) > 0 then ([$steadyE[].value.gpu_usage[0]] | min)
                   elif $ln > 0 then ([$loadedE[].value.gpu_usage[0]] | min)
                   else 0 end),
        ramp_min_freq: (if $ln > 0 then ([$loadedE[].value.gpu_usage[0]] | min) else 0 end),
        mean_freq: (if $ln > 0 then (([$loadedE[].value.gpu_usage[0]] | add) / $ln | floor) else 0 end),
        max_temp: (if $ln > 0 then ([$loadedE[].value.temp.gpu_temp_avg] | max) else 0 end) }
  ' "${telem}" 2>/dev/null || echo '{"n":0,"min_freq":0,"ramp_min_freq":0,"mean_freq":0,"max_temp":0}')"
  minf="$(echo "${RUN_STATS}" | jq -r '.min_freq')"
  n="$(echo "${RUN_STATS}" | jq -r '.n')"

  RUN_DEC="$(jq -r '.metrics.decode_seconds_per_token // empty' "${sealed}" 2>/dev/null || true)"
  RUN_PRE="$(jq -r '.metrics.prefill_seconds_per_token // empty' "${sealed}" 2>/dev/null || true)"
  err="$(jq -r '.metrics.error // ""' "${sealed}" 2>/dev/null || echo MISSING)"
  mismatch="$(jq -r '(.metrics.first_failing_step != null) or (.metrics.first_failing_case != null)' \
    "${sealed}" 2>/dev/null || echo true)"

  VERDICT=ACCEPT
  # Band/floor policy failures are tolerated: the measurement is valid, the
  # M4-calibrated reference constants are just meaningless on this hardware.
  case "${err}" in
    ""|"acceptance band failed"*|"performance floor failed"*) : ;;
    *) VERDICT=REJECT_ERROR ;;
  esac
  if [ "${mismatch}" != "false" ]; then VERDICT=REJECT_TOKEN_MISMATCH; fi
  # Metric presence: RUN_DEC/RUN_PRE come from benchmark stdout, so use the
  # E-notation-safe primitive (a fabricated "1E-8" must not read as a valid > 0
  # metric through bc's mis-parse).
  if [ -z "${RUN_DEC}" ] || [ -z "${RUN_PRE}" ] \
     || ! num_gt "${RUN_DEC:-0}" 0 \
     || ! num_gt "${RUN_PRE:-0}" 0; then
    VERDICT=REJECT_NO_METRICS
  fi
  if [ "${VERDICT}" = "ACCEPT" ] && [ "${n}" -lt "${MIN_LOADED_SAMPLES}" ]; then
    VERDICT=REJECT_NO_TELEMETRY
  fi
  if [ "${VERDICT}" = "ACCEPT" ] \
     && [ "$(echo "${minf} >= ${MIN_FREQ}" | bc -l)" != "1" ]; then
    VERDICT=REJECT_THROTTLED
  fi
  # Chain C: everything above validates the run MECHANICALLY (telemetry present,
  # not throttled, metrics present). The decode/prefill seconds-per-token are
  # still taken VERBATIM from the benchmark's own stdout, so cross-check them
  # against the independent host telemetry + absolute floors. Only when the run
  # is otherwise ACCEPT (a mechanical reject keeps its more specific reason).
  if [ "${VERDICT}" = "ACCEPT" ]; then
    plausibility_check "${telem}"
  fi
}

plausibility_check() {
  # plausibility_check <telem-jsonl> : cross-checks the already-parsed RUN_DEC /
  # RUN_PRE (seconds per token) against absolute physical floors and the
  # INDEPENDENT macmon GPU-loaded wall window. Sets VERDICT=REJECT_IMPLAUSIBLE
  # (and logs the reason) when the self-reported timing is physically impossible
  # -- e.g. "reported far faster than any load was actually observed". Loose by
  # design: honest runs clear it with room (see the constants block). All numeric
  # comparisons go through num_lt/num_gt (awk), never bc -- see that note.
  local telem="$1" window reported threshold

  # 1. Absolute per-token floors: no dense forward on this silicon is this fast.
  if num_lt "${RUN_DEC:-0}" "${MIN_DECODE_SPT}" \
     || num_lt "${RUN_PRE:-0}" "${MIN_PREFILL_SPT}"; then
    VERDICT=REJECT_IMPLAUSIBLE
    log "PLAUSIBILITY_REJECT reason=abs_floor decode=${RUN_DEC:-NA} (min ${MIN_DECODE_SPT}) prefill=${RUN_PRE:-NA} (min ${MIN_PREFILL_SPT})"
    return
  fi

  # 2. Telemetry cross-check: the reported total compute cannot be a tiny
  #    fraction of the observed GPU-loaded wall window (host clock, not stdout).
  #    Reported compute and the window*fraction threshold are computed in awk
  #    (E-notation-safe) for the same reason the floors avoid bc.
  window="$(telem_loaded_window "${telem}")"
  reported="$(awk -v p="${RUN_PRE:-0}" -v d="${RUN_DEC:-0}" \
    -v pt="${SCORED_PREFILL_TOKENS}" -v dt="${SCORED_DECODE_TOKENS}" \
    'BEGIN { printf "%.9f", (p + 0) * pt + (d + 0) * dt }')"
  threshold="$(awk -v w="${window:-0}" -v f="${MIN_REPORTED_WINDOW_FRACTION}" \
    'BEGIN { printf "%.9f", (w + 0) * f }')"
  log "PLAUSIBILITY reported_compute_s=${reported} telem_loaded_window_s=${window} min_fraction=${MIN_REPORTED_WINDOW_FRACTION}"
  if num_gt "${window}" 0 && num_lt "${reported}" "${threshold}"; then
    VERDICT=REJECT_IMPLAUSIBLE
    log "PLAUSIBILITY_REJECT reason=telemetry_window reported_compute_s=${reported} < ${MIN_REPORTED_WINDOW_FRACTION}*observed_window_s=${window}"
    return
  fi
}

run_timed() {
  # run_timed <side> <ws> <attempt>: one gated, telemetry-validated official
  # benchmark run. Sets VERDICT/RUN_DEC/RUN_PRE/RUN_STATS; on ACCEPT the sealed
  # score is at ${OUT_DIR}/score-<side>.json.
  local side="$1" ws="$2" attempt="$3"
  local telem="${OUT_DIR}/telem-${side}-a${attempt}.jsonl"
  local raw="${OUT_DIR}/score-${side}-a${attempt}.stdout"
  local blog="${OUT_DIR}/bench-${side}-a${attempt}.log"
  local sealed="${OUT_DIR}/score-${side}.json"
  local tpid bstatus

  # Identical, residue-free preconditions for every attempt on both sides of
  # the pair: reap prior bench residue, assert none holds the model, wait for
  # load/util to settle -- then the unchanged thermal gate.
  phase_quiesce "${side}-a${attempt}" "${ws}"
  thermal_gate "${side}-a${attempt}" || die 2 "thermal gate timeout before ${side} attempt ${attempt}"

  "${MACMON}" pipe -i "${TELEM_INTERVAL_MS}" > "${telem}" 2>/dev/null &
  tpid=$!

  # No set -e here: benchmark.sh exits nonzero on passed=false, which happens
  # legitimately for band/floor policy failures; verdicts come from the sealed
  # score + telemetry, not the exit code.
  exec_in_ws "${ws}" /usr/bin/env \
    MJOB_WS="${ws}" \
    MLXFAST_OFFICIAL_BENCHMARK_RUN=1 \
    MLXFAST_SKIP_TRANSFORM=1 \
    MLXFAST_WEIGHTS_PATH=weights \
    MLXFAST_CORRECTNESS_GOLDEN_PATH="${ORACLE_REL}" \
    MLXFAST_BENCHMARK_CORRECTNESS_STEPS=0 \
    MLXFAST_BENCHMARK_CHECK_GATES=0 \
    MLXFAST_BENCHMARK_SKIP_TIMED=0 \
    MLXFAST_SCORE_PATH="${SCORE_REL}" \
    MLXFAST_INTEGRITY_PATH="${INTEGRITY_REL}" \
    /bin/bash -c "${INNER_BENCH}" > "${raw}" 2> "${blog}"
  bstatus=$?

  sleep 1
  kill "${tpid}" 2>/dev/null
  wait "${tpid}" 2>/dev/null

  # benchmark.sh exits nonzero on passed=false, which on this box legitimately
  # happens for band/floor policy failures -- the exit code is informational.
  log "BENCH_EXIT side=${side} attempt=${attempt} exit=${bstatus}"

  if ! seal_score "${raw}" "${sealed}"; then
    VERDICT=REJECT_NO_SCORE
    RUN_DEC=""; RUN_PRE=""; RUN_STATS="{}"
    log "RESULT side=${side} attempt=${attempt} verdict=${VERDICT} (no valid score payload on stdout)"
    return 1
  fi

  evaluate_acceptance "${sealed}" "${telem}"
  log "RESULT side=${side} attempt=${attempt} verdict=${VERDICT} decode=${RUN_DEC:-NA} prefill=${RUN_PRE:-NA} stats=$(echo "${RUN_STATS}" | jq -c .)"
  [ "${VERDICT}" = "ACCEPT" ]
}

run_side() {
  # run_side <side> <ws>: acceptance with one gated retry.
  local side="$1" ws="$2" attempt=1
  while [ "${attempt}" -le "${MAX_ATTEMPTS}" ]; do
    if run_timed "${side}" "${ws}" "${attempt}"; then
      echo "${attempt}" > "${OUT_DIR}/attempts-${side}.txt"
      return 0
    fi
    log "RETRY side=${side} attempt=${attempt} verdict=${VERDICT}"
    attempt=$((attempt + 1))
  done
  echo "${MAX_ATTEMPTS}" > "${OUT_DIR}/attempts-${side}.txt"
  return 1
}

# --- Baseline calibration sanity --------------------------------------------------
baseline_binary_check() {
  local ws="$1" want got bin
  [ -s "${BASELINE_CALIBRATION}" ] || return 0
  want="$(jq -r '.binary_sha256 // ""' "${BASELINE_CALIBRATION}")"
  [ -n "${want}" ] || return 0
  bin="$(resolve_binary "${ws}")"
  got="$(shasum -a 256 "${bin}" | awk '{print $1}')"
  if [ "${got}" != "${want}" ]; then
    die 6 "baseline binary hash ${got} != pinned ${want}; baseline tree drifted -- regenerate per RUNBOOK"
  fi
  log "BASELINE_BINARY_OK sha256=${got}"
}

baseline_band_check() {
  # Fail closed if the baseline sample drifted outside the calibration band:
  # a pathological baseline would silently inflate/deflate every paired ratio.
  #
  # PER-METRIC bands (Chain C follow-up, 2026-07-11). One shared band is
  # floored by prefill noise: on this box decode is extremely stable (CV
  # ~0.04%; cross-session ratio 0.974 for the same pinned binary) while
  # prefill at its tiny magnitude (~0.0013-0.0016 s/tok) swings ~17% across
  # sessions (ratio 0.825 observed). A single band wide enough for prefill
  # (+/-0.25) would accept a baseline DECODE up to 25% slow -- and decode is
  # the dominant score denominator (weight 0.75), so a slow baseline decode
  # inflates every candidate's headline speedup. Split the bands: tight where
  # the metric is stable, loose only where the noise actually is.
  # Reads decode_band_low/high + prefill_band_low/high; falls back to the
  # legacy shared band_low/high (older calibration files keep working), then
  # to conservative defaults. All values here are runner-owned (calibration
  # file is manifest-pinned, ratios derive from accepted runs), never in
  # E-notation, so bc comparisons stay correct -- see the num_lt/num_gt note.
  local dec="$1" pre="$2" cal_dec cal_pre rd rp
  local dec_lo dec_hi pre_lo pre_hi fail=""
  if [ ! -s "${BASELINE_CALIBRATION}" ]; then
    if [ "${BASELINE_BAND_ENFORCE}" = "1" ]; then
      die 6 "baseline calibration ${BASELINE_CALIBRATION} missing (required with BASELINE_BAND_ENFORCE=1)"
    fi
    log "BASELINE_BAND_SKIP no calibration file"
    return 0
  fi
  cal_dec="$(jq -r '.decode_seconds_per_token' "${BASELINE_CALIBRATION}")"
  cal_pre="$(jq -r '.prefill_seconds_per_token' "${BASELINE_CALIBRATION}")"
  dec_lo="$(jq -r '.decode_band_low // .band_low // 0.90' "${BASELINE_CALIBRATION}")"
  dec_hi="$(jq -r '.decode_band_high // .band_high // 1.10' "${BASELINE_CALIBRATION}")"
  pre_lo="$(jq -r '.prefill_band_low // .band_low // 0.90' "${BASELINE_CALIBRATION}")"
  pre_hi="$(jq -r '.prefill_band_high // .band_high // 1.10' "${BASELINE_CALIBRATION}")"
  rd="$(echo "${dec} / ${cal_dec}" | bc -l)"
  rp="$(echo "${pre} / ${cal_pre}" | bc -l)"
  log "BASELINE_BAND decode_ratio=${rd} band=[${dec_lo},${dec_hi}] prefill_ratio=${rp} band=[${pre_lo},${pre_hi}]"
  if [ "$(echo "${rd} >= ${dec_lo} && ${rd} <= ${dec_hi}" | bc -l)" != "1" ]; then
    fail="decode_ratio=${rd} outside [${dec_lo},${dec_hi}]"
  fi
  if [ "$(echo "${rp} >= ${pre_lo} && ${rp} <= ${pre_hi}" | bc -l)" != "1" ]; then
    fail="${fail}${fail:+; }prefill_ratio=${rp} outside [${pre_lo},${pre_hi}]"
  fi
  if [ -n "${fail}" ]; then
    if [ "${BASELINE_BAND_ENFORCE}" = "1" ]; then
      die 6 "baseline sample outside calibration band (${fail})"
    fi
    log "BASELINE_BAND_WARN outside band but BASELINE_BAND_ENFORCE=0 (${fail})"
  fi
}

# =================================================================================
# Library / self-test hook: when this file is SOURCED with MJOB_LIB_ONLY=1, stop
# here so a test harness can call the individual functions (plausibility_check,
# telem_loaded_window, evaluate_acceptance, baseline_band_check, ws_head) with
# the real constants and code, without running a live measurement. Inert in
# production -- the variable is unset on the normal runner/operator path, so the
# full pipeline below always runs. (`return` succeeds only while sourced; the
# `|| exit 0` guards the degenerate case of executing this file with the flag.)
if [ "${MJOB_LIB_ONLY:-0}" = "1" ]; then
  # shellcheck disable=SC2317  # `exit 0` is reached only when executed (not sourced) with the flag set.
  return 0 2>/dev/null || exit 0
fi

log "measure-job tag=${TAG} exec_mode=${MEASURE_EXEC_MODE} out=${OUT_DIR}"
preflight
if [ "${PREFLIGHT_ONLY}" = "1" ]; then
  echo "PREFLIGHT_OK" > "${OUT_DIR}/verdict.txt"
  exit 0
fi

[ -n "${CANDIDATE_WS}" ] || usage
CANDIDATE_WS="$(cd "${CANDIDATE_WS}" && pwd)" || die 8 "candidate workspace not found"
check_ws_prereqs "${CANDIDATE_WS}" candidate
CAND_HEAD="$(ws_head "${CANDIDATE_WS}")"

BASE_HEAD=""
if [ "${CANDIDATE_ONLY}" != "1" ]; then
  [ -n "${BASELINE_WS}" ] || die 8 "baseline workspace required (--baseline or BASELINE_WS); use --candidate-only to skip pairing"
  BASELINE_WS="$(cd "${BASELINE_WS}" && pwd)" || die 8 "baseline workspace not found"
  if [ "${BASELINE_CLONE}" = "1" ]; then
    # Clone the pinned pristine baseline into a per-job throwaway so the pristine
    # tree never takes bench-uid writes (H2). cp -c is an APFS clonefile (instant,
    # copy-on-write) when source+dest share a volume; plain -R is the fallback.
    mkdir -p "${BASELINE_CLONE_PARENT}" || die 8 "cannot create baseline clone parent ${BASELINE_CLONE_PARENT}"
    # Sweep throwaways a prior run may have left behind (e.g. SIGKILL before the
    # cleanup trap fired). Safe: one job runs at a time on this box, and every
    # baseline-* here is a discardable per-job clone.
    rm -rf "${BASELINE_CLONE_PARENT:?}"/baseline-* 2>/dev/null || true
    clone_dir="${BASELINE_CLONE_PARENT}/baseline-${TAG}"
    rm -rf "${clone_dir}"
    if ! cp -Rc "${BASELINE_WS}" "${clone_dir}" 2>/dev/null; then
      rm -rf "${clone_dir}"
      cp -R "${BASELINE_WS}" "${clone_dir}" || die 8 "clone of baseline failed"
    fi
    BASELINE_CLONE_DIR="${clone_dir}"
    # The pinned tree is read-only to bench; grant the THROWAWAY clone a bench
    # ACL so the baseline run (as bench) can write its score/tmp there. Without
    # this the run would fail closed against a read-only tree.
    chmod -R +a "${BENCH_WS_ACL}" "${clone_dir}" 2>/dev/null || true
    BASELINE_WS="${clone_dir}"
    log "BASELINE_CLONED to=${clone_dir}"
  fi
  check_ws_prereqs "${BASELINE_WS}" baseline
  baseline_binary_check "${BASELINE_WS}"
  BASE_HEAD="$(ws_head "${BASELINE_WS}")"
fi

# Oracles first (each may cost one gated GPU generation pass on cache miss).
ensure_oracle "${CANDIDATE_WS}" candidate
if [ "${CANDIDATE_ONLY}" != "1" ]; then
  ensure_oracle "${BASELINE_WS}" baseline
fi

# Baseline first, then candidate -- mirrors the challenge timing machine's
# "measure the pinned reference minutes before the candidate" pairing.
if [ "${CANDIDATE_ONLY}" != "1" ]; then
  if ! run_side baseline "${BASELINE_WS}"; then
    die 6 "baseline run rejected (last verdict=${VERDICT})"
  fi
  BASE_DEC="${RUN_DEC}"; BASE_PRE="${RUN_PRE}"; BASE_STATS="${RUN_STATS}"
  baseline_band_check "${BASE_DEC}" "${BASE_PRE}"
fi

if ! run_side candidate "${CANDIDATE_WS}"; then
  die 5 "candidate run rejected (last verdict=${VERDICT})"
fi
CAND_DEC="${RUN_DEC}"; CAND_PRE="${RUN_PRE}"; CAND_STATS="${RUN_STATS}"

# --- Paired ratio + final sealed results -------------------------------------------
if [ "${CANDIDATE_ONLY}" = "1" ]; then
  jq -n \
    --arg tag "${TAG}" \
    --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg cws "${CANDIDATE_WS}" --arg chead "${CAND_HEAD}" \
    --arg cdec "${CAND_DEC}" --arg cpre "${CAND_PRE}" \
    --argjson cstats "${CAND_STATS}" \
    --argjson cattempts "$(cat "${OUT_DIR}/attempts-candidate.txt")" \
    '{ tag: $tag, timestamp: $ts, mode: "candidate-only",
       candidate: { workspace: $cws, head: $chead, attempts: $cattempts,
                    decode_seconds_per_token: ($cdec|tonumber),
                    prefill_seconds_per_token: ($cpre|tonumber),
                    telemetry: $cstats, verdict: "ACCEPT" } }' \
    > "${OUT_DIR}/results.json"
else
  jq -n \
    --arg tag "${TAG}" \
    --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg cws "${CANDIDATE_WS}" --arg chead "${CAND_HEAD}" \
    --arg bws "${BASELINE_WS}" --arg bhead "${BASE_HEAD}" \
    --arg cdec "${CAND_DEC}" --arg cpre "${CAND_PRE}" \
    --arg bdec "${BASE_DEC}" --arg bpre "${BASE_PRE}" \
    --argjson cstats "${CAND_STATS}" --argjson bstats "${BASE_STATS}" \
    --argjson cattempts "$(cat "${OUT_DIR}/attempts-candidate.txt")" \
    --argjson battempts "$(cat "${OUT_DIR}/attempts-baseline.txt")" \
    '
    ($cdec|tonumber) as $cd | ($cpre|tonumber) as $cp |
    ($bdec|tonumber) as $bd | ($bpre|tonumber) as $bp |
    ($bd / $cd) as $ds | ($bp / $cp) as $ps |
    { tag: $tag, timestamp: $ts, mode: "paired",
      candidate: { workspace: $cws, head: $chead, attempts: $cattempts,
                   decode_seconds_per_token: $cd, prefill_seconds_per_token: $cp,
                   telemetry: $cstats, verdict: "ACCEPT" },
      baseline:  { workspace: $bws, head: $bhead, attempts: $battempts,
                   decode_seconds_per_token: $bd, prefill_seconds_per_token: $bp,
                   telemetry: $bstats, verdict: "ACCEPT" },
      paired: { decode_speedup: $ds, prefill_speedup: $ps,
                paired_score: (pow($ds; 0.75) * pow($ps; 0.25)) } }' \
    > "${OUT_DIR}/results.json"

  # Chain C: hard ceiling on the paired speedups. The per-run plausibility_check
  # already rejects a fabricated-fast candidate at run time; this is the final
  # belt-and-suspenders on the sealed RATIO (baseline/candidate). Honest paired
  # speedups on this box are ~1.0-1.1, so a value past MAX_PLAUSIBLE_SPEEDUP is
  # not a real optimization. Fail closed (candidate rejected).
  paired_ds="$(jq -r '.paired.decode_speedup' "${OUT_DIR}/results.json")"
  paired_ps="$(jq -r '.paired.prefill_speedup' "${OUT_DIR}/results.json")"
  # num_gt (awk) not bc: an extreme fabricated ratio can render in E-notation
  # (e.g. jq prints 4.4e+18), which bc mis-evaluates -- see the num_lt/num_gt note.
  if num_gt "${paired_ds}" "${MAX_PLAUSIBLE_SPEEDUP}" || num_gt "${paired_ps}" "${MAX_PLAUSIBLE_SPEEDUP}"; then
    die 5 "paired speedup exceeds plausibility ceiling (decode_speedup=${paired_ds} prefill_speedup=${paired_ps} > ${MAX_PLAUSIBLE_SPEEDUP}); candidate timing implausible"
  fi
fi

# Seal results.json the same way score.json is sealed: a runner-owned sha256
# sidecar over the ratio source the trusted overlay consumes, so any later edit
# of results.json inside a bench-writable tree is detectable (Chain C #4).
shasum -a 256 "${OUT_DIR}/results.json" | awk '{print $1}' > "${OUT_DIR}/results.json.sha256"

echo "ACCEPT" > "${OUT_DIR}/verdict.txt"
log "DONE results=$(jq -c '.paired // .candidate' "${OUT_DIR}/results.json")"
exit 0
