# Thermal Variance in Timed Benchmarks (tenki M4 Pro) — Investigation & Recommendation

Status: investigation complete (2026-07-07). Operator-facing analysis of
benchmark timing variance on the Blacksmith/tenki M4 Pro runner class, why it
leaks past the paired baseline, and the recommended harness fix. This is not a
submission and touches no editable surface; it is a record for whoever owns the
benchmark window / ranked workflow.

## TL;DR

- The prefill "variance" (raw CV ~34%, max/min ~2.2× across machines) is **thermal
  throttling, not host lottery**. The same physical machine runs prefill ~2×
  slower hot (≈0.020 s/tok) than cool (≈0.010 s/tok).
- **Decode also drifts thermally** — ~0.6–0.8 %/min, up to +20 % over ~25 min of
  sustained load. The clean-looking 3.5 % cross-machine decode CV was an artifact
  of measuring each machine once, early; repeat measurement reveals ~10–20 %
  within-host.
- The drift is **directional in time** (baseline measured cooler, candidate
  measured later/hotter), so the paired baseline — which cancels *common-mode*
  host differences — does **not** cancel it. It biases `prefill_speedup` (and to a
  lesser degree `decode_speedup`) below 1.
- **Excluding prefill is not sufficient** (decode drifts too). **A 2-minute idle
  settle does not work** (cooldown constant is > 2 min; cooling is slow). The
  heater is **sustained inference (the ~5-min correctness pass)**, not build,
  transform, or the ~30–50 s timed window itself.
- **Recommended fix: reorder the ranked pipeline so the two timed windows
  (baseline and candidate) run back-to-back, with both correctness passes moved
  after both timed windows.** This cuts the baseline↔candidate timed-measurement
  gap from ~6–7 min to ~1–2 min (bounded below by the candidate's model load),
  reducing the thermal residual ~3–4×. Pair it with **median-of-N** on the timed
  window to absorb transient spikes. No idle wait required.

## Context

- Model: Gemma 4 31B 4-bit, dense, text tower only (~17 GB, fully RAM-resident).
- Score: `decode_speedup^0.75 * prefill_speedup^0.25`, per-axis 0.95 floors.
- Scoring uses a **paired baseline**: the pinned reference is built and timed on
  the same runner, same session, minutes before the candidate; speedups are the
  ratio. This cancels common-mode host/hour variance (see
  `benchmark-window-freeze.md`). It does **not** cancel a time-ordered drift
  between the two measurements — which is what thermal throttling is.
- Runner class: `blacksmith-12vcpu-macos-26` and `tenki-macos-latest-xlarge` are
  both `VirtualMac2,1` = Apple M4 Pro (Virtual) VMs, 12 vCPU. Measurements here
  are on tenki-xlarge; the thermal behavior is a property of the shared physical
  host, so treat magnitudes as indicative, not exact, for the ranked runner.

## What was measured

All probes ran the unmodified Gemma `main` reference on `tenki-macos-latest-xlarge`
via ad-hoc workflows on branch `gemma-tenki-decode` (baked ~/.cache Gemma
checkpoint, no re-download). Prefill is measured identically in every mode: one
cold 512-token forward.

| probe | run | what it showed |
|---|---|---|
| steady decode (`--local-submit`, 1023-step) | 28818728137 | decode CV **3.5 %** / max-min 1.10× (each machine once); prefill CV **33.8 %** / 2.23× |
| decode consistency (repeat 1023-step) | 28824046255 | within-host decode CV **14.4 %**, cross-host **2.2 %**; m1 drift **+20 %** (r=0.76), m3 +9.3 %, m2 a transient spike (0.17) |
| prefill variance (`--local-iterate`, timestamped) | live m5/m6 | cool prefill **0.0104–0.0113**, hot **0.018–0.024** (~2×); onset ~pass 2–3 (~5–10 min) |
| settle A/B (2-min idle before even windows) | 28832851608 | no-wait mean 0.0194 (CV 32 %), settle mean **0.0205** (CV 20.8 %) → settle **not cooler**; only the fresh first window is cool |
| adjacency oracle pilot (timing-only via self-built oracle) | 28836241400 | oracle path works; timing captured on floor-fail; **windows ~5 min = load-bound**; two 128-step decodes ~5 min apart nearly identical (0.170766 vs 0.170764); prefill ramped 0.011→0.025 |

Calibrated constants (blacksmith M4 Pro, `Constants.swift`): baseline decode
0.131727461265625, baseline prefill 0.01010573933984375 s/tok.

## Findings

### 1. Prefill variance is thermal throttling
Timestamped back-to-back windows show the *same machine* stepping from cool
(~0.010 s/tok, first window) to a hot plateau (~0.020 s/tok) within 1–3 windows
(~5–10 min), and staying there. Cross-machine spread (2.2×) is the same effect
sampled at different thermal states, not different hardware.

### 2. Decode also drifts (not thermally immune)
Repeated 1023-step steady decode on one host rises monotonically with time on
machine (m1: +20 %, r=0.76). The long window is "always hot" *within one
measurement*, which is why a single measurement per machine looked stable (3.5 %);
across repeated measurements the accumulating heat shows up as ~10–20 % within-host
CV. Decode tolerates a given gap better than prefill (two 128-step decodes 5 min
apart were ~identical in the pilot), but it is not flat.

### 3. The heater is sustained inference, not setup
The first timed window on a fresh machine is cool even though build + transform
already ran — so build/transform do not throttle the GPU. The ~5-min correctness
pass (256+ teacher-forced steps) is what heats it. The timed window itself
(~30–50 s of compute) heats little.

### 4. Windows are model-load-bound (~5 min), not correctness-bound
Skipping correctness entirely (timing-only via a self-built benchmark oracle)
still gave ~5-min windows (308 s process). The cost is dominated by 17 GB model
load + preflight + worker spawn. Consequence: **no separate-process probe can
measure a sub-~5-min gap**, so the true adjacent gap (~1–2 min) is only reachable
by extrapolation — or by an in-harness change that measures twice from one
resident model (not possible across two different code checkouts).

### 5. Why it leaks past the paired baseline
Score is `baseline/candidate`, both on the same host. Common-mode host speed
cancels. But the two measurements are **~6–7 min apart** today (baseline
correctness sits between the baseline timed window and the candidate timed
window), and the host is monotonically heating over that window — so the
candidate is measured hotter than the baseline. That difference does **not**
cancel; it biases the ratio.

## Recommended fix

**Adjacency reorder** (harness / `benchmark-timing-or-gates.yml`, operator
surface — not the submission editable paths):

1. Do all prep first: build baseline, build candidate, transform both.
2. Run the **two timed windows back-to-back**: baseline timed prefill+decode,
   then immediately candidate timed prefill+decode.
3. Run **both correctness passes after** both timed windows.

This collapses the baseline↔candidate timed gap from ~6–7 min to ~1–2 min (the
candidate's model load, which cannot be removed because baseline and candidate
are different code = different processes). Expected residual reduction ~3–4×,
concentrated on prefill (the drifting axis). It requires separating "timed
window" from "correctness" in `benchmark.sh` and sequencing the two timed windows
adjacently in the workflow. It does **not** change the charged work, so it is not
a re-baseline — but confirm the frozen-window invariants in
`benchmark-window-freeze.md` still hold (one validated seed prefill + N validated
decode steps; no identical repeated charged forward).

**Plus median-of-N on the timed window** to absorb transient spikes (e.g. the
one-off 0.17 decode). Reorder fixes drift; it cannot fix random spikes.

Bounds to be honest about: the gap cannot go below the model-load time, so the
reorder reduces but does not zero the thermal residual. If sub-~1 % is required,
the deeper option is to hold one resident model and measure baseline+candidate
from it — which is impossible across two checkouts and would be a much larger
harness change.

## Rejected alternatives

- **Exclude prefill (weight 0).** Simple (`Constants.swift` weights + freeze
  doc/test, no re-baseline), but decode also drifts, so it does not reach the
  target; and it removes a real optimization axis and lets submissions regress
  prefill for free. At most, *down-weight*.
- **2-minute blanket idle settle.** Empirically insufficient (run 28832851608):
  settle windows were not cooler than no-wait. Cooldown constant > 2 min; a longer
  settle would work only at large, per-measurement time cost, applied to both
  sides — expensive and unproven.
- **Warm-up-then-measure-second (same prompt).** Rejected on soundness: an
  identical untimed warmup lets editable code memoize one forward and serve the
  timed one for free (the "reclaimable warmup" the freeze doc already removed);
  also measures a warm, non-production prefill. See `benchmark-window-freeze.md`.

## Reproduce / validate

Ad-hoc workflows on `gemma-tenki-decode` (dispatch-only unless noted):
`gemma-decode-consistency.yml`, `gemma-prefill-variance.yml`,
`gemma-settle-test.yml`, `gemma-adjacency-oracle.yml`. The last builds a
benchmark oracle on the runner (`generate-golden` → assemble `BenchmarkGolden`)
so the timing-only path runs without the private golden. To get the real
128-step decode + prefill residual-vs-gap curve (gaps ≥ 5 min, then extrapolate):
`gh workflow run gemma-adjacency-oracle.yml --ref gemma-tenki-decode -f machines="[1,2]" -f passes="8"`.

The definitive validation of the fix is to implement the reorder and compare the
scored `prefill_speedup` / `decode_speedup` CV across repeated ranked runs
against today's bundled order.
