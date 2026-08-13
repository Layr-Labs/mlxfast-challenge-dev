# Qwen-MTP go-live runbook (`qwen3.6-27b-mtp-v1`)

`.github/workflows/qwen-mtp-provision-goldens.yml` points operators here from
four places, including "step A" and "step B" cited in error messages an operator
only ever sees when a dispatch fails closed. Until go-live the reference dangled
at nothing. This is it.

Everything below is work that cannot be done from the repo alone: it needs hidden
material, physical access to the serving box, or a judgement about whether the
anti-cheat is sound enough to rank on.

> **GO-LIVE COMPLETE — 2026-08-13.** Steps A–E below were EXECUTED. This runbook
> is retained as the record and as the procedure for taking the track back
> offline. Current live state: `official_scoring_enabled: true`,
> `reference_baseline.publication_allowed: true`,
> `tokenFidelityGateStatus: "implemented"`,
> `MLXFAST_QWEN_MTP_CALIBRATION_READY: "1"`, normalised decode floor `0.95`.
>
> **THIS TRACK IS BRANCH-TARGETED, PERMANENTLY** (operator design decision
> 2026-08-13). `qwen36-mtp-track` is the track's **base branch**, not a
> migration vehicle: there is no planned merge into `main`, `main` stays the
> DFlash track's base with `benchmark.json` as its manifest, and public
> submissions for this track build against **this branch's** harness. The
> ranked workflow's `qwen36-mtp-track` allowlist arm and the matching entries in
> the two `enforce-trusted-qwen-mtp-*` guard scripts are therefore **permanent**
> (grep `PERMANENT BY DESIGN (2026-08-13)`); do not remove them. Pre-go-live
> "pending / false / inert" phrasing elsewhere in the tree is HISTORY, and so is
> any "at the go-live merge" instruction.
>
> **What that means operationally.** A `submissions/*` dispatch checks its
> trusted harness out of `qwen36-mtp-track` (not `main`), so Yukon must import
> this benchmark with `sourceBranch = qwen36-mtp-track` and base every
> submission branch on that ref — a submission branched from `main` is refused
> by the merge-base check in "Verify submitted commit and modifiable surface".
> `main` keeps fail-closed copies of the three `qwen-mtp-*.yml` workflows as
> **registration stubs** (a `workflow_dispatch` workflow is only dispatchable if
> it exists on the default branch); they are frozen at the inert pre-go-live
> draft and must be neither updated to match this branch nor deleted.

## 0. What is already done and verified (do not redo)

Measured on `m5-max-128gb-3` ("box 3", Apple M5 Max), runner label
`m5-qwen3.6-27b-mtp`. Track contract in
`fixtures/qwen3_6_27b_mtp_track.json`; deploy detail in the operator repo's
`m5-machine-scripts/deploy/qwen36-mtp/DEPLOY-RUNBOOK.md`.

| thing | state |
|---|---|
| pinned baseline | `/opt/bench-runner/baseline/qwen3.6-27b-mtp-v1/current`, installed and calibrated from `e86655528c2abe534aa2e61a454565237801aa2d`. Pinned by PATH and by the signed §9d manifest — **not** by any `MLXFAST_BASELINE_COMMIT` workflow variable. That serial-era mechanism has no reader on this track and was deliberately not introduced. |
| baseline calibration | `/opt/bench-runner/state/qwen3.6-27b-mtp-v1/baseline-calibration.json`, paired schema, target `lowsim-prose-qwen-v1`. Bands the **serial denominator** of the pair (depth 0, same binary, same worker), not the baseline's absolute speed. `decode_tokens: 512` is mandatory and does not inherit: seconds/token moves ~2.1x between a 16- and a 1024-token window on this model, so a band without its window is a band against nothing. |
| MTP head | separately pinned tree, `mlx-community/Qwen3.6-27B-MTP-4bit @ 83795d54`, 8 files / 258490810 B, resident on **both** legs of the pair so its residency cost is charged to the denominator too. Only the drafting differs. |
| depth | candidate depth **2**; serial control depth **0** (MTP off — not 1; depth 1 still drafts and accepts ~70%, which understates the gain). |
| oracle cache | pinned **as empty**. Any file appearing in `state/qwen3.6-27b-mtp-v1/oracle-cache/` is drift. Correctness is the hidden golden, re-audited into `parity_all_ok`. |
| janitor audit | clean; box signed and serving. |

## Step A — the hidden timed-prompt pool (DONE)

`fixtures/qwen3_6_27b_mtp_track.json` → `timed_prompt_pool` holds **8 distinct
entries**, one per domain (beagle, botany, drama, essays, medicine, plutarch,
republic, travel), each pinned by `sha256` and `bytes`, each uploaded to R2 under
`correctness_prompts/qwen3.6-27b-mtp-v1/` and re-verified by digest after
download.

Each entry **is** a timed MTP reference-rows golden — the object the ranked job
hands to `mtp-timed --golden` — not the prose prompt it was seeded from. Each was
generated on box 3 with `mtp-verify --emitted <plan> --generate 513` from a
different 512-token prose seed, and each returned
`reference_self_consistent=true` with 513 rows.

A seed plan must be **pre-tokenized real prose**. Never `--seed-generate`: that
produces the degenerate greedy self-continuation the correctness contract
condemns, and the provisioning job refuses argv containing the flag.

### Why the pool exists, and why it must be varied

Acceptance is strongly prompt-dependent (measured 0.344–0.585 across the pool),
so the raw paired ratio spans **0.7623** (drama) to **1.0845** (medicine) — a
1.42x span for *identical code*. Sampling one prompt per run and publishing the
raw ratio would hand a candidate drawn on `medicine` a 1.42x advantage over one
drawn on `drama`. That is precisely what the normalised score removes: each
entry's own measured `noop_decode_speedup` is pinned in the fixture, the raw
ratio-of-means is divided by it, and every entry normalises to 1.0 by
construction.

### Scoring semantics: median of 8 (operator-ratified 2026-08-13)

A ranked run **times the whole pool**, not one sampled prompt. Formally it draws
one prompt per *collection*; each of the 8 collections is a singleton today, so
all 8 run.

```
per prompt p:  raw_p  = mean(serial s/tok over p's pairs)
                      / mean(mtp    s/tok over p's pairs)
               norm_p = raw_p / p.noop_decode_speedup
published:     score  = median(norm_1 .. norm_8)
```

**Why normalisation was not enough on its own.** Per-prompt normalisation
cancels the *baseline's* prompt-dependence — every entry's own no-op maps to 1.0
by construction. It cannot cancel the *candidate's* prompt-dependent improvement
profile: a change that helps `medicine` and not `drama` still scored differently
depending on the draw. Single-draw scoring was therefore a lottery over which
prompt a submission happened to be good at. The median cancels that by
construction, and being a median rather than a mean it also blunts any residual
error in a single prompt's pinned reference.

**The median rule is load-bearing because 8 is even:** the score is the mean of
the two central order statistics, *not* the lower-median rule used by the
per-pair `mtp_decode_speedup_median` diagnostic and by the CLI's own p50 fields.
`results.json` and `score.json` both name the rule they used.

**Pair budget is per prompt.** The ranked default is k=1 — 8 prompts × 1 pair =
8 pairs / 16 timed phases, roughly 45 min of timed work. The old run-level 3/4
budget bought pair-averaging on a single prompt; the median over 8 prompts does
that job and also cancels the lottery. At the measured 0.77% pair noise, k=1
puts the score's dispersion near 0.34%. Raise it with the wrapper's
`--pairs-per-prompt` if calibration data asks.

**The serial denominator band is pooled across prompts.** The serial control is
depth 0 — 512 plain forwards — and is prompt-invariant by construction, since
the measured pool spread lives entirely in the acceptance rate, which only the
numerator sees. The band therefore checks the mean of *all* serial pairs of
*all* prompts against the one top-level calibration: at 8+ pooled pairs that is
a strictly stronger test than the 4-pair band it replaces, and no re-authoring
of the installed calibration is required. Per-prompt serial means stay in the
sealed breakdown, so a denominator that began to move with the prompt would be
visible.

### Why each prompt-window still pays its own process

Batching all 8 prompts into one model residency per leg would be faster. It is
**not available**, and the reasons are worth recording because they also define
what taking that path would cost:

- the worker builds exactly one `Qwen36MTPBlockSession` before the protocol
  hello and guards `mtp_decode_begin` with `!state.began`; there is no reset
  request kind anywhere in the tree (`QwenRuntimeMTPWorker.swift`);
- the parent driver takes one golden per call and spawns a **second** worker
  afterwards for the post-window reference replay
  (`QwenRuntimeMTPDriver.swift`), so a batched leg would have to batch the audit
  replay too;
- the CLI's option parser rejects a repeated `--golden` outright ("duplicate
  option") — the batched argv shape is a hard error, not last-wins;
- **decisively**, the serial leg executes the *pinned baseline tree's* own
  prebuilt `mlxfast-swift` and its sibling worker, so no repo-side protocol
  change reaches it. Batching needs a new baseline build, a new signed §9d
  manifest and a full re-calibration;
- and it would change the measured quantity anyway: only the first
  prompt-window of a batched leg carries the cold working set, while the
  installed band was authored against per-window process start + allocator
  clear + warm + a seed prefill charged *inside* the clock.

**One consequence for the stall guard**, recorded because it constrains that
future path: under batching the post-prefill warmup belongs to the *leg*, not to
each window, so `STALL_EXCLUDE_FIRST_BLOCK` would have to become leg-aware —
exclude `block[0]` of window 1 only. Excluding `block[0]` of every window opens
8 blind spots at exactly the window boundaries where a stall is most likely;
including `block[0]` of window 1 reinstates the 5.72–5.90x false rejection the
calibration removed. Either way the CLI report would have to declare its
position within the leg. On the per-invocation shape kept here the question does
not arise: every window *is* a leg, and the guard is unchanged.

Batched legs remain a documented future path. Taking it is a measurement-
architecture change, not a scoring change.

## Step B — freeze and pin the hidden goldens (DONE)

Three classes of hidden object, all content-addressed, all pinned by digest AND
byte count in the ranked workflow's job env, with the R2 object key embedding the
same digest so key and digest cannot drift apart:

- **The hidden MTP correctness golden** — `MLXFAST_QWEN_MTP_CORRECTNESS_GOLDEN_*`.
  Generated on box 3 against the Qwen tower with the pinned head, from the hidden
  correctness prompt's own 512 seed tokens; re-verified by a
  `mtp-verify --tokens 512 --mtp-depth 2` pass returning
  `all_tokens_matched=true`, `parity_all_ok=true` and a CLOSED row ledger.
- **The reused serial correctness golden** — `MLXFAST_RAW_CORRECTNESS_GOLDEN_*`,
  pinned at `faf1a679`. It carries the derived `.benchmark` oracle the ranked
  gates phase requires; a golden without one fails closed at "Correctness and
  gates" with `benchmark golden file must contain a benchmark oracle`.
- **The GPQA reference cases** — `MLXFAST_GPQA_REFERENCE_*`, pinned at
  `c8bce79c`, with `accepted_responses` filled from the reference model's own
  captures. See "Known limits" below; this one has a real caveat.

Upload happens **after** the digest is pinned, never before. A dispatch that
reaches the download step with an unresolvable key is the intended fail-closed
ordering.

## Step C — calibration and the normalised floor (DONE)

The serial denominator was banded from gated bootstrap sessions on the parked
box, at the ranked window (`--tokens 512`, `--mtp-depth 2`), in the thermal/fan
regime the track serves in. `GPU_LOADED_UTIL` is deliberately left at the
wrapper's own `0.70` default: the serial leg's GPU-util median is 0.7935 and
never sustains 0.9, so the inherited 0.95 starves `MIN_LOADED_SAMPLES` and
false-rejects honest runs. `TELEM_INTERVAL_MS=100` is likewise not optional on
box 3.

The normalised decode floor is **0.95**, pinned equal in the ranked workflow env
(`MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR`, the enforcing site) and in
`benchmark.qwen-mtp.json` → `scoring.decodeSpeedupFloor` (documentation).
`QwenMTPTrackNamingTests` pins the two together.

**Prefill is measured but UNSCORED on this track.** `mtp_decode_speedup` is a
decode-only ratio-of-means; the seed prefill is charged *inside* the decode
window, identically on both legs; and the measurement wrapper seals
`prefill_component: "none"` in its `results.json`. A prefill figure is recorded
for historical tracking only. A reader who finds the prefill constant and assumes
it participates in scoring will mis-tune the track.

**The floor applies to the normalised MEDIAN** (see "Scoring semantics" under
step A). An unmodified candidate medians to 1.0 by construction, so 0.95 remains
the same 5% margin below parity it always was — the number did not move when the
aggregation did.

### The CALIBRATION acceptance window (derived 2026-08-13, not inherited)

This is the operator's own accept/reject band for **calibration dispatches** —
runs of a known baseline-equivalent candidate, where the published median should
land at parity. It is **not** a submission-facing gate: the 0.95 normalised
floor, the `[0.95,1.05]` serial band and `MAX_PLAUSIBLE_SPEEDUP` are untouched by
anything in this section.

The inherited serial-era ±1% was never derived for this track and is wrong for
it in both directions. The window below is derived from measured variance.

**Inputs.** Three prompts have two *independent sessions* each (a ranked dispatch
and a re-measurement session): beagle +0.228%, drama −0.261%, medicine +0.023%.
The sd of a single session estimate is
`sqrt(mean(d²)/2)` = **0.142%**.

**Decomposition.** Typical within-run per-pair ratio sd is 0.150%, so a 4-pair
mean carries 0.075% from pair noise alone. The residual
`sqrt(0.142² − 0.075²)` = **0.120%** is *session-level* — thermal and frequency
state that every prompt in a run shares. It does not average down with more
pairs, which is why `pairs_per_prompt` beyond about 2 buys very little.

**Propagating to the median of 8.** The session term is **common-mode**: a ranked
run times all 8 prompts in one thermal session, so it shifts every prompt
together and passes straight through the median. Only the independent per-prompt
term is reduced (≈0.443× for the even-n mean-of-two-central rule at n=8). At the
ranked `pairs_per_prompt: 1`:

```
sd(median) = sqrt( 0.120²  +  (0.443 × 0.150)² )  =  0.135%
```

Monte Carlo over 200k trials with the same inputs agrees: **0.135%**. Raising k
to 4 only reaches 0.124% — confirming the common-mode term dominates.

**The window.** ±3σ on the point estimate is [0.9959, 1.0041]. That estimate
rests on only three paired comparisons, so σ is inflated by 1.8× (roughly the
upper 95% χ² bound at ~3 df) before rounding outward:

> **Calibration acceptance window: 0.992 – 1.008** (parity ±0.8%).

A calibration run landing outside it is a stop-and-investigate, not a retry.
**Re-derive it** once more paired sessions exist — the 1.8× inflation is
compensating for a thin sample and should shrink, not persist.

For contrast, the same inputs give a *single-draw* window of ±0.58%, and that is
before per-prompt reference error, which is what the old single-shot references
contributed and what made a 1.1% miss look normal. Median-of-8 is both tighter
and robust to one bad reference.

## Step D — flip the two trusted-contract fields

The switch itself. In **one** commit, because the tests and docs are wired to
fail otherwise, and that friction is deliberate in both directions:

1. `fixtures/qwen3_6_27b_mtp_track.json`: `official_scoring_enabled` → `true`
   and `reference_baseline.publication_allowed` → `true`. The enablement guard
   refuses a RANKED dispatch while **either** is false, and a gates-only dry run
   publishes no score.
2. `.github/workflows/qwen-mtp-ranked-benchmark.yml`:
   `MLXFAST_QWEN_MTP_CALIBRATION_READY` → `"1"`. A hard-pinned literal, never an
   expression and never a dispatch input. It gates only TIMED runs, so
   `run_benchmark=false` correctness dispatches keep working while it is closed —
   that is the migration shape.
3. `benchmark.qwen-mtp.json`: `tokenFidelityGateStatus` → `"implemented"` and
   `scoring.decodeSpeedupFloor` → the enforced floor.
4. `Tests/MLXFastTests/QwenMTPTrackNamingTests.swift`: the armed-posture
   assertions.

Taking the track back OFFLINE means editing all four again in one commit.

## Step E — verification dispatch

`gh workflow run qwen-mtp-ranked-benchmark.yml --ref qwen36-mtp-track -f run_benchmark=true`

Runs serialise on the single `m5-qwen3.6-27b-mtp` runner regardless of
concurrency group, so dispatch **sequentially** rather than stacking the queue.
Between runs, confirm the quarantine flag is absent.

What actually gates a ranked run — budget effort accordingly. The in-harness
acceptance band does **not**: the gates pass with `MLXFAST_BENCHMARK_SKIP_TIMED=1`
(ratio exactly 1.0), and the measure wrapper explicitly tolerates
`acceptance band failed*` / `performance floor failed*`. The real guardrails are
the speedup floors in `overlay-paired-timing.sh`, `baseline_band_check` against
the on-box calibration, `plausibility_check`, and `MAX_PLAUSIBLE_SPEEDUP`.

## Rollback

Revert the Step-D commit; the old pins, fixtures and `editablePaths` come back
intact. Box-side rollback is
`sudo ./install-qwen-mtp-track.sh --execute --rollback` in the operator repo, and
deliberately does **not** remove the R2 uploads (immutable, content-addressed),
the `~gaj` staging copies, or the physical baseline tree.

If the box quarantined, read `/opt/bench/quarantine.flag.drift` for the full
`< baseline / > live` diff, fix the cause, then
`sudo /opt/bench/janitor.sh --clear-quarantine`. **Never reboot a box as a
troubleshooting step** — an OS update once reset `/etc/pf.conf`'s anchor lines
and the box served with open bench egress. Park, inspect, fix, re-sign.

## Known limits to accept, or fix first

**The semantic GPQA gate cannot reject a constant-"A" answerer.** Every
`answer_key` in the hidden GPQA fixture is `"A"` while each prompt presents four
options, and the reference model is measurably position-biased toward option A —
it tracks the correct option only ~2/9 when option order is rotated. Since
`accepted_responses` was filled from the reference model's own captures, the gate
now measures **fidelity to the reference implementation**, not question-answering
accuracy. 8 of the 9 reference captures are `"A"`, so a constant-"A" answerer
scores exactly 8 of 9 — which means a min-pass floor of 8 does **not** reject it.
Raising the floor to 9 is not the fix either: it would delete the error budget
that absorbs judge nondeterminism. **The real fix is shuffling option order,
which is organizer material and is deliberately not done in this repo.** Tracked
operator-side.

**The stall guardrail's first block is warmup, not a stall.** As measured,
`max_block_request_seconds` is the first block after the seed prefill and is flat
to within 1% across 8/32/128/512-round windows (serial ~0.19 s, depth-2
~0.267 s). At the 512 window the serial leg therefore trips the 4x rule
(measured 5.72–5.90x) while the depth-2 leg does not (3.30–3.36x). The guardrail
should exclude the first block.

**The pool's character is provisional.** Acceptance 0.344–0.585 on moderate prose;
a harder low-similarity pool would move the numbers. Re-derive the depth and the
floor across the WHOLE pool if the pool changes — a depth tuned to one prompt is
the same trap as a floor tuned to one prompt.

**The track identity is not formally ratified.** `qwen3.6-27b-mtp-v1` /
`mlxfast-challenge-dev-qwen-mtp` / `lowsim-prose-qwen-v1` are in use and embedded
in R2 object keys and goldens, and those uploads are content-addressed and
irreversible.
