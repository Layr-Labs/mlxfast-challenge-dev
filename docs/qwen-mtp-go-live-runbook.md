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
> **The merge to `main` is DEFERRED.** Everything landed on the
> `qwen36-mtp-track` ref, so the ranked workflow's TEMPORARY `qwen36-mtp-track`
> allowlist arm (grep `TEMPORARY (2026-08-13)`) is still load-bearing and must
> be removed at the merge, together with the matching entries in the two
> `enforce-trusted-qwen-mtp-*` guard scripts. Pre-go-live "pending / false /
> inert" phrasing elsewhere in the tree is HISTORY.

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
so the raw paired ratio spans **0.756** (drama) to **1.0915** (medicine) — a
1.44x span for *identical code*. Sampling one prompt per run and publishing the
raw ratio would hand a candidate drawn on `medicine` a 1.44x advantage over one
drawn on `drama`. That is precisely what the normalised score removes: each
entry's own measured `noop_decode_speedup` is pinned in the fixture, the raw
ratio-of-means is divided by it, and every entry normalises to 1.0 by
construction.

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
