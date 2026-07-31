# DFlash go-live runbook (`laguna-xs-2.1-dflash-v1`)

`.github/workflows/dflash-benchmark.yml` points operators here from four places,
including "step B", and until now the document did not exist. This is it.

Everything below is the work that CANNOT be done by an agent: it needs hidden
material, a judgement about whether the anti-cheat is sound enough to rank on, or a
change to an operator-owned measurement contract.

## 0. What is already done and verified (do not redo)

Measured on M5-C, 2026-07-30. Contract detail in
`docs/dflash-track-correctness-contract.md` (Amendments 1-25).

| thing | state |
|---|---|
| harness end-to-end | `ACCEPT`, 4/4 pairs, `dflash_decode_speedup` **0.8705** (median 0.8712, min 0.8658), band check LIVE, sealed `results.json`. **That was a DIFFERENT golden.** On the ranked hidden golden a no-op measures **0.5493** (4/4 pairs, `CALIBRATION_OK`, `PARITY_OK`) — see Amendment 26 |
| pinned baseline | `/opt/bench-runner/baseline/laguna-xs-2.1-dflash-v1/<sha>` + `current` symlink, 23 GB, weights APFS-cloned from the serial baseline |
| baseline calibration | `/opt/bench-runner/state/laguna-xs-2.1-dflash-v1/baseline-calibration.json`, authored by `/Users/gaj/author-dflash-calibration.sh` (the wrapper does NOT write it). **The band is only valid at the decode token count it was measured at** — the seed prefill is charged inside the decode window, so the same baseline reads 0.0201 s/token at 128 tokens and ~0.0148 at 512. It must record `decode_tokens`; `measure-dflash-job.sh` fails closed if that field is absent or disagrees with the run. Re-author whenever the ranked window OR the pinned baseline binary changes — see Amendment 27. |
| runner | `m5-laguna-dflash-3-*` online on **mlxfast-challenge-dev**, labels `[self-hosted, m5-laguna-dflash]` only |
| dispatch chain | verified: job scheduled -> host preflight -> trusted context -> enablement guard fails CLOSED on the contract |
| decode floor | **0.52** (re-derived 2026-07-31 from a measured no-op; supersedes 0.80, which rejected a correct no-op), agreeing in all five sites: `benchmark.dflash.json` manifest, contract fixture, workflow comments, workflow env `MLXFAST_DFLASH_DECODE_SPEEDUP_FLOOR`, and `MIN_ACCEPTED_SPEEDUP` in the box wrapper. Rationale is Amendment 26. **PROVISIONAL** — re-derive from the worst no-op across the pool once `timed_prompt_pool` is populated. |
| janitor audit | clean after re-signing; re-verified on all three boxes 2026-07-31 |
| hidden goldens | ONE pair provisioned, uploaded and pinned (correctness `185394`b, bench `185433`b). R2 keys settled by probe: the prefix is `correctness_prompts/laguna-xs-2.1-dflash/`, with NO `gautham-experiments` segment -- that is the BUCKET, carried by `R2_BUCKET_ENDPOINT`. Pinned by `DFlashGoldenKeyTests`; probe with `.github/workflows/dflash-probe-r2-keys.yml` before changing. |

## Step A — hidden timed-prompt pool (BLOCKING)

`fixtures/laguna_xs_2_1_dflash_track.json` -> `timed_prompt_pool` holds **ONE**
entry as of 2026-07-31 (it was `[]` when this runbook was written). That is
enough for a gates-only dry run and **not** enough to rank: the workflow's
"Select hidden DFlash timed target from the pool" step fails closed on an empty
pool AND refuses a run with `run_benchmark=true` while the pool is below **8**.
So **7 more entries are still required**, and the single existing entry must not
be mistaken for readiness.

Requirements, from the fixture's own `timed_prompt_pool_note`:

- **At least 8** distinct hidden timed targets.
- **Varied prompt length and domain**, so acceptance-rate tuning cannot generalise
  across the pool.
- Each entry is `{r2_path, sha256, bytes}`, verified byte-for-byte after download.
- **Hashes are only ever pinned from the uploaded objects**, never pre-filled
  off-box.

Two measured reasons the variety requirement is load-bearing, not boilerplate:

1. A greedy self-continuation of the model is degenerate — 122 distinct tokens in
   512, no top-2 logit gap below 1.8 — and every gate on this track passed on such
   material while failing honest work on real prose (Amendment 10).
2. Draft acceptance is ~100% on repetitive text and **69%** on varied prose, which
   moves the score from 1.117x to 0.840x (Amendment 11). A pool of easy prompts
   would advertise a speedup that does not exist.

### What a pool entry is made of: a pre-tokenized prose SEED

A `timed_prompt_pool` entry's `r2_path` points at a frozen **golden**, and step B
builds that golden from a **seed**. Both are operator material and both live in
R2.

The trusted binary links no tokenizer, so a seed is **token ids, not text**: an
R2 object holding a seed-only `DFlashEmittedPlan`,

```json
{ "seed_tokens": [ 1234, 5678, ... ], "emitted": [] }
```

where `seed_tokens` is the tokenization of **real prose**. The public fixture
`correctness_prompts/public_longcopy_gate_english_512_256.json` →
`cases[0].prompt_tokens` is a checked-in example of the shape and the character
(512 tokens, 276 distinct English-prose tokens); the hidden seeds are yours and
must never be derived from the model's own greedy continuation — see step B, and
Amendment 10.

Provision one such seed object per intended pool entry (at least 8, varied in
length and domain), plus one more for the untimed correctness golden. Then run
step B once per pair. If step B's job reports

> `no operator seed named ... Provision them per docs/dflash-go-live-runbook.md step A`

it is this list that is missing.

## Step B — freeze and pin the IT-target goldens (BLOCKING)

The workflow refuses to run with an empty hidden-golden pin:

> `hidden DFlash golden pin <name> is empty; freeze the IT-target goldens (go-live
> runbook step B) and pin them here before enabling the track`

**Do not run `dflash-reference` by hand for this.** Dispatch
`.github/workflows/dflash-provision-goldens.yml`, which does the whole of step B
inside a job bound to the `benchmark-private-prompts-v2` environment — the only
place the R2 credentials exist. (Generation and upload cannot be split: the
credentials are GitHub environment secrets injected per run, and are on no dev
machine and on no runner's disk. That is why run 30604267251 hit
`404 NoSuchKey` with every other gate green — nothing had ever uploaded the
objects.)

```
gh workflow run dflash-provision-goldens.yml --repo Layr-Labs/mlxfast-challenge-dev \
  --ref main \
  -f confirm_provision_goldens=true \
  -f correctness_seed_r2_path=<R2 key of a prose seed plan> \
  -f correctness_object_path=<R2 key to write the correctness golden to> \
  -f bench_seed_r2_path=<R2 key of a DIFFERENT prose seed plan> \
  -f bench_object_path=<R2 key to write this timed golden to>
```

It generates both goldens from the **pinned baseline tree**, verifies them,
uploads them, **re-downloads each object and computes `sha256` + `bytes` from the
downloaded bytes**, and prints the four pins. It does not edit
`dflash-benchmark.yml`: apply the pins by hand, in a reviewed commit. Re-dispatch
once per `timed_prompt_pool` entry (step A wants at least 8, each from a
different prose seed).

**Already done once** (2026-07-31): the correctness golden and the first timed
golden are provisioned, uploaded and pinned. Do not redo that pair; dispatch this
for entries 2-8.

**Object paths are keys, not bucket-qualified paths.** Pass
`correctness_prompts/laguna-xs-2.1-dflash/<name>.json`, with **no**
`gautham-experiments/` segment -- `gautham-experiments` is the bucket and is
carried by `R2_BUCKET_ENDPOINT`. This cost three ranked dispatches to learn, in
both directions; probe run 30613434387 settled it. If a key is ever in doubt, run
`.github/workflows/dflash-probe-r2-keys.yml` (about a minute) instead of a
dispatch (30-40 minutes).

### The seed must be PRE-TOKENIZED REAL PROSE. Never `--seed-generate`.

This step used to read, in full: "`dflash-reference` builds them;
`--seed-generate N` extends a seed and `--generate N` produces the emitted
chain." **That instruction was wrong and it contradicted this track's own
correctness contract.**

`--seed-generate N` extends the seed by N **reference-generated** tokens — greedy
self-continuation. `docs/dflash-track-correctness-contract.md` **Amendment 10**
measured exactly that material and condemned it:

| golden | seed len | distinct seed tokens | rows w/ top-2 gap < 0.25 | min top-2 gap |
|---|---|---|---|---|
| `seam-512-golden.json` | 512 | 122 | **0** | 1.875 |
| `seam-b-golden.json` | 600 | 122 | **0** | 2.625 |
| `seam-a-golden.json` | 509 | 122 | **0** | 1.875 |
| `varied-512-golden.json` (prose) | 512 | 317 | 3 | **0.0000** |

Greedy self-continuation degenerates into repetition; the near-tie regime
Criterion E exists to handle is never entered; draft acceptance measures ~100%
against 69% on varied prose; and **Amendment 11** prices the difference at
**1.117x versus 0.840x**. A ranked golden built that way would advertise a
speedup that does not exist. The hidden ranked prompts are prose.

So: **the seed is operator-supplied, pre-tokenized real prose.** The trusted
binary links no tokenizer, so a seed arrives as **token ids**, not text — an R2
object holding `{"seed_tokens": [...], "emitted": []}`. The shape of a
legitimate one is visible in the public fixture
`correctness_prompts/public_longcopy_gate_english_512_256.json` →
`cases[0].prompt_tokens` (512 tokens, 276 distinct: the tokenization of real
English prose). Hidden prose seeds come from the operator's private store; the
provisioning job never invents one, and **fails closed** with no seed named
rather than falling back to `--seed-generate`.

This is enforced, not merely written down. `.github/scripts/check-dflash-golden-degeneracy.sh`
screens the operator's seed **before** generation and both goldens **after** it,
against thresholds derived from the table above (seed variety ≥ 0.40 distinct
fraction; at least one row with a top-2 gap < 0.25; minimum top-2 gap ≤ 1.0). It
prints all three statistics on every run, pass or fail. The provisioning job also
asserts its own generation argv contains no `--seed-generate`.

### What the provisioning job verifies before it uploads

Each golden must carry `reference_self_consistent: true` **and**
`emitted_tokens[i] == rows[i].sequential_argmax` for every row — the
self-consistency flag alone did not check that until Amendment 10, and three
goldens shipped with the contradiction. It must also carry at least as many rows
as the gate will request (512), and it must pass the degeneracy screen. Any of
those failing aborts before anything is uploaded.

## Step C — the serial frequency floor (BLOCKING at the ranked window, not applied)

**Escalated 2026-07-31 from "intermittent" to BLOCKING.** Previously this was seen
only as the box warmed across pairs. Measured at the RANKED window (512 decode
tokens, K=3, the value the workflow passes), it rejects **deterministically on the
first pair and on the gated retry**:

```
pair1-serial.a1   loaded=91  steady_n=84  min=1589 MHz  max=1620 MHz  -> REJECT
pair1-serial.a2   loaded=96  steady_n=87  min=1592 MHz  max=1620 MHz  -> REJECT
                                                        FATAL(code=5) pair 1
```

The floor sits **inside the machine's normal loaded range** (1589-1620 MHz) once a
phase runs the ranked length: a 512-token serial phase takes ~52 s against ~13 s at
128 tokens, so it settles into a 2% lower steady clock. At 128 tokens the same
baseline passed at 1603 MHz — 3 MHz of margin, which is why this read as
intermittent. Sample counts are healthy (91-96 loaded), so this is purely the
frequency arm, not the sample-count arm that bit the serial production wrapper.

**Consequence: the calibration band cannot be re-authored at the ranked window
until this is resolved**, because authoring needs four accepted pairs. That makes
this blocking for go-live, and it blocks Amendment 27's fix from being completed.

Original measurement, as the box warmed at 128 tokens: 1613 -> 1604 -> **1598** MHz,
crossing the floor on the third pair; the gated retry returned 1605. Genuine
sustained throttle on this silicon is **1447-1455 MHz** (operator record
2026-07-12), so every one of these figures — including 1589 — is well over 130 MHz
clear of real throttle.

Both sides run at ~1606 MHz, but dflash is judged against 1500 (106 MHz margin) and
serial against 1600 (4-13 MHz). The 1600 value was inherited from the serial
track's workload; this track's denominator is `dflash-probe`, a width-1 forward
through the DFlash path with the drafter resident.

**Recommended:** `MIN_FREQ_SERIAL=1500`, matching the dflash side, same throttle
record, still ~45 MHz above measured throttle. One line, then
`/opt/bench/gen-manifest.sh` and `/opt/bench/janitor.sh --audit-only`, on every box
serving the track.

Not applied by an agent: the thermal/telemetry stability contract is declared
`readonly` with "do not env-override" and is operator-owned. That has not changed
with the escalation — a throttle floor is exactly the kind of gate an agent must
not relax on its own authority, even with the measurement in hand. Left at 1600 the
track does not merely carry an intermittent false-reject: **no ranked run can
complete pair 1 at the 512-token window**, and the calibration band cannot be
authored, so go-live is blocked on this one line.

## Step D — flip the two trusted-contract fields

The guard requires BOTH, on `main`:

- `fixtures/laguna_xs_2_1_dflash_track.json` -> `official_scoring_enabled: true`
- the same fixture's `reference_baseline.publication_allowed: true`

`benchmark.dflash.json` -> `scoring.tokenFidelityGateStatus` is `pending-spec`; the
pinned test `trackCannotBeEnabledWhileTheFidelityGateIsUnspecified` fails the build
if official scoring is enabled while that is not `implemented`. Decide deliberately
what that word means for this track — see "known limits" below — and set it in the
same change.

### Evidence for that decision, assembled 2026-07-31 (status NOT changed)

The declared gate is
`trusted-sequential-reverification-with-bounded-near-tie-budget`. Each of its
three clauses now has an implementation and a live measurement behind it. The
status was deliberately left at `pending-spec` — flipping it is the go-live
judgement and belongs to whoever throws the switch — but the evidence is
recorded here so that call does not have to be re-derived.

| clause | implementation | observed in run `30613617340` |
|---|---|---|
| **trusted** | verifier lives in `Sources/MLXFastTrustedHarness/LagunaRuntimeDFlash*.swift`; participant code cannot reach it (it links no MLX/model code) | gate ran inside the trusted parent, candidate confined to the bench sandbox |
| **sequential reverification** | `DFlashReferenceRow.sequentialArgmax` — "reference argmax in the K=1 sequential frame" — plus post-run replay | `reference_checked_row_total: 858`, `verify_block_replayed_round_count: 335` |
| **bounded near-tie budget** | `nearTieBudget` / `residualBudget`, bound `experimentalDFlashNearTieAdmissionBudgetPerThousand = 40`, enforced by the `residualBudgetExhausted` violation | `admissible_near_tie_count: 6`, `residual_divergence_count: 0`, i.e. the budget was exercised and not exhausted |

Supporting: `all_tokens_matched: true` with 512/512 rows admissible (503 exact +
6 near-tie + 3 declared-frame), `work_binding_comparison_count: 1716`,
`rejected_rows_reference_checked: 346`, `max_top2_logit_delta` 1.875 against the
4.875 tolerance.

**What this evidence does NOT settle.** Two questions remain judgement, not
measurement:

1. The near-tie budget of 40/1000 has only ever been exercised at 6 admissions
   per 512 rows — roughly a third of the bound. Nothing has probed what happens
   near the limit, so the bound's *value* is untested even though its
   *enforcement* is demonstrated.
2. Everything above was measured with the candidate being unmodified reference
   code. A ranked run's candidate is adversarially motivated; the gate has never
   faced one. Contract Amendments 18-21 record that this track's gates have
   repeatedly looked sound until first attacked.

Also flip `confirm_track_enabled`'s default to `true` in the workflow if Yukon will
dispatch with defaults; it is currently `false` precisely because the track is
inert.

## Step E — verification dispatch

```
gh workflow run dflash-benchmark.yml --repo Layr-Labs/mlxfast-challenge-dev \
  --ref main -f confirm_track_enabled=true -f run_benchmark=false
```

Correct behaviour BEFORE step D: fails at "Enforce DFlash track enablement" citing
`official_scoring_enabled=false, publication_allowed=false`, everything after
skipped. That is the fail-closed path, verified 2026-07-30.

AFTER step D, the same dispatch should proceed past the guard through the untimed
gates. Only then run with `run_benchmark: true` for a timed measurement.

## Rollback

Set either contract field back to `false`. The guard fails closed on the next
dispatch; nothing needs unwinding on the box. The runner can be parked with
`launchctl unload -w /Library/LaunchDaemons/com.bench.supervisor.plist`.

## Known limits to accept, or fix first

These are documented, measured, and unresolved. Enabling scoring accepts them.

1. **An unmodified candidate scores 0.5493 on the ranked golden**, so the entry bar
   is far steeper than the ~0.87/~16% figures below — those were measured on a
   DIFFERENT golden at ~69% draft acceptance and are **superseded by Amendment 26**.
   On the material this track actually ranks against, draft acceptance is ~34%, the
   block path declares **1.8125 rows of compute per emitted token** against a 1.8206
   observed slowdown, and a no-op measures 0.5493 at K=3 / 0.6201 at K=2. The floor
   is 0.52 accordingly. An entrant must therefore find roughly **1.8x of general
   forward speedup before they rank above a no-op**, which is a far less attractive
   track than the 19% figure implied — that is the decision being accepted here, and
   it is the strongest argument for per-prompt no-op normalisation before go-live.
   The historical reasoning: because the denominator is the pinned
   baseline and only the numerator is the candidate's build, general kernel wins DO
   count — but block decode itself costs ~16% on realistic prose, so entrants need
   roughly **19% of general forward speedup before they rank at all**. That is an
   entry bar, not a bug, and it decides whether the track is attractive to enter.
2. **L2's cross-build tail term is unmeasured.** Every rejected-tail number is
   same-build; cross-build was the larger term on the emitted rows (Amendment 21 §6).
3. **The L2 drift band cannot be closed by tightening** — it is occupied by honest
   cross-build drift, measured over 12,800 comparisons (Amendment 20).
4. **Drafter provenance is enforced by rule plus static review**, not at runtime;
   the exact draft-provenance detector is specified but unbuilt (Amendment 9).
5. **Two DFlash sources sit inside the SERIAL track's editable surface** via
   `benchmark.json`'s `Sources/MLXFastModel` directory entry, so serial submissions
   package them. Moving them changes a live competition's contract and was
   deliberately left alone.
