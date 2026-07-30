# DFlash go-live runbook (`laguna-xs-2.1-dflash-v1`)

`.github/workflows/dflash-benchmark.yml` points operators here from four places,
including "step B", and until now the document did not exist. This is it.

Everything below is the work that CANNOT be done by an agent: it needs hidden
material, a judgement about whether the anti-cheat is sound enough to rank on, or a
change to an operator-owned measurement contract.

## 0. What is already done and verified (do not redo)

Measured on M5-C, 2026-07-30. Contract detail in
`docs/dflash-track-correctness-contract.md` (Amendments 1-22).

| thing | state |
|---|---|
| harness end-to-end | `ACCEPT`, 4/4 pairs, `dflash_decode_speedup` **0.8705** (median 0.8712, min 0.8658), band check LIVE, sealed `results.json` |
| pinned baseline | `/opt/bench-runner/baseline/laguna-xs-2.1-dflash-v1/<sha>` + `current` symlink, 23 GB, weights APFS-cloned from the serial baseline |
| baseline calibration | `/opt/bench-runner/state/laguna-xs-2.1-dflash-v1/baseline-calibration.json`, authored by `/Users/gaj/author-dflash-calibration.sh` (the wrapper does NOT write it) |
| runner | `m5-laguna-dflash-3-*` online on **mlxfast-challenge-dev**, labels `[self-hosted, m5-laguna-dflash]` only |
| dispatch chain | verified: job scheduled -> host preflight -> trusted context -> enablement guard fails CLOSED on the contract |
| decode floor | 0.80 in all five sites (manifest, fixture, two workflow comments, workflow env, box wrapper) |
| janitor audit | clean after re-signing |

## Step A — hidden timed-prompt pool (BLOCKING)

`fixtures/laguna_xs_2_1_dflash_track.json` -> `timed_prompt_pool` is `[]` and the
workflow's "Select hidden DFlash timed target from the pool" step fails closed on an
empty pool. No ranked run can start until this is populated.

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

## Step B — freeze and pin the IT-target goldens (BLOCKING)

The workflow refuses to run with an empty hidden-golden pin:

> `hidden DFlash golden pin <name> is empty; freeze the IT-target goldens (go-live
> runbook step B) and pin them here before enabling the track`

Generate each golden with the pinned baseline binary, then pin
`MLXFAST_DFLASH_CORRECTNESS_GOLDEN_SHA256` / `_BYTES` and
`MLXFAST_DFLASH_BENCH_GOLDEN_SHA256` / `_BYTES` from the frozen objects.

`dflash-reference` builds them; `--seed-generate N` extends a seed and `--generate N`
produces the emitted chain. Verify each carries `reference_self_consistent: true`
AND that `emitted_tokens[i] == rows[i].sequential_argmax` for every row — the
self-consistency flag alone did not check that until Amendment 10, and three
goldens shipped with the contradiction.

## Step C — the serial frequency floor (RECOMMENDED, not applied)

`MIN_FREQ_SERIAL=1600` in `/opt/bench-runner/measure-dflash-job.sh` false-rejects
the honest serial denominator. Measured across one run as the box warmed:
1613 -> 1604 -> **1598** MHz, crossing the floor on the third pair; the gated retry
returned 1605. Genuine sustained throttle on this silicon is **1447-1455 MHz**
(operator record 2026-07-12), so 1598 was 145 MHz clear of throttle.

Both sides run at ~1606 MHz, but dflash is judged against 1500 (106 MHz margin) and
serial against 1600 (4-13 MHz). The 1600 value was inherited from the serial
track's workload; this track's denominator is `dflash-probe`, a width-1 forward
through the DFlash path with the drafter resident.

**Recommended:** `MIN_FREQ_SERIAL=1500`, matching the dflash side, same throttle
record, still ~45 MHz above measured throttle. One line, then
`/opt/bench/gen-manifest.sh` and `/opt/bench/janitor.sh --audit-only`, on every box
serving the track.

Not applied by an agent: the thermal/telemetry stability contract is declared
`readonly` with "do not env-override" and is operator-owned. Left at 1600 the track
carries an intermittent false-reject that consumes a run's single gated retry.

## Step D — flip the two trusted-contract fields

The guard requires BOTH, on `main`:

- `fixtures/laguna_xs_2_1_dflash_track.json` -> `official_scoring_enabled: true`
- the same fixture's `reference_baseline.publication_allowed: true`

`benchmark.dflash.json` -> `scoring.tokenFidelityGateStatus` is `pending-spec`; the
pinned test `trackCannotBeEnabledWhileTheFidelityGateIsUnspecified` fails the build
if official scoring is enabled while that is not `implemented`. Decide deliberately
what that word means for this track — see "known limits" below — and set it in the
same change.

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

1. **An unmodified candidate scores ~0.87.** Because the denominator is the pinned
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
