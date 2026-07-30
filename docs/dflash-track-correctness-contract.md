# DFlash ranked track — correctness contract (Criterion E)

**Status: DESIGN, not yet implemented.** The DFlash track is fail-closed
(`fixtures/laguna_xs_2_1_dflash_track.json` -> `official_scoring_enabled: false`)
and MUST stay so until every layer below is implemented and validated on M5-C.
`benchmark.dflash.json` carries `tokenFidelityGateStatus: "pending-spec"`; that
key flips to `"implemented"` only when L1-L6 are in place.

## Why the retired MTP contract cannot be reused

The retired MTP track required every emitted token to match a trusted
**sequential** greedy golden exactly. That is unsatisfiable for DFlash on this
model, and it is measured, not assumed.

Measurement (M5-C, 2026-07-30): 6 diverse prompts x K in {4,8,16} x 128 tokens
via `mlx-bench dflash --parity-check --parity-top-k 5`
(log: `~gaj/dflash-parity-stats.log`):

| observation | value |
|---|---|
| divergence events | 14 |
| of kind `baseline_path_block_vs_sequential` (target-only, **no drafter**) | 14 / 14 |
| DFlash-vs-baseline mismatches | **0** |
| max sequential-logit gap of the chosen token | **0.625** (mean 0.286, median 0.375; two exact 0.000 ties) |
| sequential RANK of chosen token | rank2 x12, rank3 x1, rank5 x1 |
| upper-bound divergence rate | <= 14/2304 = 0.61% of emitted positions |

Conclusions that drive the design:

1. **DFlash itself is exact** w.r.t. the target's block forward. All divergence
   is the target's own block-vs-sequential NVFP4 accumulation order, present
   with no drafter in the loop. An exact-vs-sequential gate fails honest code
   14/14.
2. **Rank is the wrong metric.** Top-2 membership fails 2/14 honest events; the
   rank-5 case sat in a flat distribution (21.50/21.25/21.125/21.00/21.00) only
   0.5 logits from top-1. Use logit/frame reasoning, never rank.
3. This divergence is **frame divergence** (the same honest computation read out
   at a different matmul width), not error -- which is what Criterion E exploits.

## Adversarial review (workflow wf_4dc2d13a-519, 4 independent red teams)

All four naive criteria were judged **non-viable as written**:

- **A) exact vs sequential** -- worst: simultaneously unsatisfiable by honest code
  AND blind to every threat (output-only predicate). Also creates a hidden-golden
  exfiltration channel if the mismatch error reaches a participant-visible log.
- **B) exact vs block-path golden** -- fixed golden freezes the accepted-length
  schedule (the very thing being optimized); candidate-derived golden is circular.
- **C) trusted sequential re-verification + epsilon budget** -- best of the four but
  economically defective: it puts honest frame divergence and cheating in the SAME
  capped bucket, so the budget must be large enough to survive honest flips and a
  cheater can spend that margin. Also had an architectural blocker (below).
- **D) top-k membership** -- as a primary criterion it IS the exploit: it relaxes
  acceptance from "equals target argmax" to "in top-k". Auxiliary floor only.

### Architectural blocker that killed the naive phrasing

`MLXFastCLI` (the trusted binary) links **no MLX** -- deps are MLXFastCore /
MLXFastTransform / MLXFastHarness / Tokenizers. It cannot compute a logit. So
"the trusted parent re-runs the target" is not implementable as stated; the
reference must be a **second worker process built from the pinned baseline tree**,
running the **organizer's** weights, strictly after the timed window.

Related trap found in review: `Sources/MLXFastTransform` is inside
`editablePaths` and the ranked workflow pins `weights.sha256` to the CANDIDATE's
own transform output. A reference verifier that read the candidate's `weights/`
would compute its "honest argmax" from candidate-perturbed numbers
(transform poisoning). The reference MUST use organizer-transformed weights.

---

## Recommended contract

CRITERION E — "Reference-frame admissibility with work-bound rounds." A layered contract, not a single test. PRIMARY: every emitted token must be the argmax of the ORGANIZER-PINNED reference target in one of two exactly-defined frames the reference itself computes (K=1 sequential, or the block width the candidate declared for that row), teacher-forced on the candidate's own emitted prefix and replayed round-by-round; a small, capped residual bucket absorbs candidate-vs-reference kernel divergence and must additionally satisfy reference top-2 membership (criterion D folded in as a floor, not a primary). AUXILIARY, and where the real anti-cheat work happens because no token-level test can ever price a verifier that is cheap at easy steps: (L2) per-row pre-lm_head hidden-state digests plus per-row top-2 logit VALUES, reference-checked for EVERY declared row including rejected ones — this binds emitted tokens to executed target compute at all positions rather than only at plateaus; (L3) parent-owned row accounting over reference-proven rows, which closes over-emit arithmetically; (L4) parent-demanded KV digests at the committed offset with vacancy sentinels beyond it, plus a mandatory sliding-window wrap-seam leg, which catches rollback elision STRUCTURALLY even when it never flips a token; (L5) reference-drafter replay, which closes the cheap-drafter and schedule-replay holes without fixing the acceptance pattern; (L6) a hidden prompt POOL sampled per run, a parent-randomized per-round K schedule, an undisclosed total decode length, redacted failure reports, and ranked-failure rate limiting, which jointly kill the resubmission lottery, the 128-query golden oracle, threshold tuning, and the in-band phase oracle. The candidate worker is NEVER asked to re-verify anything: it emits, the parent journals, and the pinned reference worker replays afterwards — so there is no distinguishable verify opcode to detect and no self-grading path.

---

## Implementable specification

```
=============================================================
CRITERION E — DFlash ranked correctness contract, implementable spec
=============================================================

0. THE ONE IDEA THAT MAKES THIS WORK
-----------------------------------
The measured 4787/5261 divergence is not ERROR, it is FRAME divergence: the
same honest computation read out at a different matmul width. Criterion C's
fatal economic defect is that it dumps frame divergence and cheating into the
SAME capped/epsilon bucket, so the budget B must be sized large enough to
survive honest frame divergence — and every unit of that margin is spendable
by a cheater.

E splits the buckets. Systematic honest divergence is absorbed EXACTLY and
WITHOUT LIMIT by admitting the reference's argmax in the block frame the
candidate declared (no epsilon, no budget, admissible set of size <= 2
specific token ids). Only the residual candidate-vs-reference KERNEL
divergence goes into a capped bucket, and because that residual is small,
the cap can be small — and a small cap is not monetizable.

Second idea: token output provably cannot see a verifier that is degraded at
every step but agrees at the 85-95% of confident steps. So E does not try.
E binds work with a ~2048-dimensional readout (hidden-state digest) plus
per-row top-2 logit VALUES, checked at EVERY row. A 30-of-40-layer exit or
top-6-of-8 expert routing perturbs those at every position, including the
confident ones where argmax hides it.

1. WHO COMPUTES THE REFERENCE (closes T1 self-grading, T2 self-oracle)
---------------------------------------------------------------------
Established fact in this repo: Package.swift wires MLXFastCLI ->
MLXFastHarness (path Sources/MLXFastTrustedHarness, -DMLXFAST_TRUSTED_HARNESS)
with only MLXFastCore / MLXFastTransform / Tokenizers. The trusted binary
links no MLX and CANNOT compute a logit. Therefore "the trusted parent
re-runs the target" must mean a SECOND PROCESS:

  REFERENCE VERIFIER := mlxfast-runtime-worker built from
  ${MLXFAST_BASELINE_WS} (/opt/bench-runner/baseline/<track>/current) at the
  pinned MLXFAST_BASELINE_COMMIT, already asserted 40-char-exact by
  .github/workflows/benchmark.yml:303-312.

Hard requirements on it:
  R1. Built from the pinned baseline tree only. Byte-pinned like the trusted
      harness (reuse .github/scripts/pin-trusted-harness.sh with a third
      role, "reference-worker"), re-verified immediately before replay.
  R2. It loads the ORGANIZER's weights/ tree, produced by the BASELINE's
      Sources/MLXFastTransform from the pinned reference checkpoint
      (referenceModelRevision 841778bda...), hash-pinned. It MUST NOT read
      the candidate's weights/. This is the only closure for the
      transform-poisoning exploit: Sources/MLXFastTransform is inside
      editablePaths and the workflow currently pins weights.sha256 to the
      CANDIDATE's own transform output, so a reference verifier reading
      candidate weights/ would compute its "honest argmax" from the
      candidate's perturbed numbers.
  R3. It runs the same organizer-pinned DFlash draft weights.
  R4. It runs STRICTLY AFTER the timed measurement, in its own process, with
      the candidate worker torn down. Two 21.6 GB residencies are affordable
      on 128 GB but must not overlap the timed window (thermal + allocator).
  R5. Reference self-consistency is asserted every run: replay one randomly
      chosen round twice and require BIT-IDENTICAL frames. The retired track
      measured reference-vs-reference instability (a step-48 argmax flip at
      identical logical offsets), so A1/A2 are not well-defined until this
      passes. Fail the run "reference nondeterministic" — an operator fault,
      not a participant fault, and it must be reported as such.

2. PROTOCOL AND WHAT THE WORKER MUST REPORT
-------------------------------------------
Only ONE request kind reaches the candidate worker on every leg:
decode_block. No verify/re-verify/attest opcode exists. This is deliberate:
LagunaRuntimeWorker.swift's env allowlist already makes the environment
byte-identical across legs to kill phase oracles, and a distinguishable
re-verify opcode would reintroduce one in band, below that defense.

Parent -> worker, per round r (extends ExperimentalDecodeBlockRequest at
LagunaRuntimeWorker.swift:968):
  { kind:"decode_block", id, token: <previous committed token>,
    max_block_size: K_r }
K_r is chosen by the PARENT from a randomized schedule (see 6c). Nothing else
is ever sent. In particular: never the total decode length, never the
remaining count, never any expected token. The current MTP validator's
`requestedMaxBlockSize <= remainingTokenCount` clamp
(LagunaRuntimeMTP.swift:327-332) MUST be removed — it leaks the remaining
budget near the end of the run. The parent instead always requests a full
K_r and truncates the SCORED prefix itself; every emitted row is still
re-verified.

Worker -> parent, per round (all fields MANDATORY; a missing or malformed
field is a hard protocol failure, never a warning):

  a. tokens: [Int]            emitted/committed tokens this round, length
                              a_r + 1 (accepted prefix plus the target
                              token), 1 <= len <= K_r.
  b. declared_rows: Int       number of target rows this round's verify
                              forward executed. Must satisfy
                              declared_rows >= len(tokens) and
                              declared_rows <= K_r.
  c. declared_block_width: Int
                              the matmul row-width the verify forward was
                              actually dispatched at, INCLUDING any padding
                              to a friendlier tile. This is the frame the
                              reference replays for A2. Must be
                              >= declared_rows.
  d. draft_tokens: [Int]      the drafter's proposals for this round, length
                              declared_rows - 1 (or declared_rows on a
                              bonus-free policy; fix one convention).
  e. row_hidden_digest: [Digest]
                              length declared_rows. Digest of the target's
                              FINAL-NORM, PRE-lm_head hidden state for that
                              row. See 3c for the digest function.
  f. row_top2: [{token,logit},{token,logit}]  x declared_rows
                              the row's top-2 target logits BY VALUE. Sorted
                              descending, deduped, finite, in vocab range
                              (reuse validatedWorkerTopLogits' internal
                              consistency checks at
                              LagunaRuntimeCorrectnessCompare.swift:429 —
                              but note those checks are necessary, never
                              sufficient; truth comes only from 3).
  g. accepted_count: Int      diagnostic only; the parent recomputes
                              acceptance from (d) and the reference frames
                              and IGNORES this field for any gating. It is
                              retained solely so a forged value is itself a
                              detectable inconsistency.
  h. kv_digest: Digest?       present only when the parent set a
                              kv_checkpoint flag on the request (see 5).

The parent writes all of the above to a SEALED TRUSTED-SIDE JOURNAL as it
arrives. The journal is parent-written, never worker-written, never inside
${MLXFAST_JOB_WS}. Score publication is gated on replay of that journal.

3. PRIMARY CRITERION — per-position frame admissibility
------------------------------------------------------
Replay is TEACHER-FORCED on the candidate's own emitted prefix. This is
mandatory and is the single most likely implementation mistake: generating a
free-running reference continuation and comparing positionally turns one
honest divergence at step 3 into ~125 failures and silently reduces E to
criterion A.

For each round r, the parent asks the reference worker for TWO frames over
the same input rows the candidate declared:
  F_seq(r): the reference target's argmax and top-2-by-value for each row,
            computed at K=1 (one row per forward, sequential accumulation
            order).
  F_blk(r): the same, computed in ONE forward at width
            declared_block_width — the frame the candidate declared.

Emitted token t at row j is ADMISSIBLE iff any of:
  A1  t == argmax(F_seq(r)[j])                      uncapped, free
  A2  t == argmax(F_blk(r)[j])                      uncapped, free
  A3  t in top2(F_seq(r)[j]) AND
      logit_seq[argmax] - logit_seq[t]
        <= EPS_REL * max(1, |logit_seq[argmax]|)     CAPPED, budgeted

A1/A2 carry no budget because they are the reference's own exact answers in
two exactly-specified frames — the admissible set is at most 2 specific
token ids, usually 1. Contrast criterion C's epsilon at a plateau, where a
0.25-wide window on a bf16 grid of spacing 0.125 admits 3-10 tokens: E does
not grant that at all.

A3 exists only for candidate-vs-reference KERNEL divergence (the honest
consequence of the new steel tile / fused NVFP4 dequant / _nax variant the
participant is being paid to write). Two properties matter:
  - EPS_REL is RELATIVE, not absolute. bf16 has 8 mantissa bits, so ULP at
    |logit| ~ 30 is 0.125 (~0.4% relative). An absolute epsilon false-fails
    an honest divergence whose sequential gap is 0.06 while the frame
    perturbation was 0.125. Note the cited 4787/5261 case is an EXACT tie in
    the sequential frame (both 30.7500) and so survives even 1e-6 — that is
    the easy case and must not be used to argue a tight absolute epsilon is
    safe.
  - The top-2 membership conjunct is criterion D folded in as a FLOOR, not a
    primary. It bounds how far a tolerated divergence can wander in token
    space, which epsilon alone cannot do on a flat plateau.
  - top2 is queried as an explicit per-token logit lookup, NOT a top-8 slice.
    MLXFastConstants.correctnessTopLogits = 8 would fail-closed on an honest
    divergence whose token sits at reference rank 9+, which is reachable when
    the bf16 grid packs many tokens onto two grid points.

Budgets: A3 count <= B_A3_TIMED over the 128-step timed leg, and A3 RATE <=
R_A3_MAX over each long fidelity leg. Any position that is not A1, A2, or a
within-budget A3 fails the run.

STEERING-EVENT CONTAINMENT. Because replay is teacher-forced, an excused A3
at step t re-bases every subsequent reference argmax, so B A3 events are B
arbitrary redirections of the continuation, not B token-sized errors. Three
additional constraints bound this and MUST be stated in the contract:
  - MIN_A3_SPACING: no two A3 events within N steps.
  - A3 events may not exceed B_A3_WINDOW within any sliding window of W steps.
  - the A3 token must be reference top-2 (already in A3), so a redirection
    cannot be to an arbitrary plateau member.

4. WORK BINDING — the layer that actually defeats T1 and T2
----------------------------------------------------------
This is the part that no purely token-level criterion can supply, and it is
the core of E.

4a. THE DEFECT BEING CLOSED. Argmax is a near-1-bit readout. A verifier
truncated to 30 of 40 layers, or routing top-6 of 8 experts, or dropping the
shared expert on verify rows, or accumulating NVFP4 coarsely, agrees with the
honest argmax at the 85-95% of confident steps and disagrees only at
plateaus — which any tolerance-bearing contract excuses. So the cheater buys
a permanent per-step FLOP discount for a bounded number of charge events.
Likewise lm_head elision (run the 40-layer trunk honestly so KV rows are
genuine, then project only the bonus row and take rows 0..K-2 from the
drafter unverified) deletes up to 15/16 of the verify's vocab work while
every KV invariant and every emitted token stays correct.

4b. THE CLOSURE. For EVERY declared row — accepted, rejected, and bonus —
the parent requires row_hidden_digest and row_top2, and checks them against
the reference:
  - row_hidden_digest[j] must agree with the reference's digest of the same
    row's final-norm pre-lm_head hidden state, within DIGEST_TOL, in
    whichever of F_seq/F_blk the position was admitted under (and the parent
    accepts agreement with EITHER frame's digest, mirroring A1/A2).
  - row_top2[j].logit values must agree with the reference's corresponding
    logit values within LOGIT_VAL_TOL (relative), and row_top2[j].token
    identities must match the reference's top-2 as a SET unless the position
    is a legitimate near-tie by the same EPS_REL test.

Why this binds work rather than output:
  - The hidden digest cannot be produced without executing the row's full
    40-layer trunk at that row. Layer truncation, expert-count reduction,
    and coarse accumulation all move a 2048-dim state far more than kernel
    reassociation does, at EVERY position — so T2 becomes detectable at
    confident steps, not just at plateaus. That is the economic inversion:
    the cheater now pays at all 128 steps instead of at 3.
  - row_top2 by VALUE cannot be produced without the row's 100352-wide
    vocab projection. This is what forces the per-row lm_head that
    4a-elision deletes, and it is why the values (not just the identities)
    are checked.
  - Requiring both for REJECTED rows too is essential: the drafter-confidence
    gate ("if margin > tau, return drafts, dispatch zero target rows") and
    the MLX-laziness variant (build the verify graph and never eval() it) can
    only satisfy this by actually running the rows.
  - A participant who computes honest digests alongside a cheap decode path
    has done strictly MORE work, so there is no speed win — the exploit
    closes economically, not just detectably.

4c. DIGEST FUNCTION. Must be (i) cheap enough to sit on the timed path,
(ii) coarse enough that honest kernel reassociation agrees, (iii) fine enough
that a 30-layer exit disagrees. Recommended shape: project the row's
final-norm hidden state onto P fixed pseudorandom sign vectors seeded from
the pinned prompt-target id (so it is not a knob the participant chooses),
quantize each projection to a coarse grid, and report the P values. P
reductions over 2048 dims is negligible against a 100352-wide lm_head. Do
NOT use a cryptographic hash: it has no tolerance and would false-fail every
honest kernel edit. Do NOT let the worker choose the projection.

4d. COST SYMMETRY. Digest+top2 reporting sits on the timed path for BOTH the
candidate and the pinned DFlash baseline, so the paired ratio stays fair.
Calibrate DIGEST_OVERHEAD_MAX and require the measured reporting cost to be
below it, else the contract taxes the thing it is measuring.

4e. ROW ACCOUNTING (closes T3 arithmetically). The parent owns the token
budget and the denominator (as today: LagunaRuntimeMTP.swift:778 divides the
parent's own wall time by its own totalTokenCount). It additionally enforces
per round len(tokens) <= declared_rows <= K_r, and globally
sum(len(tokens)) == totalTokenCount with every declared row
reference-checked. Because declared rows are PROVEN by 4b rather than
asserted, over-emission requires proving the row — which is the honest work.
CRITICAL: this replaces the forgeable pattern the retired track used. The
row equation at LagunaRuntimeMTP.swift:758-766
(2*pairs - rollbacks + serialRows == totalTokenCount) is satisfiable by
arithmetic alone from worker-reported integers; a worker that verified
nothing passes it. NO worker-reported counter may carry audit weight unless
the parent independently checks the underlying computation. That is exactly
what 4b does and what the old counters did not.

5. KV STRUCTURAL GATE (closes T4 without waiting for a token to flip)
--------------------------------------------------------------------
Rollback elision is invisible to every output criterion whose window never
reaches the corruption. The timed window is a 512-token seed plus 128 steps
against a 512-position sliding window, so the wrap seam may never be crossed
— a submission can leave stale rows, pass, and be broken in production.

5a. KV DIGEST CHECKPOINTS. On parent-chosen rounds (flagged on the request;
the flag is the ONLY request-side variation and it is present on both legs so
it is not a leg oracle), the worker must return kv_digest for a
PARENT-CHOSEN (layer set, position window) — chosen after the fact is
impossible, so choose it pseudorandomly per run and per round from a seed the
worker never sees before the request. The window MUST include positions
beyond the committed offset, which must be reported as a canonical VACANT
sentinel. The parent compares against the reference worker's digest after
teacher-forcing the same committed prefix. Stale speculative rows left in
place fail here structurally, whether or not they ever flip a token.

5b. COVERAGE. The digest must span the target caches AND the drafter caches
AND the shared-KV view. The "roll back the target but not the drafter /
shared-KV snapshot" elision degrades acceptance while keeping tokens exact,
so it passes every token contract and must be caught here.

5c. WRAP-SEAM LEG. Mandatory untimed leg long enough to cross the
512-position sliding window at least WRAP_CROSSINGS times (the existing
experimentalMTPMaxConfiguredTotalTokens = 1536 is 3x the window and is the
right order). KV checkpoints must be scheduled ON the seam. This is also
where the retired track's own documented eligibility loss at the wrap gets
exercised rather than assumed.

6. DEFEATING THE META-EXPLOITS (the lottery, the oracle, the phase probe)
------------------------------------------------------------------------
6a. PROMPT POOL, NOT A FROZEN PROMPT. Replace the single pinned timed prompt
with a pool of N_PROMPTS hidden prompts, all individually
baseline-calibrated, ONE SAMPLED PER RUN, with the pinned DFlash baseline
measured on the SAME sampled prompt in the same session (the paired ratio
still cancels host drift). This is what kills threshold tuning, step-indexed
replay tables, and memorized continuations. Note the pool ALSO removes the
premise that makes fixed-golden criterion B even definable — which is fine,
because E does not use a fixed golden.

6b. UNDISCLOSED LENGTH. The parent issues rounds one at a time and never
reveals the total. The worker therefore cannot cheat "only inside the timed
128" because at round 5 it cannot know whether this run is 128 or 1536 steps.
Combined with 6c this is the structural closure for the in-band phase oracle;
it requires the remaining-count clamp removal noted in 2.

6c. RANDOMIZED K SCHEDULE. K_r is drawn per round by the parent from a
calibrated distribution over 1..16, including K_r = 1. This (i) prevents any
fixed acceptance pattern from being assumed — satisfying requirement (iv) —
(ii) forces the candidate's kernels to be correct at every width rather than
hand-fitted to the widths that appear in one reference segmentation, and
(iii) makes "always pad to the fastest tile" a declared, replayed choice
rather than a hidden one (declared_block_width in 2c).

6d. REDACTION. Failure reports are (step, admissible=false, category) only.
NEVER expected_token, NEVER reference logits, NEVER reference top-2, NEVER
per-round reference frames. The retired validator's own error object carries
the golden (LagunaRuntimeMTP.swift:359-367 populates
BenchmarkTokenMismatchError with expectedToken) and Sources/MLXFastCLI/main.swift:69
prints it to stderr; on the serial pipeline that is safe only because the
step uses the PUBLIC fixture. Against hidden material it is a 128-query
golden oracle. Extend the existing assertions
(.github/workflows/benchmark.yml:1166-1168 and 1542-1545, which already
require first_failing_step == null and expected_token == null) to cover every
new DFlash field: declared frames, A3 positions, digest deltas, KV digests.
Every new diagnostic field is a fresh redaction surface and must be added to
those jq assertions in the same commit that adds the field.

6e. RATE LIMIT. Cap ranked failures per participant per window. Without it,
"failed runs are free and dispatches queue rather than cancel" turns any
probabilistic contract into submit-until-green.

7. LEG STRUCTURE AND ORDERING
-----------------------------
  L-public   public drift tripwire, unchanged, before any hidden material.
  L-fid      LONG untimed fidelity leg(s): sampled prompt(s), 512-token seed,
             FID_STEPS steps (>= wrap requirement), randomized K, KV
             checkpoints on the seam. Full journal.
  L-gates    hidden teacher-forced base case, anchor, free-run, GPQA TTFT,
             semantic GPQA judge — UNCHANGED and still the fidelity
             authority. E does not replace them; a block-shaped contract
             permanently forfeits "matches the model's sequential greedy
             output" as a definition of correct, so the semantic anchor is
             load-bearing, not decorative.
  L-timed    frozen 512+128 window, sampled prompt, journal only, no
             reference process running.
  L-replay   NEW, after timing, after teardown: pin-verify the reference
             worker, replay the journals of BOTH L-fid and L-timed against
             the reference. Score publication is GATED on this step.

REPLAY BUDGET. Full replay is ~1x the candidate's decode work plus the
reference drafter, untimed. Affordable at 128 steps; not at 1536 x N legs.
So: replay ALL rows of L-timed, and on L-fid replay all A3-claimed positions,
all KV checkpoints, plus a post-hoc random AUDIT_FRACTION of rounds chosen
AFTER the run from a seed the worker never saw. Cheating at per-token rate p
over S steps with audit fraction f is detected with probability
1 - (1-f)^(S*p); state the target detection power explicitly when calibrating
f, because f is the knob that converts job minutes into anti-cheat power.

8. DRAFTER FIDELITY (closes the cheap-drafter hole; does NOT freeze the pattern)
-------------------------------------------------------------------------------
The drafter weights are organizer-provisioned but participant-RUN, so
artifact hashing catches substitution on disk and catches nothing about a
runtime that loads the pinned bytes and executes 2 of 5 layers, or skips the
aux hidden-state fusion. Closure: the parent replays the REFERENCE drafter,
teacher-forced on the same emitted prefix, asked for the same number of
proposals the candidate declared, and requires draft_tokens to be admissible
under the same A1/A2/A3-style frame set with its own small budget
B_A3_DRAFT / EPS_REL_DRAFT.

This does NOT violate requirement (iv). The drafter is fixed BY THE TRACK
DEFINITION, so pinning its outputs removes no intended optimization axis —
participants optimize how fast they compute those proposals, not what they
are. K-policy, adaptive K, splitting, padding, kernel choice, cache layout,
and rollback implementation all stay free, and the ACCEPTANCE PATTERN stays
free because admissibility flows through A2 at the candidate's own declared
width. Nothing in E requires or assumes any particular acceptance pattern.

9. SCORING NOTES
----------------
  - The paired baseline MUST be the pinned DFlash reference implementation,
    not the serial K=1 decode. Otherwise the ratio mixes "we changed tracks"
    with "we optimized," and the floors become meaningless.
  - The old "same output, less time" defense of the paired ratio is gone by
    construction. Replace it in the published contract with "both outputs are
    reference-admissible under E, and the semantic gates bound answer
    quality." Say this explicitly; it is the honest framing and it is
    defensible.
  - Keep both component floors hard. Note the standing finding that the
    acceptance band is unenforced on the ranked path (it sees only
    gates-pass placeholders), so the floors are the only ranked timing gate —
    do not assume the band adds protection here.

10. WHAT MUST CHANGE IN EXISTING CODE
-------------------------------------
  - Sources/MLXFastCore/Constants.swift: experimentalMTPMaxBlockSize 4 -> 16
    (DFlash block_size); add every knob in the calibration list.
  - Sources/MLXFastTrustedHarness/LagunaRuntimeMTP.swift:318-371: replace
    ExperimentalMTPBlockValidator.accept's expectedTokens comparison
    wholesale. Delete the remainingTokenCount clamp (327-332). Delete the
    forgeable row equation (758-766) and the worker-reported
    exactPair*/serialVerificationRowCount fields as GATING inputs; retain
    only as diagnostics.
  - LagunaRuntimeWorker.swift:968 ExperimentalDecodeBlockRequest and
    validateExperimentalDecodeBlockRequest: add the kv_checkpoint flag;
    forbid everything else as today.
  - New trusted module: reference-worker spawn + frame query + journal
    replay. It drives the reference worker over the SAME JSON protocol, so it
    needs no MLX and respects the trusted-binary boundary.
  - .github/scripts/pin-trusted-harness.sh: third role "reference-worker."
  - benchmark.yml: reference weights/ provisioning from the baseline
    transform; L-replay step gating score publication; extended redaction
    assertions; prompt-pool sampling; rate limit.

```

## Numeric knobs requiring empirical calibration

- EPS_REL — relative near-tie tolerance for A3. Measure: run the candidate-shaped honest path and the pinned reference across many positions on held-out prompts; histogram (ref_seq_logit[argmax] - ref_seq_logit[emitted]) / max(1,|argmax logit|) for every position NOT admitted by A1 or A2. Set EPS_REL just above the observed upper tail (target ~1-2 bf16 ULP, i.e. ~0.4-0.8% relative at |logit| ~ 30). Must be expressed relatively, never absolutely.
- B_A3_TIMED — A3 budget over the 128-step timed leg. Measure: A3 count per 128 steps across >= 30 honest runs spanning several deliberately different honest kernel variants (new steel tile, fused NVFP4 dequant, _nax vs plain), since A3 volume scales with how much honest kernel reassociation was done. Set to observed_max plus a margin, but note the margin is directly spendable by a cheater — the design goal is for A2 to absorb enough that observed_max is single-digit. If it is not single-digit, the A2 frame definition is wrong (most likely declared_block_width is not capturing the actual dispatch width) and must be fixed rather than compensated by a bigger budget.
- R_A3_MAX — A3 rate ceiling on the long fidelity leg (A3 events per token). Measure: same honest runs at FID_STEPS length. This is the statistical detector for confidence-gated skip-the-verify strategies, which have a nonzero per-token error rate; state the minimum detectable p at the chosen FID_STEPS and AUDIT_FRACTION.
- MIN_A3_SPACING, B_A3_WINDOW, W — steering-event containment. Measure: honest A3 inter-arrival distribution; set spacing below the observed honest minimum gap and the windowed cap above the observed honest window max. These bound how often a tolerated divergence can re-base the trajectory.
- P (digest projection count), digest quantization grid, DIGEST_TOL — the central calibration and the highest-risk item. Measure TWO distributions: (i) HONEST digest deltas between the candidate and reference across the honest kernel variants above; (ii) ADVERSARIAL digest deltas for each concrete degradation the threat model names — 30-of-40-layer early exit, 35-of-40, top-6-of-8 expert routing, shared-expert drop on verify rows, coarser NVFP4 group/accumulation, sliding-attention truncation. Report the SEPARATION RATIO (adversarial min / honest max) per degradation. If any named degradation is not separated by a comfortable factor, increase P or refine the grid before shipping. A contract whose digest tolerance does not separate a 30-layer exit from an honest kernel edit provides no T2 coverage at all.
- LOGIT_VAL_TOL — relative tolerance on row_top2 logit VALUES against the reference. Measure: honest per-row logit value deltas across kernel variants. Must be tight enough that lm_head elision cannot fabricate plausible values without doing the projection, loose enough to survive honest reassociation.
- KV digest projection/grid and KV_DIGEST_TOL; KV_CHECKPOINT_COUNT, layers sampled per checkpoint, positions per window — measure honest candidate-vs-reference KV digest deltas, and separately measure the delta produced by deliberate rollback elision (logical-offset-only rewind) so the tolerance provably separates them. Also measure the per-checkpoint reporting cost.
- VACANT sentinel definition and the required number of beyond-offset positions per KV window — needs a decision plus a measurement that an honest implementation can always report vacancy cheaply (i.e. it does not force an otherwise-unneeded cache scan on the timed path).
- EPS_REL_DRAFT and B_A3_DRAFT — drafter-proposal admissibility tolerance and budget. Measure honest candidate-vs-reference drafter proposal divergence rates, and separately measure the divergence produced by a deliberately truncated drafter (2 of 5 layers, aux-fusion skipped) to confirm separation.
- DIGEST_OVERHEAD_MAX — the fraction of per-round wall time consumed by digest + top2 reporting. Measure on the M5 for both candidate and baseline. Target well under 1% of round time; if it exceeds that, the contract is taxing the measurement and P must shrink.
- AUDIT_FRACTION f and FID_STEPS — replay coverage on the long leg. Calibrate jointly against a stated target detection power: detection probability is 1-(1-f)^(FID_STEPS*p) for a cheater at per-token rate p; pick the smallest (f, FID_STEPS) meeting the target within the ranked job's minute budget. Also measure the wall-clock cost of L-replay end to end, since it roughly doubles the model-bearing work in the job.
- WRAP_CROSSINGS and the resulting minimum FID_STEPS — must exceed the 512-position sliding window enough times to exercise the seam under realistic rejection patterns; 1536 (3x window, the existing experimentalMTPMaxConfiguredTotalTokens) is the starting point but confirm empirically how many crossings are needed before a deliberate rollback elision is reliably caught by 5a.
- K_r schedule distribution over 1..16 — the parent's randomized per-round max_block_size. Calibrate so it (a) spans every width the candidate's kernels can dispatch, (b) includes K_r=1, (c) does not itself destroy achievable speedup relative to the reference implementation's own K policy. Both candidate and baseline must receive the SAME schedule per run.
- N_PROMPTS — hidden timed prompt pool size, plus per-prompt baseline calibration bands. Measure per-prompt baseline seconds/token variance on the M5 so the paired floors mean the same thing whichever prompt is sampled. Also re-run the existing n-gram susceptibility analyzer (benchmarkMaxPromptLookupHitRate = 0.03) on every pool member.
- Reference reproducibility: number of repeat replays and the required identical fraction for the R5 self-consistency assertion. Measure by replaying identical rounds N times on the box; the retired track observed reference-vs-reference instability, so this must be established, not assumed, before A1/A2 are well-defined.
- RANKED_FAILURE_LIMIT per participant per window — set from the observed honest false-failure rate (which must be driven near zero by the above) plus the number of queue slots the single M5 can absorb.
- semanticGPQAMinPassCount and correctnessGPQAMaxNewTokens — MUST be recalibrated on Poolside NVFP4 M5 runs. Constants.swift flags both as TEMPORARY from the superseded mlx-community affine checkpoint, and min_pass currently sits AT the observed floor (observations 2/5, 2/5, 1/5, 2/5 -> threshold 1), so the semantic backstop is simultaneously too weak to see plateau-concentrated quality loss and one judge coin-flip from false-failing. E leans on this gate as the fidelity authority, so it cannot stay at its floor.
- declared_block_width semantics — a DECISION requiring measurement: does A2 replay the logical row count or the padded dispatch width? Measure whether honest padded dispatches produce frames distinguishable from unpadded ones on the M5; if they do, the padded width is the correct declaration and the reference must be able to replay at it.

## Residual risks (what this contract still does NOT catch)

["MICRO-DEGRADATION BELOW DIGEST_TOL. A verifier degraded just enough to stay inside DIGEST_TOL at every row (e.g. dropping the final 1 of 40 layers, or 1 expert of 8, or shaving the last accumulation refinement) is not caught by the work-binding layer. The benefit is correspondingly small, so this is a bounded-gain residual rather than an open door — but it is real and it is the direct price of using a tolerant digest instead of a hash (a hash would false-fail every honest kernel edit). COVERED BY: the measured separation ratios in calibration set the actual boundary, so publish them; the semantic GPQA judge and free-run gates bound the resulting quality loss; static review of the verify path's layer/expert configuration. Explicitly NOT covered by any token-level test.", "PROMPT-INDEPENDENT-LOOKING KERNEL FITTING. A kernel whose reduction tree is hand-fitted to specific row widths, with a fast non-order-preserving path elsewhere, satisfies E at the widths that appear and is junk elsewhere. E's randomized K schedule (6c) plus the declared_block_width replay shrinks this a lot — the participant must be correct at every width the parent might request, and the parent requests K_r=1 through 16 — but a shape-conditional tile selection in mlx-generated/*.cpp still looks like generic C++ to a diff-based review. COVERED BY: 6c randomization, plus static review specifically instructed to flag shape-conditional numeric paths, plus the CLAUDE.md prompt-independence rule as an enforceable review criterion rather than advice.", "T6 IS NOT COVERED BY E AT ALL, AND MUST NOT BE CLAIMED. A memoized or hardcoded continuation for a timed prompt is bit-identical to the reference argmax trajectory and passes E with zero A3 spend — the cleanest possible result — and it would even satisfy the digest layer if the digests were also memoized. E is an admissibility relation between emitted tokens and the reference's own computation; a lookup that reproduces the reference is indistinguishable from perfect verification. COVERED BY: the prompt POOL sampled per run (6a), which is the primary defense and the reason the pool is non-optional; the n-gram susceptibility analyzer (benchmarkMaxPromptLookupHitRate = 0.03); the no-input-keyed-cache rule; static review; and note that a step-indexed table keyed on decode step rather than input tokens slips that rule as literally worded, so the rule text needs widening to 'any cache whose only possible hit is the measurement protocol, however keyed.'", "REFERENCE-VERIFIER NUMERIC DRIFT. The reference worker is pinned by commit and hash, but its output depends on the box's macOS/Metal stack. A macOS update can shift reference frames, moving positions from A1/A2 into A3 and burning budget on unchanged honest code. COVERED BY: the R5 per-run self-consistency assertion catches nondeterminism within a run but NOT a stable shift across an OS update; that needs an operator runbook item — re-measure EPS_REL, B_A3_TIMED, and DIGEST_TOL after any box software change, and treat a reference shift as a ranking-contract change requiring rebaseline (same discipline the existing pipeline applies to the pinned baseline tree).", "JOB COST AND OPERATIONAL FRAGILITY. L-replay roughly doubles the model-bearing work in the ranked job, on a single serially-queued M5, and adds a second 21.6 GB residency. AUDIT_FRACTION is the dial, but a slow job increases queue pressure, which increases the value of a submit-until-green strategy. COVERED BY: AUDIT_FRACTION plus RANKED_FAILURE_LIMIT together; if the job cannot fit, reduce FID_STEPS before reducing L-timed replay coverage, since the timed leg is the one attached to score.", "ADAPTIVE-DEPTH VERIFICATION KEYED ON DRAFTER CONFIDENCE, IN ITS SURVIVING FORM. E kills the version that dispatches zero target rows (the digest cannot be produced) and the version that skips the lm_head (row_top2 values cannot be produced). What survives is a strategy that runs all rows honestly but chooses among honest kernel variants based on drafter confidence — which is legitimate optimization and should survive. Flagging this as residual only to note that the boundary is now drawn at 'was the row's trunk and vocab projection executed', not at 'was the same kernel used', and that is the intended line.", "STATISTICAL, NOT ABSOLUTE, COVERAGE OF LOW-RATE GAMBLING. A cheater who skips verification on a very small fraction of rounds has a small per-token error rate and may pass a given run. E converts this from 'free on a frozen prompt' into 'detected with probability 1-(1-f)^(S*p) per run, against a sampled prompt, with failures rate-limited', which is a genuine economic closure but not a proof. COVERED BY: FID_STEPS x AUDIT_FRACTION detection power (state it numerically in the published contract), R_A3_MAX, 6a prompt sampling, and 6e rate limiting. Do not describe this as airtight.", "WHAT THE CONTRACT GIVES UP DELIBERATELY. 'Correct' no longer means 'matches the model's sequential greedy output' — A2 admits reference block-frame tokens by design, so the emitted text can legitimately differ from a sequential run, and the candidate and the pinned DFlash baseline can emit different text. This forfeits the operator's ability to cross-validate a suspicious submission against a plain sequential run, and it must be stated in the published rules rather than discovered. COVERED BY: the hidden teacher-forced base case, anchor, free-run, and semantic GPQA gates remain the fidelity authority; they are the only remaining independent meaning of 'correct' and therefore cannot stay at their calibration floors."]

---

# Amendment 1 (2026-07-30) — L2 corrected: logit values bind work, hidden digests do not

Implementation of Criterion E surfaced a defect in L2 as specified above.

**What was wrong.** L2 called for per-row pre-`lm_head` hidden-state digests
"reference-checked for EVERY declared row". That cannot work as an *exact
cross-build* check, for exactly the reason the primary criterion exists: the
candidate build and the pinned-reference build do not produce bit-identical
tensors. The measured near-tie divergence is a scalar-argmax symptom of
accumulation-order differences; a 10240-dimensional hidden vector diverges
*more* readily, not less. An exact digest comparison between candidate and
reference would therefore fail honest submissions on essentially every row.

**What binds work instead.** The load-bearing work binder is
`per_row_top2_logits` — the per-row top-2 **logit values** — compared against
the reference within a tolerance (`DFlashWorkBindingTolerance`, initial
absolute 0.75 / relative 0.02, pending calibration on M5-C by the same method
that produced the near-tie table above). This retains the property that makes
L2 valuable: a verifier degraded at *every* step (early exit, reduced layer
count, coarser dequantization, truncated attention, reduced expert routing)
perturbs logit VALUES at every row — including the 85-95% confident rows where
argmax hides the degradation — whereas `lm_head` elision on a row leaves no
logit values to report at all.

**What the hidden digest is still good for.** `per_row_hidden_digest` remains
in the protocol but is scoped to *self*-consistency, where bit-identity is a
valid expectation:

1. Reference determinism (requirement R5): replay a round twice **in the same
   reference build** and require identical digests. This is what makes the
   admissible sets well-defined; the retired track measured reference-vs-
   reference instability, so this assertion is not optional.
2. Candidate self-replay: the same candidate build re-running the same round
   must reproduce its own digests, which catches nondeterminism and
   state-dependent shortcuts inside one build.

It must NOT be compared candidate-against-reference.

**Calibration owed.** `DFlashWorkBindingTolerance` must be set from measured
honest candidate-vs-reference logit deltas on M5-C, sized with headroom over
the observed maximum, and small enough that a degraded verifier cannot hide
inside it. Until that measurement exists the tolerance is a placeholder and
`tokenFidelityGateStatus` stays `pending-spec`.

# Validation record (2026-07-30, M5-C)

What has actually been executed, as opposed to written. Everything below ran on
M5-C against the organizer-provisioned target
(`/opt/bench-runner/cache/dflash/laguna-xs-2.1-dflash-v1/target`, 13/13 hardlinks
to the pinned NVFP4 snapshot plus the one audited tokenizer overlay) and drafter
(`.../assistant`, manifest-pinned).

| check | result |
|---|---|
| `swift build` both products (CLI + runtime worker) | clean |
| root `swift test` | **459 tests, 7 suites, all pass** |
| vendored `DFlashRollbackSeamTests` | **6/6 pass**, including the crossing-round regression |
| `dflash-reference` (L1) | golden written; `reference_self_consistent=true`; replay **bit-identical** (R5 satisfied) |
| `dflash-probe` (serial K=1 control) | rc=0, `all_tokens_matched=true`, report JSON on stdout |
| `dflash-benchmark` (block, K=8) | rc=0, `all_tokens_matched=true`, rounds=1, accepted=0, rejected=3 |
| KV ledger (L3) | `target_cache_offset_final=13` == 12 seed + 1 decoded |
| row accounting (L3) | `declared_rows_total` == `reference_checked_row_total` == 1 |
| negative path | a plan with fabricated emitted tokens is refused with `tokenNotAdmissible at step 1` — no token id or logit value in the message (L6 redaction holds) |

Both directions are therefore demonstrated: an honest run is admitted, and a run
whose tokens are not reference-admissible is refused, fail-closed, without
leaking reference material.

NOT yet validated, and each blocks enablement:

1. **Long-context viability** — see Amendment 5. This is the substantive
   blocker: the harness is correct but the track may not be rankable.
2. ~~A multi-token, seam-crossing timed run.~~ **DONE — see Amendment 6.** It
   also found a second, unfixed ring-seam defect, this one on the DRAFTER side,
   which fails every block round at a ranked-length seed.
3. ~~`DFlashWorkBindingTolerance` calibration.~~ **MEASURED — see Amendment 6.**
   The constant is now set from observation (5.0 / 0.25, was 0.75 / 0.02), but
   two parts of the calibration remain open and are named there.
4. **L4 ring-index consistency** (Amendment 4) and **L5 reference-drafter
   replay** are designed but unimplemented.
5. ~~The box wrapper passes nonexistent flags.~~ Reconciled; the wrapper now
   matches the CLI surface, `bash -n` clean, `--preflight-only` OK, manifest
   re-signed, janitor audit clean.
6. **Pre-generated goldens desynchronize.** A golden built before the run is
   teacher-forced on the REFERENCE's chain. The first token a candidate emits
   that is admitted but different (declared-frame or residual) puts every later
   row on a different context, and the run then fails `tokenNotAdmissible` for
   reasons that are an artifact of golden pre-generation, not misbehaviour. See
   Amendment 6; the ranked pipeline is unaffected because it generates the
   golden AFTER the run from the observed emitted plan, but every local
   pre-generated golden inherits this.

# Amendment 5 (2026-07-30) — MEASURED AT LONG CONTEXT: DFlash is currently a NET SLOWDOWN

The first measurement ever taken at a long prompt (1755 tokens, K=8, M5-C,
`mlx-bench dflash`) shows the drafter losing to plain sequential decode:

| tree | base tok/s | dflash tok/s | speedup | accepted |
|---|---|---|---|---|
| pre-fix standalone (`~/projects/laguna-dflash`) | 81.9 | 55.8 | **0.68x** | 1.47/7 |
| fixed dev-repo tree (organizer target + drafter) | 81.6 | 43.5 | **0.53x** | 1.47/7 |

Compare the short-prompt sweep (51-token prompt, same K=8): 83.4 -> 154.9 tok/s,
**1.86x**, accepted **3.55/7**.

So acceptance collapses from 3.55/7 to 1.47/7 as context grows, and with it the
speedup falls from 1.86x to below parity. Fewer accepted drafts means the wasted
verify rows dominate: at K=8 a round costs an 8-row target forward regardless,
and emitting ~2.5 tokens for it is worse than emitting 1 token from a 1-row
forward.

Consequences that the track cannot be enabled without resolving:

1. **The scoring floor rejects everything.** `decodeSpeedupFloor` is a hard 1.0
   on the ratio-of-means aggregate. At the ranked window (512-token seed) the
   drafter is far closer to this long-context regime than to the 51-token regime
   the 1.86x came from, so the expected outcome is REJECT for every submission,
   including an honest optimal one. A track that cannot rank anything is not
   ready.
2. **Every previously reported DFlash number is short-prompt-only.** The 1.56x-
   1.86x sweep, the K=4..16 acceptance table, and the dev-repo 1.60x figure were
   all taken at 26-68 token prompts. None of them describes ranked behaviour.
   They must not be quoted as evidence for this track.
3. **The cause needs isolating before any go-live.** Candidate explanations, in
   the order worth testing: (a) the drafter conditions on target hidden states
   captured at layers [1,13,25,33,39] and its acceptance may simply degrade with
   sequence length, i.e. an inherent property of this checkpoint; (b) the
   sliding-window seam forces a cache snapshot every round once the ring has
   wrapped (see Amendment 3), which is real per-round cost the short runs never
   paid -- note the fixed tree is SLOWER than the pre-fix one here (0.53x vs
   0.68x), consistent with paying snapshot cost, though the two runs also differ
   in target directory and are single samples; (c) block-shaped attention over a
   wrapped ring may hit a slower kernel path than the 1-row decode.
4. **Decide the block size from long-context data, not the short sweep.**

## RESOLVED (same day): the track IS viable, but only at K=3

A K sweep at the same 1755-token prompt (64 decode tokens, base 81.5 tok/s)
answers the viability question:

| K | 2 | **3** | **4** | 6 | 8 | 16 |
|---|---|---|---|---|---|---|
| dflash tok/s | 66.7 | **89.2** | **86.1** | 77.4 | 74.7 | 64.6 |
| speedup | 0.82x | **1.09x** | **1.06x** | 0.95x | 0.92x | 0.79x |
| accepted | 0.62/1 | 1.17/2 | 1.25/3 | 1.33/5 | 1.33/7 | 1.33/15 |

Findings:

- **K=3 clears the floor at 1.09x**; K=4 marginally at 1.06x. Everything else,
  including the short-prompt optimum K=8, is BELOW 1.0.
- **Acceptance saturates at ~1.33 accepted drafts for every K >= 6.** The drafter
  reliably gets only about one and a third tokens right at this context length,
  so wider blocks buy nothing and pay for more verify rows. That is the whole
  explanation for the ranking above, and it is why the short-prompt sweep (which
  peaked at K=8 with 3.55/7 accepted) inverts here.
- The ranked default is therefore **3**, set in the workflow with this table as
  its recorded basis.

Two consequences the operator must weigh before enabling:

1. **The margin is thin against a hard floor.** 1.09x sits 9% above a
   pass/fail boundary, and the paired protocol tolerates roughly 0.6% run-to-run
   variation, so the headroom is real but not generous. Consider whether the
   floor should be expressed with an explicit tolerance for this track.
2. **The score measures block-path efficiency only.** Both sides of the ratio
   run the SAME candidate build, so a generic kernel win that speeds up
   sequential decode and block verify equally cancels out. Participants can only
   move the score by making speculation specifically cheaper — verify batching,
   cache handling, rollback cost — not by optimising the model forward in
   general. That is a narrower optimisation surface than the serial track, and it
   should be stated in any participant-facing description.

`official_scoring_enabled` still stays false until the remaining items in the
validation record are closed (tolerance calibration, L4/L5, the seam-crossing
timed run, and the box-wrapper flag fixes), but the blocking *question* — is
there any configuration in which this track can rank — is now answered: yes, at
K=3.

# Amendment 4 (2026-07-30) — L4 corrected: KV digests cannot be compared across builds either

L4 as specified asks for "parent-demanded KV digests at the committed offset with
vacancy sentinels beyond it". The digest half of that does not work, for exactly
the reason Amendment 1 gave for hidden states: the candidate build and the
pinned-reference build do not produce bit-identical tensors, and a KV tensor is
far larger than a single hidden vector. A candidate-vs-reference KV digest
comparison would fail honest submissions on the first round.

What L4 can and cannot contribute, corrected:

1. **NOT checkable:** candidate KV digest vs reference KV digest. Dropped. The
   `kv_digest` / `kv_vacancy_digest` protocol fields stay RESERVED and unused
   rather than carrying a check that cannot hold — shipping a comparison that
   fails honest code would be worse than shipping none.
2. **Already covered, and this is the important realisation:** the *effect* of
   stale KV rows is caught by L2. If a submission leaves rejected rows in the
   cache, every subsequent row is computed against polluted state, so its
   per-row top-2 logit VALUES drift from the reference's — and L2 compares
   exactly those within a tolerance. Rollback elision does not need its own
   output test; it needs its numerical consequence to be bound, which it is.
3. **Genuinely additional and checkable WITHIN a build:** ring-index
   consistency. `RotatingKVCache.trim` moves `offset` and `idx` together. An
   elision that decrements the logical offset but leaves the physical write
   index (and therefore the rejected bytes) in place is visible as
   `idx != offset mod maxSize`, with no cross-build comparison involved. This is
   the structural check L4 should carry, alongside the cache-offset ledger the
   session already enforces.
4. **The wrap-seam leg is promoted from audit instrument to mandatory
   regression test** — see Amendment 3. It is where rollback is hardest and
   where a shortcut is most tempting, and Amendment 3 shows a real defect lived
   there undetected because no short-prompt run reaches it.

Net: L4's value is (3) plus (4), not the digest comparison it was written
around. `tokenFidelityGateStatus` stays `pending-spec` until (3) is implemented
and (4) has actually run green at a 512-token seed.

> **Item (3) is withdrawn by Amendment 8**, and with it this paragraph's gate
> condition. `idx == offset mod maxSize` is false on `updateConcat`, which is the
> path every seed prefill and every K>=2 verify takes, so the check would have
> rejected the first round of every honest ranked run. The gate stays
> `pending-spec` for the reasons Amendment 8 lists instead. Item (4) stands and
> has now run.

# Amendment 3 (2026-07-30) — the wrap seam was a real defect, not just an audit target

L4 specified a sliding-window wrap-seam leg as an *audit instrument*, on the
theory that rollback elision might hide there. Implementation found something
stronger: **block decode was outright broken at the seam**, and the ranked
window is precisely where it bites.

Mechanism. `RotatingKVCache.isTrimmable` is `offset < maxCacheSize`, which is
correct rather than conservative — once the ring wraps, rolling the offset back
would need the entries the wrap just overwrote, and those are the oldest rows
still inside the window. So a wrapped cache must be rolled back by snapshot and
replay, not by trimming. But `makeDefaultDFlashCacheRollbackState` decided
whether to snapshot *before* the block was written, while the trim happens
*after*. The single round that STARTS trimmable and ENDS wrapped therefore got no
snapshot and could not trim, and threw `untrimmableCache`.

Why it was not seen earlier: Laguna's sliding window is 512 and every bring-up
measurement used a 26-68 token prompt, so the seam was never crossed.

**CORRECTION (measured after this amendment was first written).** The original
text claimed "every scored run would have failed". That is WRONG and is retracted.
A 1755-token prompt runs to completion on the PRE-FIX binary, because by then the
ring is already wrapped when decode starts: `isTrimmable` is false from the first
round, so a snapshot is always taken and copy+replay works. The defect requires
the round to CROSS the boundary mid-flight, i.e. a seed length in roughly
[maxSize - K + 1, maxSize - 1] — about 505..511 for K=8 — which is exactly what
the agent's 512-token warmup seed hit. So the hole is real but NARROW, not
universal.

The fix stands on its own merits (that band includes seeds a ranked run can
legitimately produce, and failing a run for crossing a ring boundary is never
correct), but the severity claim was overstated and the seam is not what makes
this track unready. Amendment 5 records what actually does.

Fix: the snapshot decision now takes the width the round is about to write
(`plannedWriteCount`) and snapshots when any cache would cross its ring
boundary; and a short trim falls through to the snapshot instead of throwing, so
a partial trim is discarded rather than compounded. All three round
implementations (greedy, batched generator, batched benchmark) pass the width.

Consequences for the contract:
- L4's wrap-seam leg is now MANDATORY as a *regression test*, not only an audit:
  a seam-crossing run must be part of validation, because a seam bug is
  invisible to any short-prompt measurement.
- The measured 1.56x-1.86x speedups were all obtained on short prompts and are
  therefore NOT evidence about ranked-window behaviour. Speedup at a 512-token
  seed must be re-measured, and it may differ: the seam forces a snapshot round
  whose cost the short-prompt runs never paid.

# Amendment 2 (2026-07-30) — what this track actually measures

The DFlash target is the **vendored** `LagunaModel` (reached through
`LLMModelFactory`), because that is the type conforming to `DFlashTargetModel`.
It is NOT `Sources/MLXFastModel/LagunaRuntimeModel.swift`, the heavily
optimized forward that the serial ranked track scores. Consequences, which
belong in any participant-facing description of the track:

- DFlash speedups are measured against the **reference** target forward, so
  they are not additive with serial-track optimizations and are not comparable
  to serial-track scores as absolute tokens/second.
- The track's editable surface is correspondingly the DFlash runtime
  (`MLXSpeculative/*`, `DFlashTarget.swift`, `DFlashVerifyLinear.swift`) plus
  the vendored model/kernels — consistent with `benchmark.dflash.json`.
- Both tracks nonetheless load the SAME NVFP4 group-16 reference checkpoint, so
  the model under test is identical; only the forward implementation differs.

---

# Amendment 6 (2026-07-30) — the seam-crossing run happened; it found a SECOND seam defect, and it calibrated L2

Validation-record items 2 and 3 are closed by measurement, and the measurement
produced three findings that change the contract. All numbers below are M5-C,
one build for both candidate and reference, organizer-provisioned target and
drafter, `official_scoring_enabled` still false.

## 0. How a long seed was reached at all

`dflash-reference` gained a chain-generation mode. The trusted binary links no
tokenizer, so a long seed cannot be written as text; instead the reference
GENERATES it — `--seed-generate M` extends the seed by M of the reference's own
width-1 argmax tokens, then `--generate N` produces the N scored tokens, all in
one process with the model resident once. That is what makes seed length a dial
and the 512-slot ring reachable. Each step is its own stateless reference
request, so every generated token comes from exactly the computation the
golden's replay later uses; the cost is quadratic in the number of steps, paid
once per golden, outside any timed window.

Goldens also now record a GENUINE block frame for every width the parent could
declare (`count ... blockSize`, plus width 1 for free, since a one-row frame is
the sequential frame) rather than one width computed at the wrong width.
Widening the replayed frame is sound for the scored rows: attention is causally
masked, so rows `0 ..< count` cannot see the tail rows the widening filled in
with later emitted tokens instead of the candidate's unknown rejected drafts.

## 1. The target-side wrap-seam fix HOLDS

| seed | regime | K | tokens | rounds | accepted | rejected | offset_final | verdict |
|---|---|---|---|---|---|---|---|---|
| 509 | ring seam (starts trimmable, ends wrapped) | 1 | 24 | 24 | 0 | 0 | 533 = 509+24 | **pass**, 24/24 exact |
| 509 | ring seam | 3 | 24 | 10 | 15 | 0 | 533 = 509+24 | **pass**, 24/24 exact |
| 600 | fully wrapped | 1 | 128 | 128 | 0 | 0 | 728 = 600+128 | **pass**, 128/128 exact |

No `untrimmableCache` on any of these, `residual_divergence_count` 0
throughout, `declared_rows_total >= emitted_token_total` with equality except
the one K=3 round that declared 3 rows and emitted 2 at the window edge, and the
parent-derived KV ledger matching `seed + committed` exactly in every case.
Amendment 3's fix is therefore confirmed by a run that actually crosses the
boundary, which is what that amendment asked for.

## 2. NEW DEFECT: the drafter-side failure (root-caused and FIXED in §5)

> **Superseded by §5.** The trace and the ranked-impact assessment below are
> accurate and are kept as the investigation record. The diagnosis is not: this
> is not a second wrap seam and none of the three fixes proposed at the end of
> this section was the right one. The drafter never needed to trim at all — the
> session was asking it to, because of an off-by-one in the argument it passed.
> Read §5 before acting on anything here.


Seed 600 with K=3 fails on its FIRST scored round:

```
runtime worker dflash_decode_block failed: untrimmableCache
```

Traced (`MLX_DFLASH_TRACE_CACHE_SEAM=1`; the `MLX_` prefix is load-bearing,
`MLXFAST_*` is deliberately not in `sanitizedRuntimeWorkerEnvironment`'s
allowlist so a harness-named variable never reaches worker code):

```
seed 509, round 1: draft_offset=509 committed=508 extra=1 draft_trimmable=true  draft_max=511  -> trim OK
seed 509, round 2: draft_offset=511 committed=511 extra=0 draft_trimmable=false draft_max=511  -> no trim attempted
seed 600, round 1: draft_offset=600 committed=599 extra=1 draft_trimmable=false draft_max=511  -> THROWS
```

The throw is `DFlashGreedyRound.swift:124`, the DRAFT cache alignment trim — not
the target rollback that Amendment 3 fixed. The drafter is 5 sliding-window
layers with `sliding_window: 512`, so its `RotatingKVCache` has
`maxSize` 511 and `isTrimmable == offset < maxSize`; once the seed alone exceeds
that, `trimPromptCache` returns 0, the requested 1 is not satisfied, and the
round fails. Seed 509 survives only by luck: `extra` is 1 on the single round
where the cache is still trimmable and 0 forever after, and the trim is guarded
by `if extraDraftContext > 0`.

**This fires on the ranked configuration.** The frozen window is a 512-token
seed, so `draft_offset` is already past 511 at round 1 and EVERY block round
fails. The serial control (K=1) is unaffected because it never drafts, so the
paired measurement fails on exactly the side that produces the speedup.

Not fixed here, deliberately. Every candidate fix is a design decision the
parent should make rather than something to patch under a measurement task:
snapshot/restore the draft cache the way the target does; align the drafter's
cache at `begin()` so the extra row never exists; or relax `RotatingKVCache`'s
trim refusal for the recoverable "forget the last n writes" case — the last
would change semantics for every caller including the serial track's sliding
layers, and should not be done casually. It also interacts with the unimplemented
L5 drafter replay.

## 3. L2 calibration: measured, constant changed, and two gaps stay open

Gap = `|candidate - reference|` per top-2 slot. Logits are BF16; top-1 magnitudes
ran 21.4..26.5, so one ULP is 0.125.

| regime | context | n | max | p99 | p50 | mean | max relative |
|---|---|---|---|---|---|---|---|
| short, K=1 | 12..21 | 18 | 0.250 | 0.250 | 0.125 | 0.104 | 0.015 |
| seed 509, K=1 | 509..533 | 48 | **3.375** | 3.375 | 0.250 | 0.401 | — |
| seed 509, K=3 | 509..533 | 48 | 2.750 | 2.750 | 0.375 | 0.495 | — |
| seed 600, K=1 | 600..728 | 256 | 2.500 | 1.750 | 0.250 | 0.316 | 0.120 |

Two things in that table decide the constant:

- The gap is 1-2 ULP at short context and up to **27 ULP** once context passes
  the sliding window. This is a WINDOW-EDGE FRAME effect — incremental
  ring-cache decode versus the reference's stateless prefill-and-replay — not
  arithmetic noise. The ranked window lies entirely in the wrapped regime, so the
  large-gap regime is the only one that matters.
- It is not a seam transient. In the 256-sample run the 16 gaps above 1.0 are
  spread across the whole run (rows 7, 23, 25, 32, 34, 39, 46, 61, 63, 76, 81,
  89, ...), not clustered at the crossing.

`DFlashWorkBindingTolerance` therefore moves from the placeholder 0.75 / 0.02 to
**absolute 5.0, relative 0.25**: 1.5x the observed maximum over 352 long-context
comparisons (headroom for the unobserved tail of a sample that small, without
doubling), and ~2x the largest observed relative gap. The old relative 0.02 was
dead code — below even the short-context gaps. All three passing runs in §1 were
re-run with the compiled-in constant and no override.

What this does NOT establish, stated because it bounds the claim:

- **The cross-build term is unmeasured.** Candidate and reference were the SAME
  build here, so these gaps are frame differences only. A real submission adds
  its own kernels, and that term is additive on top.
- **The false-negative side is unmeasured.** No deliberately degraded verifier
  was run, so "a cheaper verifier cannot hide inside 5.0" is NOT a measured
  claim. At 5.0 against top-1 logits near 21, the binder prices only gross
  degradation. It still constrains a verifier that skips the per-row lm_head
  entirely — it then has no values to report at all — but it does not price a
  subtly reduced trunk. `LOGIT_VAL_TOL` in the calibration list stays open, and
  the honest reading is that L2's discriminating power at long context is WEAK
  and the fidelity authority remains the hidden gates.

Both gaps must close before `official_scoring_enabled` is turned on.

## 4. Calibration machinery, and why it cannot weaken a ranked run

Measuring §3 required running with the check widened, which means a flag that
widens a gate now exists (`--work-binding-tolerance-absolute/-relative`). It is
refused outright when `MLXFAST_OFFICIAL_BENCHMARK_RUN=1`, prints a WARNING that
the run is not contract-enforcing, and is pinned by
`wideningTheWorkBindingToleranceIsRefusedOnTheOfficialPath` in
`DFlashTrackTests`. The per-comparison gap array is published only on such a
run: on a ranked run a per-row proximity trace against a hidden prompt is an
oracle signal, which is what L6 keeps out of published artifacts. Aggregates
(count, max, p99, p50, mean, and the tolerance actually used) are always
reported, so an audit can see both the distribution and that the binding
compared anything at all.

# Amendment 7 (2026-07-30): the drafter failure was an off-by-one, not a seam

§2 reported that seed 600 + K=3 throws `untrimmableCache` from the DRAFT cache
alignment trim, and that this fires on the ranked 512-token seed. Both facts
hold. The diagnosis — "the same wrap seam exists on the drafter side" — does
not, and the three fixes it proposed (snapshot the draft cache, align it at
`begin()`, relax `RotatingKVCache`'s trim refusal) would all have been wrong.
The last would have been actively harmful: it would have weakened a guard that
is protecting a real invariant, for every caller including the serial track's
sliding-window layers.

## What the drafter's cache actually is

`DFlashDraftModel`'s attention (`DFlashDraftModel.swift`) does not keep a
conventional autoregressive KV cache. Every round it re-supplies the target
hidden states as cross-attention context, writes ONLY those context rows to the
cache (`cache.update(keys: contextKeys, values: contextValues)` — the proposal
rows are concatenated for attention and never cached), and derives all RoPE
positions from the cache's offset:

```swift
queries      = rope(queries,      offset: baseOffset + contextLength)
contextKeys  = rope(contextKeys,  offset: baseOffset)
proposalKeys = rope(proposalKeys, offset: baseOffset + contextLength)
```

So `cache.offset` is a POSITION COUNTER, and the sliding-window truncation a few
lines above bumps it deliberately (`cache.offset += skip`) when the supplied
context is wider than the window, precisely to keep absolute positions right.
For the invariant to hold, `baseOffset + contextLength` must equal the absolute
position of the round's bonus token, which is the target cache's offset:
`seedTokenCount + decodedTokenCount`.

## The actual defect

`runDFlashGreedyRound` derives the drafter's committed position as
`promptTokenCount + generatedTokenCount - 1`, where `generatedTokenCount` counts
EMITTED tokens — including the seed prefill's bonus token, which is why
`measureDFlashThroughput` initializes its counter to 1 rather than 0.
`LagunaDFlashBlockSession` passed `decodedTokenCount`, which counts KV ROWS: one
fewer, because the bonus token's row is written by the round that consumes it as
its first verify row. So the session declared a committed position one short of
the truth, and the round dutifully asked the drafter to give back one row.

Everything in §2's trace follows from that one row:

* `draft_offset=509 committed=508 extra=1` — the drafter is at the correct
  position; the SESSION's idea of committed is 1 low.
* Seed 509 "survives by luck" — no: the 1-row trim succeeds while the ring is
  still unsaturated, and from the next round on `extra` is 0 because the trim
  itself established the (wrong, 1-low) alignment as self-consistent.
* Seed 600 throws — the same 1-row trim, now on a ring that has saturated and
  correctly refuses to rewind.

The fix is the one-line convention correction, in
`Sources/MLXFastModel/LagunaDFlashBlockSession.swift`:

```swift
generatedTokenCount: decodedTokenCount + 1,
```

`committedDraftOffset` then equals `seedTokenCount + decodedTokenCount`, which
is exactly the target cache offset the session already asserts in its ledger.
`extra` is 0 on every round including the first, the trim never fires, and the
saturated-ring refusal is never reached. No snapshot, no new cache API, and no
change to `RotatingKVCache` — whose `isTrimmable == offset < maxSize` was right
all along.

## The second, quieter half

The off-by-one also mis-positioned the drafter on every seed, not just seeds past
the window. Round 1 wrote its context rows at correct absolute positions; the
1-row trim then pulled the counter back, so from round 2 onward every drafter
query and context key sat one position low. The shift is uniform, so relative
geometry within the decode region survived — but the seed-to-decode boundary was
squeezed by one position (round 2's context key lands on the position the seed's
last token already occupies). That is a draft-QUALITY defect, not a correctness
one: the target verifies every emitted token, so a misaligned drafter can only
lower acceptance. It is a plausible contributor to acceptance saturating at
1.33 for every K >= 6, and the K sweep that produced the viability numbers ran
through the fork's own benchmark path, which uses the correct convention — so
those numbers are NOT invalidated, but the session path was not measuring the
same drafter alignment they did.

## Hardening

The alignment now has an explicit precondition instead of being an emergent
property of two counters agreeing. Before each block round the session requires
the supplied context width to equal the number of positions the target advanced
since the drafter last wrote, and refuses the block otherwise. That closes a
related silent failure the old code had no defence against: a width-1 serial row
in the middle of a session advances the target without feeding the drafter, so
every later block would have drafted from a prefix the drafter never saw — with
no symptom except worse acceptance.

The stale comment on the construction-time `canTrimPromptCache` guard ("the
round trims the draft cache back to the committed offset every time") is
corrected. That comment is what makes the wrong fix look right, and it is the
reason §2 reached for one.

# Amendment 8 (2026-07-30): L4's ring-index check is not implementable as specified

Amendment 4 reduced L4 to two contributions after dropping the KV-digest
comparison: (3) ring-index consistency, `idx != offset mod maxSize`, as a
build-independent structural tell for an elision that rewinds the logical offset
but leaves the rejected bytes resident; and (4) the wrap-seam leg promoted to a
mandatory regression test. It also made `tokenFidelityGateStatus` conditional on
(3) being implemented.

(3) cannot be implemented, because the invariant is false on the code path block
decode takes. `RotatingKVCache.update` dispatches on write width. Only a
single-row write goes through `updateInPlace`, where `idx` is genuinely a ring
write index. Every wider write — the seed prefill, and every K>=2 verify block —
goes through `updateConcat`, which rebuilds the array in TEMPORAL order and sets
`idx` to the physical row count, which it deliberately lets grow to
`maxCacheSize + S - 1` so that every row still gets a full window. On that path
`idx` is not a ring index at all.

Concretely, for the ring the target actually uses (`maxSize` 511), a 600-row
prefill followed by one K=3 block leaves `offset == 603`, `idx == 513`, and
`offset % maxSize == 92`. A gate demanding `idx == offset % maxSize` would have
rejected that honest round — the FIRST round of every ranked run. This is now
pinned by `multiRowWritesLeaveTheRingInTemporalOrderNotAtOffsetModuloMaxSize` in
`DFlashRollbackSeamTests`, so the claim is executable rather than asserted.

This is the third time a proposed cross-checking instrument for this track has
turned out to be unsound on contact with the implementation (Amendment 1: hidden
digests across builds; Amendment 4: KV digests across builds; here: ring indices
within a build). The pattern is worth naming: instruments designed from the
contract's vocabulary rather than from the cache's actual code keep asserting
invariants the code does not hold. Shipping any of them would have failed honest
submissions on round 1 while catching nothing.

## What carries L4's weight instead

1. **The cache-offset ledger**, already enforced per round in
   `LagunaDFlashBlockSession`: physical KV offset must equal
   `seedTokenCount + decodedTokenCount`. A submission that skips rollback to save
   time trips this before any of its tokens is scored.
2. **L2's numerical consequence binding** (Amendment 4 item 2, unchanged and
   still the strongest argument here): stale rows pollute every subsequent row's
   logits, and L2 compares per-row top-2 logit VALUES against the reference.
   Rollback elision does not need its own output test, it needs its consequence
   priced — with the caveat Amendment 6 section 3 records about how weak that
   pricing is at long context.
3. **Snapshot exactness**, newly pinned by
   `copyingARotatingCacheCapturesRingStateExactlyAcrossTheSeam`. The seam fix
   falls back on `copy()` whenever a round would cross the ring boundary, so the
   fix is only as good as that copy: the test asserts offset, ring index,
   physical row count, `maxSize` and the stored BYTES all survive a write that
   happens after the snapshot is taken. This is a self-consistency property, so
   unlike items 1 and 2 of Amendment 4 it is build-independent by construction.
4. **The mandatory seam leg** (Amendment 4 item 4), which has now actually run:
   seeds 509 and 600 complete with every token exact, and seed 600 with K=3 is
   the case that exposed the drafter defect of Amendment 7. A short-prompt suite
   would have caught none of it.

## Effect on the gate

`tokenFidelityGateStatus` stays `pending-spec`, and the reason is NOT item (3) —
that condition is withdrawn as unsatisfiable rather than met. What still blocks
`official_scoring_enabled`:

* **L5 reference-drafter replay is unimplemented.** It is the only layer that
  binds the DRAFTER to the pinned weights; nothing else does. Note the threat it
  covers is narrower than it first appears, because a degraded drafter is largely
  self-punishing: worse drafts mean lower acceptance and therefore a lower score,
  and every emitted token is still verified by a target whose per-row work L2/L3
  bind. What L5 uniquely closes is a *different, cheaper* drafter that is still
  good — most sharply, an excluded input-derived drafter (prompt-lookup, n-gram,
  token-history) substituted for the trained one. Its design inherits Amendment
  1's cross-build problem: the reference replays on the candidate's own target
  hidden states, so draft tokens can differ at near-ties and the check needs a
  residual bucket rather than equality.
* **The two L2 gaps of Amendment 6 section 3**: the cross-build tolerance term is
  unmeasured and additive, and no degraded verifier has been run, so "5.0 is
  tight enough to catch cheapening" remains unmeasured.

Until those close, the track is runnable and honest about what it does not yet
prove, which is the state the pinned
`officialScoringRequiresAnImplementedTokenFidelityGate` test enforces.

# Amendment 9 (2026-07-30): L5 reframed — the drafter is mostly self-policing

L5 (reference-drafter replay) is the last layer Amendment 8 lists as blocking
`official_scoring_enabled`. Doing the cross-build analysis BEFORE implementing it
— the step whose absence produced Amendments 1, 4 and 8 — shows the replay is a
weak instrument, and that the threat it aims at is better handled the way the
serial track already handles the identical technique.

## Why the replay is weak

The reference cannot replay the drafter on the candidate's own inputs. The
drafter's input is the target's hidden-state context, which is megabytes per
round; it does not cross the JSON protocol, and Amendment 1 already established
that a digest of it cannot be compared across builds. So a replay must run the
pinned drafter on the REFERENCE's hidden states, which differ numerically from
the candidate's. Draft tokens are argmaxes of a 5-layer model — far less
confident than the target's — so honest candidates would disagree with the
replay at a substantial and unpredictable rate. The check would need a residual
bucket wide enough to admit that, at which point a cheaper-but-decent drafter
fits inside the bucket too. This is Amendment 6 section 3's problem, worse.

## Why most of the threat does not need catching

A degraded drafter is self-punishing. Worse proposals mean a shorter accepted
prefix, which means more rounds for the same token count, which means a lower
score — and every emitted token is still produced and verified by the target,
whose per-row work L2 and L3 bind. There is no version of "cheapen the drafter"
that raises the score through a mechanism the contract does not already price.
The saving is bounded anyway: the drafter is 5 layers against the target's 40.

What is left is genuinely adversarial but narrow: a *different and still good*
proposal source. Concretely, an input-derived one — prompt-lookup, n-gram,
suffix or token-history drafting — which on repetitive text proposes well while
skipping the drafter entirely.

## How that is enforced

By rule and static review, which is exactly how the serial track enforces its own
speculation ban against the same list of techniques. The rule is now stated for
participants in `AGENTS.md`: every proposed token must come from a forward pass
of the pinned drafter on that round's bonus token and hidden context; input-derived
proposal sources are excluded even when generic, production-useful or bit-exact,
and so are hybrids that fall back to one. Dispatch-level optimisation of the
drafter remains explicitly allowed.

Note the serial track ranks today with no L2-equivalent instrument at all: it
relies on exact-token gates plus static review. The DFlash track cannot use
exact-token gates — that is the premise of this whole contract — which is why it
carries L1/L2/L3/L6 as well. It is not weaker than the live track here; it is
strictly stronger, and it uses the live track's mechanism for the one threat that
mechanism already covers.

## A detector worth building later, and why it is not shipped now

There IS an exact, build-independent instrument for this specific threat, because
it tests token ids rather than floats: **draft provenance**. An input-derived
drafter proposes, by construction, n-grams that already occur in the context. So
journal each round's proposed tokens and measure how often the proposed block
appears verbatim earlier in the sequence. A trained drafter's proposals do that
sometimes; a lookup drafter's do it nearly always, and the separation is in the
token ids, with no cross-build term at all.

It is not shipped because it is a statistical detector and its threshold must
come from a measured null distribution on the pinned drafter across several
prompt types — including deliberately repetitive text, where a trained drafter's
provenance rate legitimately rises. Shipping it with a guessed threshold would
repeat the mistake this contract has already made three times. The protocol
field to journal proposed draft tokens is the prerequisite, and is not yet added.

## Effect on the gate

L5 is withdrawn as a *runtime* blocker: it is replaced by a participant rule with
the same enforcement mechanism the live serial track uses, plus a specified
future detector. `tokenFidelityGateStatus` therefore turns on the two L2 gaps of
Amendment 6 section 3 alone — the unmeasured cross-build tolerance term, and the
absence of any degraded-verifier run to show the tolerance catches cheapening.
Both are measurable without new design work.

Whether those two gaps must close before the track ranks, or whether they are
acceptable as documented limits given that the hidden gates remain the fidelity
authority, is a policy decision for the organizer and deliberately not decided
here. The pinned test keeps official scoring off until someone decides.

# Amendment 10 (2026-07-30): every green result so far was measured on degenerate text

This is the most important entry in this document. Two contract defects are
recorded below, but the framing matters more than either: **the entire validation
record of this track was established on seed material with no near-ties in it**,
which is precisely the condition the contract exists to handle.

Every golden used for the e2e passes, the seam-crossing runs, the L2 calibration
and the K sweep was built from a greedy SELF-continuation of the reference. Greedy
self-continuation degenerates into repetition, and the resulting goldens are not
representative in the one dimension that decides admissibility:

| golden | seed len | distinct seed tokens | rows with top-2 gap < 0.25 | min top-2 gap |
|---|---|---|---|---|
| `seam-512-golden.json` (ranked-window fixture) | 512 | 122 | **0** | 1.875 |
| `seam-b-golden.json` | 600 | 122 | **0** | 2.625 |
| `seam-a-golden.json` | 509 | 122 | **0** | 1.875 |
| `varied-512b-golden.json` (varied prose/code) | 512 | 317 | 3 | **0.0000** |
| `varied-512-golden.json` | 512 | 317 | 3 | **0.0000** |

The repetitive goldens contain no row whose top-2 logits are within 1.8 of each
other. The model is so confident on repetitive text that the near-tie regime —
the regime Criterion E was designed for, and the reason exact-token matching was
abandoned — is never entered. Varied text contains EXACT 0.0000 ties, where the
argmax is decided by tie-break order and is therefore frame-dependent by
construction.

Consequences: draft acceptance measured 100% on the repetitive fixtures and 69%
on varied text; the residual bucket was never exercised; and no measurement taken
against those fixtures should be treated as evidence about ranked behaviour. The
hidden ranked prompts are prose, not degenerate self-continuations.

## Defect 1: `reference_self_consistent` does not check what its name claims

Three goldens report `reference_self_consistent: true` while containing rows where
the reference contradicts ITSELF: `emitted_tokens[i]` differs from
`rows[i].sequential_argmax` for the same position i.

```
varied-512-golden.json    emitted != sequential_argmax at rows 1, 3
varied-512b-golden.json   emitted != sequential_argmax at row 5
varied-512b-k3-golden.json  none  (same seed -- so this is frame/schedule dependent)
```

The cause is not nondeterminism, it is two frames inside one artifact: the
reference EMITS by incremental decode and SCORES by stateless prefill replay, and
those two frames disagree at near-ties — the same divergence Amendment 6 section 3
measured at up to 27 ULP. The self-consistency check evidently compares repeated
scoring runs to each other, not the emitted chain to the scored chain, so it
passes artifacts that contain the contradiction.

The consequence is severe: a golden with such a row REJECTS the honest candidate
that reproduces the reference's own emitted token, because admissibility is
checked against `sequential_argmax`. The reference is the authority and it is
disagreeing with itself.

Fix direction (not yet implemented): anchor the admissibility rows to the frame
the reference actually emitted in, or compute `sequential_argmax` incrementally
so the two frames coincide; and make the self-consistency check compare
`emitted_tokens` against `rows[*].sequential_argmax` so an artifact carrying the
contradiction cannot be published as consistent.

## Defect 2: the residual budget is 1 slot at the ranked window

`experimentalDFlashResidualDivergenceBudgetPerThousand` is 5, and the budget is
`max(1, ceil(totalTokenCount * 5 / 1000))`. At the frozen 128-token decode window
that is `ceil(0.64) = 1`. One slot, for the whole scored run.

The calibration behind 5-per-thousand is recorded in `Constants.swift` as "14
events across 2,304 emitted positions (<= 0.61%), every one of them explained by
the declared-frame admission" — i.e. the residual bucket itself was measured as
essentially unused. That measurement was taken on the degenerate fixtures above,
where there are no near-ties to diverge at.

On varied text, a 64-token control run already consumes its 1 slot
(`residual_divergences: 1`), and a 128-token run hit `residualBudgetExhausted`.
With 3 sub-0.25-gap rows per 128 positions plus the Defect 1 contradictions, an
honest submission on a realistic hidden prompt is expected to need several slots
and will instead be failed.

Note also that the budget does not scale within the ranked window: 64 tokens and
128 tokens both yield exactly 1, because 5-per-thousand rounds to 1 for anything
under 200 tokens. Whatever the right rate is, the ranked window's budget should
not be a rounding artifact.

Fix direction (not yet implemented): recalibrate the rate against VARIED goldens,
and size it from the measured near-tie density rather than from a run that never
entered the near-tie regime. Both defects must close before
`official_scoring_enabled` can be considered, and Defect 1 should close first
because it corrupts the material any recalibration would use.

# Amendment 11 (2026-07-30): matched-length pairs, and what the score is actually measuring

Amendment 10 established that the fixtures were degenerate. This entry establishes
the performance consequence, and corrects two wrong readings made along the way —
one of them mine.

## The measurement trap: token counts must match

`seconds_per_token` on this track is NOT comparable across runs of different
length, because a large fixed cost sits inside the measured window. Measured on
the SAME golden, same code, same box:

| golden | K | 64 tokens | 128 tokens |
|---|---|---|---|
| repetitive | 1 | 0.029784 s/tok | 0.021420 s/tok |
| repetitive | 4 | 0.028149 s/tok | 0.019600 s/tok |

Decomposing total time as `fixed + n * marginal`:

```
K=1:  marginal 13.056 ms/token,  fixed 1.071 s
K=4:  marginal 11.051 ms/token,  fixed 1.094 s
```

The fixed term is the 512-token seed prefill, which the driver deliberately
charges into the decode measurement ("the clock starts immediately before the
request so the seed cost cannot be hidden"). At the frozen window that is ~1.08 s
of ~2.74 s — **about 39% of the scored quantity is prefill, not decode.** It is
identical on both sides, so it does not bias direction, but it compresses every
ratio toward 1.0: the true steady-state decode speedup on the degenerate fixture
is 13.056/11.051 = **1.181x**, reported as 1.093x.

Two consequences worth stating plainly. A 64-token run is ~39% slower per token
than a 128-token run for reasons that have nothing to do with the code under test.
And any comparison that pairs a 64-token control against a 128-token candidate
manufactures a ~1.4x artifact.

## Correction 1: the "varied serial decode is 39% slower" claim was that artifact

It was inferred from a 64-token varied control against 128-token varied
candidates, and attributed to MoE expert-routing locality on varied tokens. Direct
test at matched length:

```
varied golden,     K=1, 64 tokens: 0.029889 s/token
repetitive golden, K=1, 64 tokens: 0.029784 s/token     -> 0.35% apart
```

The control's cost is prompt-INDEPENDENT, as it should be: at width 1 every round
declares exactly one row regardless of content, and the routed expert count per
token is fixed at 8. There is no routing-locality penalty to find.

## Correction 2: my "per-row cost is flat, so speculation buys nothing" was wrong

Per DECLARED row (declared = accepted + rejected + rounds, the L3 identity):
21.42 ms at K=1, 20.64 at K=3, 19.29 at K=4. There is a real ~10% economy of scale
at K=4, and it is exactly what produces the measured speedup on the degenerate
fixture. The correct model is not "batching is free" nor "batching is useless" but:

```
speedup ~= 21.42 / (declared_rows_per_emitted_token * ~19.6)
```

## What the track actually scores, on each kind of text

| text | draft acceptance | declared/emitted | K=4 vs K=1, matched length |
|---|---|---|---|
| degenerate (122 distinct/512, no near-ties) | ~100% | 1.016 | **1.093x** (1.181x steady-state) |
| varied prose/code (317 distinct/512) | 69% | 1.250 | **0.823x** (0.742x steady-state) |

The varied figure pairs the varied 128-token candidate against the 128-token
control, which the prompt-independence measurement above licenses. Break-even
needs `declared/emitted < 1.092`, i.e. draft acceptance above roughly 92%.

So the track produces a speedup on repetitive text and a ~1.2x SLOWDOWN on
realistic text, against a hard 1.0 floor. This is not a tuning problem: the
drafter is already accepting ~100% on easy text, so the ~10% per-row economy is
the entire ceiling, and every rejected row spends it. The hidden ranked prompts
are prose.

Recorded as measurement, not as a recommendation. Whether the track ships with the
floor removed, ships with a different scored quantity (e.g. excluding prefill from
the decode metric, which would restore ~1.18x/0.74x separation), or does not ship
on Laguna, is the organizer's decision.

# Amendment 12 (2026-07-30): the stall guardrail rejects essentially every run

`benchmark.dflash.json`'s scoring block specifies:

> `stall_guardrail`: a run whose max block latency exceeds 4x its p50 block
> latency is rejected as measurement-invalid with one gated retry

Every run ever recorded on M5-C was audited against that rule using the
`max_block_request_seconds` and `p50_block_request_seconds` the runs already
report. **25 of 28 runs violate it.** The three that pass are two single-token
diagnostics where max and p50 are the same measurement, and one benchmark that
happened to start with a warm on-disk kernel cache.

Representative figures at the ranked window (512-token seed, 128 decode steps):

| run | K | max block (s) | p50 block (s) | ratio | 4x rule |
|---|---|---|---|---|---|
| `r512-k1` (the paired denominator) | 1 | 0.19990 | 0.01303 | **15.34** | rejected |
| `r512-k3` | 3 | 0.16679 | 0.03128 | **5.33** | rejected |
| `r512-k4` | 4 | 0.22017 | 0.03155 | **6.98** | rejected |
| `r512-k6` | 6 | 0.77711 | 0.03360 | **23.13** | rejected |
| `variedb-bench-k4` (varied prompt) | 4 | 0.36574 | 0.03377 | **10.83** | rejected |

The serial control is the worst offender at 15x, and it is the side the score
divides by — so the guardrail would reject the denominator of every ranked
measurement, take its one gated retry, and reject that too.

## Cause, and why it is not the same problem as the prefill dilution

The offending sample is the FIRST block request, which pays first-touch
compilation for the decode-shaped graphs. For K=1 that is ~0.198 s against a
0.013 s steady-state round: roughly 0.185 s of one-time cost sitting inside the
timed window as a single outlier.

This is a DIFFERENT fixed cost from the ~1.07 s seed prefill measured in
Amendment 11. The prefill is charged into `decode_seconds` deliberately
(`prefill_component: "none; seed prefill is charged inside the decode
measurement"`) and, being equal on both sides, only compresses ratios toward 1.0.
The first-block spike is a single-sample outlier and is what the max/p50 rule
trips on. Fixing one does not fix the other.

Evidence the cause is first-touch rather than a real stall: `fix-b-bench` ran as
the second `mlxfast-swift` invocation of its session, with MLX's on-disk kernel
cache already warm from the first, and its ratio is 2.97 — inside the rule. Every
cold-start run is outside it.

## Fix, and its status

`LagunaDFlashBlockSession.warmAllBlockWidths()` was changed earlier today to warm
a seed past the sliding-window ring and EVERY legal block width including width 1,
from one shared helper used by both warm points. That change targets exactly this
spike, and it was made before this audit for an unrelated reason (an asymmetric
first-touch between the width-1 control and the width-K candidate biases the
ratio, not just its variance). It has not yet been measured against the guardrail.

The bar it has to clear is concrete: for the K=1 control, max block latency must
fall below 4 x 0.01303 = **0.052 s**, from 0.19990 today. If warming does not get
there, the remaining options are to move the first request outside the scored
window, or to change the guardrail from a max/p50 rule to one that tolerates a
single cold-start sample (for example p99/p50, or discarding the first round from
the latency statistic while still charging its time).

Recorded here because a guardrail that rejects 25 of 28 honest runs is not a
guardrail, and `official_scoring_enabled` cannot flip while it stands as written.

# Amendment 13 (2026-07-30): the warm move works, and a fourth blocker is structural

## The warm placement is measured and settled

Amendment 12 set a concrete bar: the K=1 control's max block latency had to fall
under 4 x 0.01303 = 0.052 s. Measured on M5-C at the ranked window:

| binary | K=1 total | K=1 s/token | max/p50 | 4x rule |
|---|---|---|---|---|
| `b8795c6` (before any warm change) | 2.743 s | 0.021429 | 15.2 | rejected |
| `068a940` (warm widened, INSIDE the window) | 3.409 s | 0.026629 | 1.10 | ok, but 24% slower |
| `598f642` (warm moved OUTSIDE the window) | **2.4135 s** | **0.018855** | **1.10** | **ok, and 12% faster** |

K=4 moves the same way: 2.507 s -> 3.198 s -> 2.1930 s. So the final state passes
the stall guardrail with a 3.6x margin AND beats the original absolute time,
because the first-touch spike is now genuinely outside the scored window instead
of merely relocated within it. The paired ratio on the repetitive fixture is
1.1005 (0.018855 / 0.017133), against 1.094 before — the warm move helps both
sides, as a shared cost should.

It also produced, for the first time, a varied-prompt K=1 control at the full
128-token window: **0.018889 s/token, against the repetitive fixture's 0.018855 —
0.18% apart.** That independently reconfirms Amendment 11's finding that the
control is prompt-independent, now with the corrected reference frame and a
matched token count. The performance verdict is unchanged: the warm move improves
both sides equally, so varied text stays at ~0.82x.

## Blocker 4: the scored path cannot tolerate ANY candidate divergence

With the reference's width-1 frame corrected (commit `068a940`), the varied-prompt
K=1 control passes 128 tokens cleanly — `residual_divergences=0`,
`admissible_exact_count=128`. But the width-4 block run on the same corrected
golden now fails at step 6, and the cause is not the fix.

Two mechanisms compound:

**4a. The pre-generated golden is teacher-forced on the REFERENCE's chain.** The
scored path builds its oracle as `DFlashGoldenReferenceOracle(golden:)` — rows
read from a file — and the ranked wrapper passes that file in
(`--golden PATH`, "hidden block-decode golden / oracle file", installed
into the workspace, hash-verified, scrubbed after). So the moment the candidate
legitimately diverges at one near-tie and the residual bucket admits it, every
later row is anchored to a prefix the candidate no longer has, and the run
cascades into `tokenNotAdmissible`. On the corrected varied golden, 108 of 128
chain tokens differ from the old one starting at index 5.

This contradicts the contract's own stated design: "the parent journals, and the
pinned reference worker replays afterwards", teacher-forced "on the candidate's
own emitted prefix". A pre-generated file cannot do that. The live reference path
(`dflash_reference_rows`, now stateful and correct) can, but the scored path never
issues a reference request.

**4b. Criterion E's two admitted frames do not cover the frame a block round
actually runs in.** At the failing row the reference's width-1 frame says 3997 and
its recorded width-2 declared frame ALSO says 3997, while the candidate's round —
a 2-row verify plus rollback at `accepted_draft_rate=0.167` — produces 4414, the
other member of a 2-ULP top-2 pair (margin 0.25).

The declared frame is computed as a clean width-w forward over the prefix. A real
block round is a width-w verify against a cache that has been continuously
advanced and rolled back. Those are different constructions, so they differ at
near-ties. Criterion E admits two frames; honest block decode runs in a third.

## What this means

Blocker 4 is a contract-design gap, not a bug to patch. Fixing 4a means moving
validation out of the timed loop into a post-run replay against the candidate's
journal — the design the contract already specifies. Fixing 4b means either
recording the declared frame in the construction a real round uses, or admitting
near-tie rows on numerical plausibility (top-2 membership AND a top-2 gap inside
the measured frame-divergence envelope, which the L2 calibration already puts near
3.375 logits) rather than on frame equality. Both are needed; neither is small.

Combined with Amendment 11's performance result, the honest summary is that this
track needs a contract redesign rather than a bug-fix pass — and that it would
still score ~0.82x on realistic text afterwards.

# Amendment 14 (2026-07-30): near-tie rows are admitted on plausibility (Blocker 4b)

Criterion E admitted an emitted token only if it matched the reference in one of
two frames, and charged anything else to a residual budget that Amendment 10
showed is exactly one slot at the frozen window. Amendment 13 (Blocker 4b) showed
why that fails honest code: at a row where the reference's OWN top-1 and top-2 sit
within build-to-build drift, "the reference says X" is not a fact about
correctness. Either token is what the reference itself would emit under a
differently-ordered accumulation. The candidate is being failed for a tie the
reference cannot break.

## The rule

A new outcome, `admissibleNearTie`, sits between the declared-frame admission and
the residual bucket. A token is admitted under it when BOTH hold:

1. it is one of the reference's top-2 tokens at that row, and
2. the reference's own top-1-minus-top-2 gap is within
   `experimentalDFlashNearTieLogitEnvelope`.

Such rows do NOT spend the residual budget. A top-2 divergence at a CONFIDENT row
is untouched: that is genuine candidate-vs-reference drift, it keeps spending the
deliberately small residual budget, and it can still exhaust it.

## Where the envelope comes from

Derived, not guessed — the discipline Amendment 8 said this contract kept failing
to apply. The L2 work-binding calibration (Amendment 6) measured a maximum
candidate-vs-reference logit delta of **3.375** across 352 long-context
comparisons. Reordering top-1 and top-2 requires the gap to fall below the
DIFFERENCE of two such per-logit drifts, which is bounded by `2 x 3.375 = 6.75`.
That is the envelope.

Measured density of rows inside it: **3 per 128 positions on varied prose, 0 per
128 on the degenerate self-continuation fixtures.** The second number is why the
one-slot budget survived every test until real text was finally run.

## Why it cannot be farmed

The gap belongs to the REFERENCE's logits, not the candidate's. A submission can
neither manufacture near-tie rows nor learn which positions are near-ties without
doing the reference's own work — and if it did that work, it has not cheapened
anything. The envelope also widens only WHICH of two already-plausible tokens is
allowed, never how many: a token outside the reference's top 2 is rejected at a
near-tie row exactly as anywhere else, which is pinned by
`nearTieRowsDoNotAdmitTokensOutsideTheReferenceTopTwo`.

A count cap of 40 per thousand tokens (6 slots at the frozen window, against the
measured need of 3) bounds the blast radius if some prompt turns out flatter than
anything measured. It is a backstop; the envelope test is the control.

## What this does NOT fix

Blocker 4a is untouched. The scored path's oracle is still a pre-generated golden
teacher-forced on the reference's chain, so the moment a near-tie divergence is
admitted — which this amendment makes MORE likely, by design — every later row is
anchored to a prefix the candidate no longer has. Admitting the divergence and
then failing three rows later is not progress on its own. 4a needs validation
moved out of the timed loop into a post-run replay against the candidate's
journal, which is what the contract specified in the first place.

## Note on the rounding artifact

`(tokens * rate + 999) / 1000` rounds up to one slot for any window under 25
tokens at rate 40. That is the same artifact Amendment 10 criticised in the
residual budget, and it is retained rather than special-cased: the ranked window
is 128 tokens, where the rate gives 6. It is recorded here so that a future short
diagnostic run that trips the cap is recognised as this, and not as a finding.

# Amendment 15 (2026-07-30): Blocker 4a is fixed, and it uncovers the next one

Validation is now split in two, which is what the contract said all along:
"the parent journals, and the pinned reference worker replays afterwards",
teacher-forced "on the candidate's own emitted prefix".

`LagunaDFlashBlockValidator.acceptStructural(round:)` keeps inline, in the timed
loop, everything that is arithmetic on the worker's own report: vocabulary range,
block bounds, emitted-vs-declared row accounting, the KV-offset ledger, and the
shape of the work-binding readout. It journals the round and scores nothing.
`validateJournalAgainstReference(oracle:)` runs after the timed loop, with the
candidate worker closed, and does everything that needs a reference:
sequential / declared-frame / near-tie / residual admissibility plus the L2 top-2
logit comparison. A report cannot be published without it
(`requireReferenceValidated()`).

The replay opens the reference worker only AFTER closing the candidate — two live
text towers would be 43 GB — prefills it over the same seed, and walks the journal
in emission order, which keeps the reference's continuous width-1 cache a pure
continuation. Measured: **zero** `rebuilding the width-1 frame` events across a
128-token width-4 run, i.e. the whole replay was one continuation.

`DFlashGoldenReferenceOracle` stays the fallback oracle and keeps its two
remaining scored jobs: the seed-token expectation and the `--tokens` bound.

## Measured on M5-C, `fix3-golden.json` (varied prose, 512-token seed)

| run | before (stored golden) | after (live replay) |
| --- | --- | --- |
| K=1 probe, 128 tokens | pass, 128 exact | pass, 128 exact |
| K=4 `--schedule-seed 7`, 128 tokens | **`tokenNotAdmissible` at step 6** | **`tokenNotAdmissible` at step 109** |
| K=4, 109 tokens | — | pass: 108 exact + 1 near-tie, 0 residual |

The one near-tie is at row 5 (reference top-2 gap 0.25, logits 15.3125 /
15.0625) — the exact divergence that used to poison every later golden row. The
replay absorbs it and then scores 103 more rows exactly, which is what 4a was
costing.

Timing is unchanged, against the expectation that removing in-loop oracle work
would lower it. Three runs each on `seam-512-golden.json`, 128 tokens:

| | before | after | delta |
| --- | --- | --- | --- |
| K=1 `parent_measured_seconds_per_token` | 0.018887 | 0.018920 | +0.17% |
| K=4 `parent_measured_seconds_per_token` | 0.017188 | 0.017167 | -0.13% |

Both deltas are inside the run-to-run spread of either set. The in-loop oracle was
an array index plus a handful of double comparisons per token — microseconds
against a 17-19 ms token. Wall-clock per invocation roughly doubles (10s -> 20s)
because a second model load now happens, outside the clock.

## The L2 tolerance was calibrated against an artifact

`max_top2_logit_delta` on the same fixtures, stored golden vs live replay:
K=1 **2.625 -> 0**, K=4 **3.375 -> 1.75**. The K=1 control is now BIT-IDENTICAL to
the reference's width-1 walk. So a large part of the 3.375 that Amendment 6 used
to derive both the 5.0 work-binding tolerance and the 6.75 near-tie envelope was
golden pre-generation drift, not candidate-vs-reference drift. Both constants are
now looser than the evidence requires and should be recalibrated against live
replay before `official_scoring_enabled` is considered.

## Blocker 4c: a two-token admissible set cannot express a three-way tie

The width-4 run still fails, deterministically (bit-identical over three runs),
and NOT for 4a's reason. At row 109, with the reference teacher-forced on the
candidate's own prefix:

```
cand=1972  k1=268  declared frame (width 4)=268
reference top-2 = [268, 85]    logits [14.625, 13.875]
candidate top-2 = [1972, 85]   logits [14.5,   13.9375]
```

Three tokens sit inside 0.75 logits at a row where the reference's top-1 is only
14.625 — some 7 logits below this model's typical confident top-1 of 21-26. The
row IS a near-tie by Amendment 14's own predicate (gap 0.75 << 6.75 envelope).
It is rejected only because the admissible set is built from the reference's
top-**2**, and the candidate's argmax is the reference's #3. The shared slot
(token 85) agrees to 1 ULP in both value and rank, so the L2 binding says the
candidate did the work.

This was masked by 4a: the run never used to reach row 109.

Fixing it means widening what the reference REPORTS (top-k, k>2, or every token
within the envelope of the reference's top-1) so the near-tie predicate can be
applied to the set it was always meant to describe. That is a contract amendment
with its own anti-farming argument to make, not a validator tweak, and it is
deliberately NOT done here: widening an admissible set to make a run pass is
exactly the move this document exists to prevent someone making quietly.
