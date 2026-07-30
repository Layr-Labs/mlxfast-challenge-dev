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
2. **A multi-token, seam-crossing timed run.** The passing e2e above is a
   1-token run, so its timings (0.33 vs 0.18 s/token) are fixed-cost dominated
   and meaningless, and it does not cross the ring boundary.
3. **`DFlashWorkBindingTolerance` calibration** (Amendment 1) from measured
   honest candidate-vs-reference logit deltas.
4. **L4 ring-index consistency** (Amendment 4) and **L5 reference-drafter
   replay** are designed but unimplemented.
5. The box wrapper `/opt/bench-runner/measure-dflash-job.sh` still passes the
   nonexistent `--contract` and `--require-trained-drafter` flags, omits
   `--drafter` on the probe side, and asserts
   `reference_checked_row_total == declared_rows_total` where the reference can
   only score EMITTED rows (it should be `>= emitted_token_total`).

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
4. **Decide the block size from long-context data, not the short sweep.** K=8
   was chosen because it peaked at 1.86x on short prompts. At low acceptance a
   smaller K wastes fewer verify rows, so the ranked default may well be lower.

Until (3) is isolated and a long-context configuration clears 1.0 with margin,
`official_scoring_enabled` must stay false regardless of how complete the harness
is. The harness being correct and the track being viable are separate questions,
and this amendment records that the second one is currently unanswered.

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
