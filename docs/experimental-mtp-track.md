# Gemma 4 31B-IT MTP Track

## Status and default routing

This is the default official ranked track, with track ID
`gemma4-31b-it-mtp-v1`. Yukon reads `benchmark.json`, dispatches
`.github/workflows/benchmark.yml`, and ingests `score.json`. Official scoring
is decode-only paired speedup against the pinned serial K=1 target baseline.
The former base-model serial challenge is archived as
`benchmark.serial.json` and `.github/workflows/serial-benchmark.yml`.

The two experimental commands are:

- `mtp-probe`: PR #424's serial target-only block control.
- `mtp-benchmark`: a real trained-assistant MTP path that refuses to run
  without `--require-trained-assistant`.

The `mtp-*` commands remain protocol-explicit; no environment variable can
turn the archived serial `benchmark` command into MTP.

## Compatibility conclusion

The current challenge target is:

- Runtime checkpoint:
  `mlx-community/gemma-4-31b-4bit@e236b3eb2f9567ded5875cfa89f1666afa1acbf1`
- Declared upstream base:
  `google/gemma-4-31B`
- Checked-in byte manifest:
  `fixtures/reference_gemma_4_31b_4bit.sha256`

Google's public MTP drafter family is explicitly paired by its model cards
with the instruction-tuned target, not the base target:

- Target:
  `google/gemma-4-31B-it@518276fb130dc81caf9a4f772e65e63ef2526493`
- Assistant (organizer-pinned QAT 4-bit conversion):
  `mlx-community/gemma-4-31B-it-qat-assistant-4bit@5234fd588403c9b68f3bd20a140b7e61700cb7e2`
  (affine 4-bit, group size 64; converted from Google's QAT drafter
  `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`, which the model
  card pairs with `google/gemma-4-31B-it`)

No matched public 31B base assistant was found. Architecture compatibility
alone does not prove training compatibility, so this prototype does not bind
the IT assistant to the base challenge checkpoint. It defines a separate
Gemma 4 31B-IT MTP track instead.

The MLX target conversion is pinned independently:

- `mlx-community/gemma-4-31b-it-4bit@696d436c404745a59f30e4939a658162b0a9e57f`
- Declared base model: `google/gemma-4-31B-it`

The target model cards declare Apache-2.0 and link the Gemma license at
<https://ai.google.dev/gemma/docs/gemma_4_license>; the QAT assistant
conversion's card declares the Gemma license and derives from Google's
Apache-2.0 QAT drafter. The live ranked track remains subject to those model
license terms.

## Provenance contract

`fixtures/gemma_4_31b_it_mtp_track.json` binds:

- track identity and enabled official-scoring state;
- target and assistant model IDs plus immutable repository revisions;
- the exact pinned `mlx-swift-lm` revision
  `bc1c0ee67d15798343be17c9f8f61f7c0d977149`;
- frozen Gemma 4 target and assistant architecture fields;
- SHA256 manifest identities and artifact byte budgets;
- block size four, a compatibility default of 128 parent-counted decode
  tokens, and a hard trusted-parent cap of 512;
- the established paired-reference baseline and publication status.

Byte manifests:

- `fixtures/mtp_gemma_4_31b_it_4bit.sha256`
- `fixtures/mtp_gemma_4_31b_it_assistant_qat4bit.sha256`

The assistant runtime inventory is exactly two regular, single-link files:

- `config.json`: 2,962 bytes
- `model.safetensors`: 264,141,359 bytes

The combined assistant directory is exactly 264,144,321 bytes. Extra files,
symlinks, hardlinks, size mismatches, hash mismatches, incompatible config
fields (including the pinned affine 4-bit group-64 quantization block), or a
total above 300,000,000 bytes fail before model load.

The target source manifest totals 18,444,420,181 bytes. The transformed
text-only target is capped at 20 GiB. The assistant remains an
organizer-provisioned read-only sidecar and is not copied into participant
submission artifacts.

`setup-mtp.sh` is the only MTP provisioner. It uses separate target and
assistant cache directories, immutable revision URLs, resumable `.partial`
downloads, exact size/SHA256 verification, and strict flat inventories. It
never changes `setup.sh`'s base checkpoint.

`mtp-benchmark` also requires `--target-source` and revalidates that complete
18.4 GB source inventory in the trusted parent. This prevents an accidental
base-target/IT-assistant pairing. In an official pipeline, the trusted
transform step must consume that same validated source; returned-token parity
then binds the transformed runtime to the IT oracle.

No weights are committed to this repository.

## Pinned MLX APIs and model integration

The Package.resolved revision exposes:

- `Gemma4MTPTarget`
- `Gemma4MTPForward`
- `Gemma4SharedKV`
- `Gemma4AssistantConfiguration`
- `Gemma4AssistantDraftModel.load(from:)`
- assistant `bind(target:)`
- target `forwardForMTP`
- target `rollbackSpeculativeCache`
- `Gemma4MTPTokenIterator` and `runGemma4MTPRounds`

`Gemma4RuntimeModel` conforms to `Gemma4MTPTarget`. Its MTP-only forward uses
the pinned library trunk's pre-norm/shared-KV capture hook. The ordinary
serial entry point is unchanged. The fast engine has a separate exact-pair
entry point that is unreachable from serial `benchmark`.

`Gemma4TrainedMTPBlockSession` persists, for one request:

- the target KV caches;
- the last committed target bonus token;
- the target pre-norm hidden state;
- the last full-attention and sliding-attention shared K/V snapshots;
- a host mirror of the target cache offset.

Target verification is selected explicitly:

- `--target-verification exact-pair` is the default trained-MTP path.
- `--target-verification serial` is the K=1 control.

An exact-pair session fails during construction if the target does not have
all required packed metadata and kernels. It never catches a numerical
mismatch and silently reruns that pair serially.

In this track, `exact` means exact target argmax/returned-token verification
and logical protocol/report consistency at the ranked boundary.
It does not require serial K=1 accumulation order or bit-identical logits,
hidden state, or KV tensors. Reassociation, fusion, and alternate reduction
strategies are allowed when intermediates are finite and stay inside the
regression suite's separately named logits, hidden-state, and KV numeric
envelopes. Tensor shapes, relevant dtypes, cache offsets and lengths, state
counts, commit/rollback, physical row accounting, and target token decisions
remain exact implementation contracts.

The model-backed suite is a trusted/local/upstream development regression. It
uses public or operator-controlled non-hidden inputs, casts both sides to
float32, and applies `allClose` with these provisional heuristic envelopes:

- logits: `rtol=1e-5`, `atol=1e-5`, plus exact argmax token IDs;
- pre-norm hidden state: `rtol=1e-2`, `atol=2e-3`;
- physical/shared KV values: `rtol=1e-2`, `atol=1e-3`;
- attention activations: `rtol=1e-2`, `atol=2e-3`.

MLX reports unit-scale epsilon near `9.77e-4` for Float16 and `7.8125e-3`
for BF16. Existing repository tests use `rtol=1e-5`, `atol=1e-5` for Gemma 4
logits and `rtol=1e-2`, `atol=2e-3` for BF16 attention parity. The KV
near-zero absolute tolerance is a heuristic intended to increase sensitivity;
it cannot guarantee every stale or shifted value fails. All values above are
provisional regression heuristics informed by those tests and quantization
behavior, not representation-derived bounds. An optimization must calibrate
on M5 before relying on or loosening them.

These tensor checks are not ranked submission gates. Candidate-linked test
code never receives hidden correctness or benchmark oracle paths. Ranked
enforcement combines parent-owned exact returned token IDs and logical
protocol/report-consistency validation (`all_tokens_matched` and
`parity_all_ok`) with track-aware static review and hidden behavioral checks.
Cache offsets, physical rollback/geometry, and numerical state are checked
inside the candidate worker/model plus trusted implementation regressions and
manual/operator validation, not independently observed by the parent.

For configured block size `K` in `2...4`, one round:

1. Uses the trained assistant to draft `K - 1` tokens from the current target
   token, target hidden state, and shared K/V.
2. For K=4, the direct `exactMTPFour` forward is attempted only when exact-four
   is enabled, the adaptive draft-margin selector approves it (or adaptation is
   disabled), and every layer/cache reports four-row eligibility at the current
   geometry. If the margin selector declines or four-row engine/cache geometry
   is unavailable, verification falls back to pair composition; deterministic
   rotating-cache wrap geometry may then force the existing serial tail. With
   the pinned 1,024-token window, four rows fit through starting offset 1,020,
   one pair fits through 1,022, F1 applies at 1,021–1,022 after one accepted
   pair, and F0 applies from 1,023 onward (including post-wrap). There is no
   ranked non-geometry serial fallback: engine ineligibility fails closed. K=3
   uses one pair plus an exact K=1 bonus tail; K=2 uses one pair; K=1 is
   target-only. Diagnostics count one direct four-row forward as two
   pair-equivalent segments.
3. Compares each draft to the corresponding target argmax in order and starts
   no later segment after the first rejection.
4. Emits only the target-confirmed prefix plus the target token at the first
   rejection (between one and `K` tokens).
5. If row zero of a pair rejects, removes row one's physical cache position
   from every layer and slices the shared-KV view before returning. If row one
   rejects, row zero's accepted input remains committed.
6. Persists hidden/shared-KV state at the committed target position.
7. Checks every target cache offset against the host mirror.

The current exact-pair kernels share packed weight traversal across sliding
Q/K/V, full Q/K, attention output, gate/up activation, down projection, layer
boundaries, and the tied packed13 vocabulary head. RMS normalization and full
D=512 attention currently remain row-serial. Sliding D=256 attention uses one
causal two-query dispatch only below the library's 1,024-key shape switch;
otherwise it is row-serial. Those are implementation choices, not mandated
reduction boundaries. Any replacement must keep causality—row two sees row
one's committed K/V—and pass the behavior and local numeric regressions above.

`exactMTPFour` is not implemented entirely as two exact-pair calls: it uses
dedicated four-row QKV, output, gate/up, down, and vocabulary-head kernels
(with some current attention/boundary fallbacks split into two-row pieces).
The model-backed regression therefore compares its four logits, pre-norm
hidden rows, shared K/V, every physical cache state, and exact metadata against
four serial rows directly. Pair-level envelopes cover pair segments; forced
K=4 session tests separately retain exact token, rollback, and physical-row
accounting.

The expanded M5 pilot proved why an ordinary mathematical K-row target
forward is insufficient: retained layer-1 K/V diverged after the first
full-acceptance block and a public prose continuation flipped argmax at decode
step 48, despite identical logical offsets. The exact-pair path is not that
ordinary batched path. Its regression gate compares both pair rows against two
K=1 forwards using float32 numeric envelopes for logits, pre-norm hidden
state, and every layer's physical K/V values while keeping token decisions,
tensor geometry, cache offsets/state counts, and rollback state exact at
multiple offsets.

Zero, partial, and full acceptance use the same charged path. A deterministic
serial tail is used only when a block shape requires one row or a rotating
cache reaches geometry where a prefix-preserving pair cannot be formed. These
rows are counted in `serial_verification_row_count`; engine ineligibility
fails closed rather than becoming a timed fallback.

Any exception after a block begins poisons the worker session. The worker
cannot continue from ambiguous cache state.

## Trusted block protocol

The trained worker accepts only:

- `mtp_decode_begin`: the target seed prompt.
- `mtp_decode_block`: request ID, last parent-committed token, and maximum block
  size.

Block requests contain no expected token, future oracle token, accepted count,
worker duration, score, denominator, prompt hash, or continuation history.
Request IDs must be monotonic. Block responses carry a nonce and a nonempty
token block only. The separate post-phase diagnostic response can report
memory, selected verification mode, exact-pair segments, pair rollbacks, and
serial geometry/tail rows; none has timing or score authority. Unknown fields
such as `accepted_count`, `token_count`, `seconds`, or `future_tokens` are
rejected by the parent decoder.

The parent preallocates a private `(starting logical target offset, requested
max block size, returned token count)` history before timing and appends one
entry per accepted block. After the elapsed time is captured, it validates that
the final `(pair segments, rollback rows, serial rows)` diagnostic tuple is
exactly reachable from that history under the geometry-aware
pair/direct-four/serial-fallback transition table. This adds no timed wire
fields. The preallocated append is bounded trusted bookkeeping inside the
parent wall-time measurement, so its small optimized overhead is charged to
MTP; only the final reachability DP runs after elapsed time is captured.

Raw gate reports, worker diagnostics, sealed paired results, private logs, and
free-form measure verdicts can carry hidden-prompt-influenced data. Gate
stdout/logs are removed by the gate-step trap and verified absent by the
pre-timing scrub. Ranked sealed results/verdicts are deleted immediately after
the minimized score is constructed and schema-validated, before staging; the
always-run cleanup remains defense in depth. A correctness-only run publishes
only a canonically reconstructed fixed-schema trusted verdict containing
schema version, track ID, a passed boolean, and the fixed token count. A ranked
run publishes only a canonically reconstructed minimized Yukon score payload.
The score and its
parent-measured timing basis remain an unavoidable low-bandwidth feedback
channel; artifact minimization removes avoidable candidate-controlled fields
but does not claim to eliminate that timing/score side channel.

The trusted parent:

1. Validates target/assistant artifacts without invoking editable model code.
2. Starts its timer before the prompt-bearing begin request.
3. Checks the target-produced seed token.
4. Sends only the last committed token on each block request.
5. Rejects missing, empty, oversized, out-of-vocabulary, overrun, or
   mismatching blocks.
6. Checks every returned token against an independent serial target oracle.
7. Stops only after the explicit trusted total (default 128, maximum 512).
8. Divides its own wall time by that parent-owned total.

All seed prefill, drafting, target verification, rejection, rollback, IPC, and
final-tail work is charged. Worker-reported timing and acceptance have no
authority.

The MTP worker receives target, assistant, and contract paths as explicit
arguments. Its environment uses the existing allowlist sanitizer. Its Seatbelt
profile denies network, process creation, and all writes except `/dev/null`.
It validates all artifacts before loading and revalidates them after loading,
closing the practical replace-during-load window. On the official runner,
operator-owned read-only cache permissions remain required.

## Static-review policy

`run-submission-static-review.sh` defaults to `serial`, preserving the current
ban on all speculation. Trusted rollout code may set:

```text
MLXFAST_SUBMISSION_TRACK_ID=gemma4-31b-it-mtp-v1
```

The MTP policy permits only organizer-assistant, target-verified block
speculation and its within-request commit/rollback state. It still rejects:

- prompt, suffix, n-gram, history, or prompt-hash lookup;
- known fixed-token-total or call-count specialization;
- participant-provided, replaced, tampered, or extra assistants;
- unverified drafter tokens or skipped target verification;
- fake timing, token counts, acceptance, artifact data, or scores;
- stale target logits;
- logical or physical KV state beyond the committed prefix;
- hidden future-token/logit/KV buffering outside the protocol;
- prompt-dependent warmup or precompute before the trusted timer.

Static review is a gate, not a proof. The trusted parent's independent oracle
and fixed timing/count contract remain the substantive runtime controls.

## Memory and artifact budget

Known on-disk bytes:

- target source: 18,444,420,181 bytes;
- assistant sidecar: 264,144,321 bytes;
- combined: 18,708,564,502 bytes (about 17.4 GiB).

The assistant is the ~470M-parameter drafter stored as affine 4-bit
(group size 64) packed U32 weights with BF16 scales/biases according to its
repository metadata. The M5 matrix measured up to 47.6 GiB peak process RSS,
31.2 GiB MLX active memory, 3.4 GiB MLX cache memory, and 34.1 GiB MLX allocator
peak. Runtime peak is higher than file size because target and assistant
weights, target KV, shared K/V, verification logits/activations, and allocator
state coexist. This fits the 128 GiB runner but exceeds the base track's
practical 36 GiB local budget; a public track needs its own documented memory
minimum and host-enforced cap.

## Participant surface and track design

The MTP track is designed so participants can make block decode faster
without being able to weaken the correctness or measurement guarantees.

### Trusted surface (operator-owned, not submittable)

These components define the measurement and cannot be changed by a
submission; they live outside `benchmark.json` `editablePaths`:

- `Sources/MLXFastHarness/GemmaRuntimeMTP.swift`: the trusted parent. It
  owns the timer (started before the prompt-bearing begin request), the
  serial-oracle validation of every returned token, the configured decode
  total used as the fixed denominator, block-size bounds, and the bounded
  diagnostics accounting.
- `Sources/MLXFastHarness/GemmaRuntimeMTPWorker.swift`: the sandboxed worker
  protocol shell. It validates artifacts before and after model load,
  enforces the strict `mtp_decode_begin`/`mtp_decode_block` request schema,
  monotonic request IDs, in-vocabulary bounded blocks, the cache-offset
  ledger, and session poisoning after any failure.
- `Sources/MLXFastHarness/GemmaRuntimeMTPProvenance.swift`,
  `fixtures/gemma_4_31b_it_mtp_track.json`, and the SHA256 manifests: the
  pinned target/assistant identity and architecture contract.
- `Sources/MLXFastCLI/main.swift`: track dispatch and the worker Seatbelt
  sandbox profile.
- `setup-mtp.sh` and the organizer cache: artifact provisioning.
- `.github/scripts/run-submission-static-review.sh`: the MTP-track
  static-review policy.

### Editable surface (what participants optimize)

Submissions change only `Sources/MLXFastModel/` (and
`Sources/MLXFastTransform/`), which contains the whole speculative strategy:

- `Gemma4MTPRuntime.swift`: `Gemma4TrainedMTPBlockSession` — the drafting
  loop, verification composition (exact pairs, serial tails), acceptance
  handling, commit/rollback, and per-request state reuse.
- `Gemma4ExactTwo*.swift` and the fast-engine pair path: behavior-exact,
  numerically bounded multi-row verification kernels. Participants may
  reassociate or fuse reductions, improve dispatch, extend coverage (for
  example the direct four-row path), or replace the strategy entirely.
- Everything the serial track already allows: quantized matmul dispatch,
  attention restructuring, KV-cache handling, weight layout, scheduling.

Improvement directions intentionally left open: higher drafter acceptance,
fewer target dispatches per committed token, reducing the one-time
seed-prefill first-touch cost, cheaper drafter execution, and deeper kernel
fusion — all subject to the same oracle. (The first-block stall itself is
already fixed; see "First-block stall: root cause and fix".)

### Why participants cannot cheat the measurement

- The parent validates every returned token against a serial oracle golden
  generated from the pinned reference; one divergent token fails the run.
  Exact returned-token parity is therefore a hard gate, not a convention: a
  submission that "wins" by degrading output cannot score. The legacy
  `bitExactTokenGate`/`bit_exact_gate` schema names apply only to returned
  token IDs, not floating-point intermediates.
- The parent owns wall time and divides by its own configured decode total.
  Worker-reported timing, acceptance, and counters have no score authority,
  are bounded before use, and must satisfy the physical row-accounting
  equation.
- Blocks are capped at four positions; oversized, empty, out-of-vocabulary,
  or overrun blocks are rejected; request IDs are monotonic; the worker sees
  no future oracle tokens (requests carry only the last committed token).
- The worker process receives the prompt only inside the timed window, runs
  sandboxed (no network, no writes, no process spawning), and artifact
  hashes are revalidated after model load, so assistant substitution and
  pre-timer prompt work fail closed.
- What the parent cannot prove (internal use of the assistant, real target
  verification, physical rollback correctness when outputs still match) is
  covered by the MTP static-review policy, hidden prompt-independent tests,
  and the cache-offset ledger. Trusted/local/upstream tensor regressions add
  development coverage without exposing hidden oracle paths to candidate code.

### Adopted scoring

The contract's `proposed_scoring` block records the adopted ranked form:

```text
mtp_decode_speedup = paired_serial_decode_sec_per_token / mtp_decode_sec_per_token
score = mtp_decode_speedup          (decode-only; floor >= 1.0)
```

- The denominator authority is the trusted parent: wall time divided by the
  parent-configured decode total.
- The paired serial reference is the trusted serial K=1 target decode
  (the `mtp-probe` path) over the same golden, measured in the same session
  behind the same thermal gate on the same box — the same pairing discipline
  the serial track uses, so host drift cancels.
- Decode-only: the MTP protocol charges seed prefill inside the decode
  measurement, so there is no separately scored prefill component.
- The floor is 1.0: an MTP submission that is not actually faster than
  serial decode does not rank. The unmodified reference implementation
  measures about 1.2-1.3x on the M5, so the track starts with visible
  headroom rather than a hard-to-move 1.0.
- Scores publish only from at least three accepted thermal-gated pairs per
  session with alternating candidate/reference order; single-pair scores
  are not publishable because a bounded one-time seed-prefill first-touch
  (~1.5s run-to-run) still needs multi-pair averaging (see "First-block
  stall: root cause and fix").
- The leaderboard namespace is `gemma4-31b-it-mtp-v1`, fully separate from
  the archived serial configuration; `official_scoring_enabled` is true
  after the hidden IT-target goldens and M5 floors were frozen.

## Scoring and rebaseline contract

### M5 parity incident and fix

The expanded public/synthetic pilot found a deterministic prose mismatch at
decode step 48. Safe differential replay established:

- the failing block was round 13 at target offset 560;
- it rejected at its first draft position;
- every logical target cache offset was 561 after that position;
- an independent serial shadow cache had the same offsets;
- real and serial physical K/V first diverged at layer 1 immediately after the
  first K-row full-acceptance block (global decode step 4);
- fast-vs-library target and combined-vs-split cache variants all failed at the
  same step.

The root cause was shape-dependent floating-point target execution, not prompt
lookup, assistant incompatibility, host offset arithmetic, or a missing trim.
An ordinary batched K-row target graph was mathematically causal but differed
enough from K serial target graphs to flip a returned token. The first fix
verified supplied target rows with the serial K=1 graph. The current default
uses dedicated exact-pair kernels that share packed weight traversal; the
serial K=1 verifier remains the explicit control mode. Trusted/local
model-backed regressions preserve the failing seam with exact target token
decisions and exact cache geometry plus finite float32 numeric envelopes for
logits, pre-norm hidden state, and every layer's physical K/V values. They use
non-hidden inputs and cover multiple offsets, forced zero/partial/full and
second-segment acceptance, and deep growth/wrap boundaries without mandating a
reduction order.

### Prior K=1-control public/synthetic matrix

The corrected serial-equivalent verifier completed 12 thermal-gated M5 pairs:
three each for copy (N=255), prose (N=256), code (N=257), and reasoning
(N=256). All target tokens matched. The matrix exercised full, partial, and
zero draft acceptance plus one-token tails. Mean paired candidate/serial
speedups were 0.741x copy, 0.798x prose, 0.860x code, and 0.883x reasoning;
after the complete fixed K=1 warmup, a fresh copy triplet measured 0.860x.
That K=1 verifier remains the explicit correctness control; these numbers do
not describe the exact-pair implementation.

Per-category paired CV was initially 7.8% copy, 1.9% prose, 4.0% code, and
1.1% reasoning. Phase diagnostics showed copy's variation came from seed and
first-block onset (up to 1.77s) while median block latency and thermal/frequency
telemetry stayed stable. The fixed BOS warmup had not guaranteed all four exact
target rows were compiled when the assistant rejected early. Explicitly warming
four deterministic K=1 target rows reduced a fresh three-pair copy run to 3.2%
paired CV without retaining allocator buffers or using prompt-specific work.
Across accepted runs, maximum temperature was 56.1C and minimum steady GPU
frequency was 1604 MHz.

### Exact-pair public/synthetic matrices

The exact-pair verifier ran the same balanced thermal-gated protocol
(paired `mtp-probe` serial control versus `mtp-benchmark`, K=4, alternating
order, 512-token seeds, N=255/256/257) with every token matching the serial
oracle and every cache-offset invariant passing:

- First matrix, 12 pairs (3 per category): mean paired speedup 1.34x copy,
  1.27x prose, 1.33x code, 1.26x reasoning; overall mean 1.30x,
  ratio-of-means 1.28x.
- Extended matrix, 20 pairs (5 per category) after acceptance-independent
  pair warmup: mean 1.32x copy, 1.37x prose, 1.33x code, 1.30x reasoning;
  overall mean 1.33x, ratio-of-means 1.31x; slowest single pair 1.04x.
- Serial-control seconds per token were unchanged from the pre-exact-pair
  baseline (about 0.03425 s/token), so the generalized boundary kernels did
  not regress the K=1 path.

Every category mean and median clears 1.10x by a wide margin. The remaining
honesty gap is run-to-run spread: per-category paired CV ranged 2.4-13.1%
(first matrix) and 6.3-14.0% (extended matrix). The spread comes from
sporadic single-block stalls up to ~1.7s on the MTP side while p50 block
latency stays at ~75ms and telemetry stays clean (max loaded temperature
58.6C, minimum steady frequency 1612 MHz, peak RSS 47.5 GiB); no outliers
were deleted. The stalls are not explained by kernel compilation (fixed
warmup precedes the timer), thermal throttling, or frequency drops, and they
appear in a minority of fresh processes. Until their source is isolated, the
paired-CV<=5% pilot criterion is met only per-run-set, not universally, and
scores from single pairs must not be quoted. (This "sporadic ~1.7s stall" was
later root-caused as a deterministic block-0 first-touch after the trusted
allocator clear and fixed; see "First-block stall: root cause and fix" below.
The residual variance is a bounded one-time seed-prefill first-touch.)

### Committed-tip revalidation matrix (commit e71f2a9)

The consolidated committed tip (separate MTP-only pair boundary kernels,
serial K=1 boundary kernels restored unchanged from the base; audit
counter hardening) was rerun on the freed dev box (`m5-max-128gb-1`, drained
from ranked serving) as a fresh 12-pair thermal-gated matrix, K=4,
alternating order, 512-token seeds, N=255/256/257:

- All four trusted model-backed regressions passed against the pinned IT target
  and assistant on public/synthetic inputs (artifact validation, basic pair
  numeric/behavior parity, forced zero/partial/full/second-segment acceptance
  seams, deep growth/wrap). They were operator development checks, not
  candidate-linked ranked tests with hidden paths.
- Every returned token matched the serial oracle across all 24 runs
  (`parity_all_ok=true`), and every cache-offset invariant held.
- Mean paired decode speedup: 1.29x copy, 1.22x prose, 1.23x code,
  1.23x reasoning; overall mean 1.24x, median 1.29x, slowest single pair
  1.03x, fastest 1.42x.
- Serial-control seconds per token stayed at ~0.03425 s/token, confirming the
  boundary-kernel split did not regress the K=1 path.
- Telemetry clean: max loaded temperature 56.1C, minimum steady frequency
  1612 MHz, peak process RSS 47.5 GiB.

### First-block stall: root cause and fix

The "sporadic ~1.7s single-block stall" was root-caused on the dev box and
fixed. Per-block parent-side tracing showed it was not sporadic: it is
deterministic on **block 0**, the first timed decode block, with a magnitude
that varies 0.6-1.6s run to run (which is why per-run `max_block` sometimes
looked like a mild 0.15-0.6s and sometimes 1.6s). Every later block is a flat
~75ms.

Localization:

- An in-process driver (real drafter, real exact-pair verify, identical
  warmup, but no IPC and no worker sandbox) never stalled: 396 blocks across
  6 fresh processes held max 130ms once and ~77ms otherwise, flat allocator
  memory. That ruled out block compute, the allocator growth path, kernel
  JIT, and shape warmup coverage.
- The real parent+worker path stalled only on block 0 (569-1592ms across
  5 runs). Toggling the worker Seatbelt sandbox on/off made no difference,
  ruling out the sandbox.
- The one thing the worker does that the in-process driver did not is
  `resetRuntimeWorkerAllocatorForPhaseStart()` at `mtp_decode_begin`, the
  trusted anti-subsidy control that `clearCache()`es every free buffer the
  untimed warmup allocated. The worker-internal split confirmed block 0's
  cost is in the exact-pair `verify` `eval` (515-888ms) with host-side
  bookkeeping ~0ms and the allocator free-cache dropping right before it: the
  first timed exact-pair forward pays a one-time Metal buffer first-touch for
  the whole 60-layer working set the clear had freed.

Fix (`Gemma4TrainedMTPBlockSession.warmWorkingSetAfterAllocatorReset`, called
by the worker after the trusted clear and before `begin`): re-touch the
drafter and exact-pair working set on throwaway BOS state so those buffers are
resident in the free pool before the first real block. It is charged (inside
the timed window), input-independent (runs before any seed is applied, on
throwaway caches, never touching the real session cache/hidden/shared-KV or
any committed token), and only warms the allocator — the pinned pipelines were
already compiled during untimed init. The trusted `clearCache()` is unchanged;
the serial path and the #586 boundary are untouched. It skips the redundant
512-token prefill warm because `begin`'s charged seed prefill reallocates that
same working set immediately after.

Result on the dev box (real `mtp-benchmark`, sandbox on): block 0 drops from
0.6-1.6s to ~74-79ms, `max_block` from ~1.6s to <=79ms, every block uniform,
token parity preserved (`all_tokens_matched=true`) in all runs, and mean
decode spt improved slightly (the redundant prefill was removed). The one-time
first-touch cost is not eliminated — it cannot be while the trusted clear
stands — but it is relocated into the already-charged seed-prefill phase and
made bounded, so the per-block decode distribution is now trustworthy for
single-pass MTP timing. A residual run-to-run variation of ~1.5s in that
one-time seed-prefill first-touch remains (a genuine Metal residency effect,
not thermal/frequency, and present with or without inter-run cooldown); the
multi-pair alternating-order protocol below averages it out, so single-pair
totals still must not be quoted from one pair.

The publication checklist is complete. The live default workflow is
`.github/workflows/benchmark.yml`; the contract has
`official_scoring_enabled=true`, the paired serial baseline and 1.0 floor are
established, and hidden pins are populated. Ranked runs publish the adopted
ratio-of-means score only after parent-owned token/protocol gates and the
multi-pair thermal/telemetry checks pass. Historical enablement procedure is
retained in `docs/mtp-track-golive-runbook.md`.

The archived serial configuration remains separate from MTP results.

## Local/operator workflow

Provisioning (large download; do not run merely to compile the prototype):

```bash
MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh
./setup-mtp.sh
eval "$(./setup-mtp.sh --print-paths)"

.build/release/mlxfast-swift transform \
  --reference "${MLXFAST_MTP_TARGET_DIR}" \
  --output mtp-weights
```

Generate a target-matched public oracle on the M5 from a prompt of at least
512 tokens:

```bash
.build/release/mlxfast-swift generate-golden \
  --prompt-file correctness_prompts/public_longcopy_gate_english.txt \
  --weights mtp-weights \
  --tokenizer "${MLXFAST_MTP_TARGET_DIR}" \
  --output /tmp/gemma4-31b-it-mtp-public.json \
  --name gemma4-31b-it-mtp-public \
  --steps 129
```

Run the serial block control first:

```bash
.build/release/mlxfast-swift mtp-probe \
  --weights mtp-weights \
  --golden /tmp/gemma4-31b-it-mtp-public.json \
  --block-size 4 \
  --tokens 128
```

Then run the trained-assistant prototype:

```bash
.build/release/mlxfast-swift mtp-benchmark \
  --target-source "${MLXFAST_MTP_TARGET_DIR}" \
  --weights mtp-weights \
  --assistant "${MLXFAST_MTP_ASSISTANT_DIR}" \
  --contract fixtures/gemma_4_31b_it_mtp_track.json \
  --golden /tmp/gemma4-31b-it-mtp-public.json \
  --block-size 4 \
  --tokens 128 \
  --target-verification exact-pair \
  --require-trained-assistant
```

`--tokens` defaults to 128 and accepts `1...512`; the selected golden must
contain one seed token plus the requested number of decode tokens. The parent
owns this total, validates every token, and uses it as the fixed denominator.
`--block-size` accepts `2...4`. A one-position block is used only internally
when the fixed parent-owned decode total leaves a final target-only tail; the
trained-assistant command rejects configured `K=1` because it would never
draft.

For opt-in trusted/local/upstream model-backed validation, run the following
regressions — artifact loading, direct pair/four numeric parity, direct-four
accounting, forced pair-composition seams, and deep growth/wrap — using only
the public/generated non-hidden oracle:

```bash
MLXFAST_RUN_MTP_RUNTIME_TESTS=1 \
MLXFAST_MTP_WEIGHTS_PATH="${PWD}/mtp-weights" \
MLXFAST_MTP_TARGET_DIR="${MLXFAST_MTP_TARGET_DIR}" \
MLXFAST_MTP_ASSISTANT_DIR="${MLXFAST_MTP_ASSISTANT_DIR}" \
swift test -c release --filter trainedMTPArtifactValidationRuntimeGate

MLXFAST_RUN_MTP_EXACT_PAIR_TESTS=1 \
MLXFAST_MTP_WEIGHTS_PATH="${PWD}/mtp-weights" \
MLXFAST_MTP_PAIR_GOLDEN_PATH=/tmp/gemma4-31b-it-mtp-public.json \
swift test --filter exactPairAndFourRuntimePreserveTokensWithinNumericEnvelope

MLXFAST_RUN_MTP_EXACT_PAIR_TESTS=1 \
MLXFAST_MTP_WEIGHTS_PATH="${PWD}/mtp-weights" \
MLXFAST_MTP_ASSISTANT_DIR="${MLXFAST_MTP_ASSISTANT_DIR}" \
MLXFAST_MTP_PAIR_GOLDEN_PATH=/tmp/gemma4-31b-it-mtp-public.json \
swift test --filter exactFourSessionCountersCoverFullZeroAndPartialAcceptance

MLXFAST_RUN_MTP_EXACT_PAIR_TESTS=1 \
MLXFAST_MTP_WEIGHTS_PATH="${PWD}/mtp-weights" \
MLXFAST_MTP_ASSISTANT_DIR="${MLXFAST_MTP_ASSISTANT_DIR}" \
MLXFAST_MTP_PAIR_GOLDEN_PATH=/tmp/gemma4-31b-it-mtp-public.json \
swift test --filter exactPairSessionForcedAcceptanceSeamsMatchSerial

MLXFAST_RUN_MTP_EXACT_PAIR_TESTS=1 \
MLXFAST_RUN_MTP_DEEP_OFFSET_TESTS=1 \
MLXFAST_MTP_WEIGHTS_PATH="${PWD}/mtp-weights" \
swift test --filter exactPairDeepOffsetsGrowthAndWrapMatchSerial
```

Do not claim model-backed success unless all commands run with pinned weights
and a public or operator-controlled non-hidden IT-target oracle. Hosted CI
cannot run them because it has no weights. The ranked workflow deliberately
does not invoke candidate-linked tensor tests or pass hidden oracle paths to
them. Its controls are parent-owned exact token IDs and logical
protocol/report-consistency validation, track-aware static review, and hidden
behavior checks. Cache offsets, physical rollback/geometry, and numerical
state remain checked inside the candidate worker/model plus trusted
implementation tests and manual/operator validation. Before relying on or
loosening an envelope for a kernel change, run a separate trusted M5 calibration
that is not linked into submission code.

## Residual risks

- Public/synthetic IT-target M5 goldens have been exercised remotely; hidden
  correctness and benchmark goldens remain operator-owned and are intentionally
  not exposed to tensor regression tests.
- The paired serial reference and aggregate 1.0 component floor are established
  in the live workflow/fixture.
- The exact-pair kernels amortize dense weight traversal for two rows at a
  time, while the direct exact-four path uses dedicated four-row kernels.
  Trusted/local regressions require exact token behavior/cache geometry and
  bounded finite intermediates, not a particular reduction order. They require
  M5 calibration before an optimization relies on or loosens the provisional
  envelopes; the serial K=1 mode remains the fail-closed development control.
- Decodes whose sliding offset crosses the 1,024-position window wrap lose
  pair eligibility and serialize the remaining rows; within the current
  512-seed/512-decode contract this affects only the final pair position.
- The path-based upstream loader cannot atomically bind open descriptors;
  before/after hash validation plus read-only operator ownership mitigates
  TOCTOU, so official provisioning must continue enforcing ownership and
  permissions.
- The parent can prove returned-token parity, bounds, nonce/ID behavior, wall
  time, and denominator. It cannot directly prove that editable model code
  internally used the assistant, ran target verification, or rolled every
  physical buffer correctly when outputs still match. Static review,
  prompt-independent hidden tests, candidate-worker cache-offset checks, memory
  telemetry, and manual frontier audit are still required.
- Seatbelt does not prove absence of already-running background GPU work.
  Fresh workers, allocator reset, no prompt before the timer, process reaping,
  and operator telemetry are required; active-allocation accounting remains a
  tracked hardening item.
