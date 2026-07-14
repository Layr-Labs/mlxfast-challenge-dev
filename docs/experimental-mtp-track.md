# Experimental Gemma 4 31B-IT MTP Track

## Status and isolation

This is a Phase 1, opt-in prototype. It is not an official ranked track and
cannot emit a score. The existing base-model `benchmark`,
`--local-iterate`, `--local-submit`, worker protocol, score formula, baseline,
and speculation ban remain unchanged.

The two experimental commands are:

- `mtp-probe`: PR #424's serial target-only block control.
- `mtp-benchmark`: a real trained-assistant MTP path that refuses to run
  without `--require-trained-assistant`.

No environment variable can turn serial `benchmark` into MTP.

## Compatibility conclusion

The current challenge target is:

- Runtime checkpoint:
  `mlx-community/gemma-4-31b-4bit@e236b3eb2f9567ded5875cfa89f1666afa1acbf1`
- Declared upstream base:
  `google/gemma-4-31B`
- Checked-in byte manifest:
  `fixtures/reference_gemma_4_31b_4bit.sha256`

Google's public MTP drafter is explicitly paired by its model card with the
instruction-tuned target, not the base target:

- Target:
  `google/gemma-4-31B-it@518276fb130dc81caf9a4f772e65e63ef2526493`
- Assistant:
  `google/gemma-4-31B-it-assistant@6c9152a7639e1f87626e4d4fd4dd9f3e20c9f3fb`

No matched public 31B base assistant was found. Architecture compatibility
alone does not prove training compatibility, so this prototype does not bind
the IT assistant to the base challenge checkpoint. It defines a separate
Gemma 4 31B-IT MTP track instead.

The MLX target conversion is pinned independently:

- `mlx-community/gemma-4-31b-it-4bit@696d436c404745a59f30e4939a658162b0a9e57f`
- Declared base model: `google/gemma-4-31B-it`

All three model cards currently declare Apache-2.0 and link the Gemma license
at <https://ai.google.dev/gemma/docs/gemma_4_license>. Operators must still
review the license terms before enabling a public ranked track.

## Provenance contract

`fixtures/gemma_4_31b_it_mtp_track.json` binds:

- track identity and disabled-scoring state;
- target and assistant model IDs plus immutable repository revisions;
- the exact pinned `mlx-swift-lm` revision
  `bc1c0ee67d15798343be17c9f8f61f7c0d977149`;
- frozen Gemma 4 target and assistant architecture fields;
- SHA256 manifest identities and artifact byte budgets;
- block size four, a compatibility default of 128 parent-counted decode
  tokens, and a hard trusted-parent cap of 512;
- the missing-reference-baseline status.

Byte manifests:

- `fixtures/mtp_gemma_4_31b_it_4bit.sha256`
- `fixtures/mtp_gemma_4_31b_it_assistant_bf16.sha256`

The assistant runtime inventory is exactly two regular, single-link files:

- `config.json`: 2,316 bytes
- `model.safetensors`: 939,042,560 bytes

The combined assistant directory is exactly 939,044,876 bytes. Extra files,
symlinks, hardlinks, size mismatches, hash mismatches, incompatible config
fields, or a total above 1,000,000,000 bytes fail before model load.

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

For configured block size `K` in `2...4`, one round:

1. Uses the trained assistant to draft `K - 1` tokens from the current target
   token, target hidden state, and shared K/V.
2. Verifies two target rows at a time when a pair is available. K=4 composes
   two exact pairs; K=3 uses one pair plus an exact K=1 bonus tail; K=2 uses one
   pair; K=1 is target-only.
3. Compares each draft to the corresponding target argmax in order and starts
   no later segment after the first rejection.
4. Emits only the target-confirmed prefix plus the target token at the first
   rejection (between one and `K` tokens).
5. If row zero of a pair rejects, removes row one's physical cache position
   from every layer and slices the shared-KV view before returning. If row one
   rejects, row zero's accepted input remains committed.
6. Persists hidden/shared-KV state at the committed target position.
7. Checks every target cache offset against the host mirror.

The exact-pair kernels share packed weight traversal while preserving each
row's K=1 accumulation and reduction order. They cover sliding Q/K/V, full
Q/K, attention output, gate/up activation, down projection, layer boundaries,
and the tied packed13 vocabulary head. RMS normalization remains row-serial.
Full D=512 attention remains two K=1 calls. Sliding D=256 attention uses one
causal two-query dispatch only below the library's 1,024-key shape switch;
otherwise it is row-serial. Row two therefore sees row one's K/V exactly as it
does in serial decoding.

The expanded M5 pilot proved why an ordinary mathematical K-row target
forward is insufficient: retained layer-1 K/V diverged after the first
full-acceptance block and a public prose continuation flipped argmax at decode
step 48, despite identical logical offsets. The exact-pair path is not that
ordinary batched path. Its regression gate compares both pair rows against two
K=1 forwards using exact logits, pre-norm hidden bits, every layer's physical
K/V state, and rollback state at multiple offsets.

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
- assistant sidecar: 939,044,876 bytes;
- combined: 19,383,465,057 bytes (about 18.1 GiB).

The assistant has 469,518,596 BF16 parameters according to its repository
metadata. The M5 matrix measured up to 47.6 GiB peak process RSS,
31.2 GiB MLX active memory, 3.4 GiB MLX cache memory, and 34.1 GiB MLX allocator
peak. Runtime peak is higher than file size because target and assistant
weights, target KV, shared K/V, verification logits/activations, and allocator
state coexist. This fits the 128 GiB runner but exceeds the base track's
practical 36 GiB local budget; a public track needs its own documented memory
minimum and host-enforced cap.

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
An ordinary batched K-row target graph is mathematically causal but is not
bit-identical to K serial target graphs on this runtime. The first fix
verified supplied target rows using the exact serial K=1 graph. The current
default replaces the ordinary batched graph with dedicated exact-pair kernels
that keep each row's K=1 accumulation order while sharing packed weight
traversal; the serial K=1 verifier is retained as the explicit control mode.
Runtime-gated parity tests preserve the failing seam: exact logits, pre-norm
hidden bits, and every layer's physical K/V bytes are compared against two
serial forwards at multiple offsets, across forced zero/partial/full and
second-segment acceptance, and across the deep growth/wrap boundaries.

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
scores from single pairs must not be quoted.

Phase 1 emits diagnostic JSON with no `score` or `speedup`.

Before publication, organizers must:

1. Freeze an IT-target public and hidden correctness set and serial oracle.
2. Freeze a trusted MTP reference implementation at the same target,
   assistant, block size, prompt, token count, sandbox, and thermal policy.
3. Measure candidate and reference back-to-back on `m5-bench`.
4. Decide whether the separate track is decode-only or keeps the existing
   decode/prefill weighting, then calibrate new component floors.
5. Add MTP-specific behavior and parity gates.
6. Enable the track only through a distinct workflow/track ID.

The base leaderboard and its paired serial reference are never mixed with MTP
results.

## Local/operator workflow

Provisioning (large download; do not run merely to compile the prototype):

```bash
./setup.sh
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

For the opt-in model-backed validation, run all four runtime gates — the
artifact gate, the basic pair-parity gate, the forced acceptance-seam gate,
and the deep growth/wrap gate:

```bash
MLXFAST_RUN_MTP_RUNTIME_TESTS=1 \
MLXFAST_MTP_WEIGHTS_PATH="${PWD}/mtp-weights" \
MLXFAST_MTP_ASSISTANT_DIR="${MLXFAST_MTP_ASSISTANT_DIR}" \
swift test -c release --filter trainedMTPArtifactValidationRuntimeGate

MLXFAST_RUN_MTP_EXACT_PAIR_TESTS=1 \
MLXFAST_MTP_WEIGHTS_PATH="${PWD}/mtp-weights" \
MLXFAST_MTP_PAIR_GOLDEN_PATH=/tmp/gemma4-31b-it-mtp-public.json \
swift test --filter exactPairRuntimeMatchesTwoSerialRowsBitForBit

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

Do not claim model-backed success unless all of these commands run with the
pinned weights and an IT-target oracle. CI cannot run them (no weights on
hosted runners); any official MTP workflow must run the full set on the
benchmark host before a kernel change ships.

## Rollout blockers and residual risks

- Public/synthetic IT-target M5 goldens have been exercised remotely, but no
  IT hidden behavior suite is checked in.
- No paired MTP reference baseline or component floors exist.
- The exact-pair kernels amortize dense weight traversal for two rows at a
  time; they are proven bit-identical by runtime gates, not by construction.
  Any future kernel change must rerun the exact-pair parity gates before it
  can ship, and the serial K=1 mode remains the fail-closed control.
- Decodes whose sliding offset crosses the 1,024-position window wrap lose
  pair eligibility and serialize the remaining rows; within the current
  512-seed/512-decode contract this affects only the final pair position.
- The path-based upstream loader cannot atomically bind open descriptors;
  before/after hash validation plus read-only operator ownership mitigates
  TOCTOU, but official provisioning must enforce ownership and permissions.
- The parent can prove returned-token parity, bounds, nonce/ID behavior, wall
  time, and denominator. It cannot directly prove that editable model code
  internally used the assistant, ran target verification, or rolled every
  physical buffer correctly when outputs still match. Static review,
  prompt-independent hidden tests, cache-offset checks, memory telemetry, and
  manual frontier audit are still required.
- Seatbelt does not prove absence of already-running background GPU work.
  Fresh workers, allocator reset, no prompt before the timer, process reaping,
  and operator telemetry are required; active-allocation accounting remains a
  rollout item.
