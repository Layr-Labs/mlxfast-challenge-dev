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

`Gemma4RuntimeModel` now conforms to `Gemma4MTPTarget`. Its MTP-only forward
uses the pinned library trunk's pre-norm/shared-KV capture hook. The ordinary
serial forward and optimized fast engine are unchanged.

`Gemma4TrainedMTPBlockSession` persists, for one request:

- the target KV caches;
- the last committed target bonus token;
- the target pre-norm hidden state;
- the last full-attention and sliding-attention shared K/V snapshots;
- a host mirror of the target cache offset.

For block size `K` (initially four), one round:

1. Uses the trained assistant to draft `K - 1` tokens from the current target
   token, target hidden state, and shared K/V.
2. Constructs one charged block verification graph from `K` exact serial-shape
   target forwards over the committed token followed by drafts.
3. Compares each draft to the corresponding target argmax in order.
4. Emits only the target-confirmed prefix plus the target token at the first
   rejection (between one and `K` tokens).
5. Stops target execution at the first rejection, so no rejected physical KV
   row is written; every retained hidden/K/V row uses the serial K=1 shape.
6. Persists hidden/shared-KV state at the committed target position.
7. Checks every target cache offset against the host mirror.

The serial target shape is deliberate. The expanded M5 pilot proved that a
single mathematical K-row target forward changes floating-point reduction
order: retained layer-1 K/V diverged after the first full-acceptance block and
a public prose continuation flipped argmax at decode step 48. Logical offsets
were still identical. Verifying each supplied row through the K=1 target shape
preserves exact token and physical-cache parity without hiding a fallback or
discarding unvalidated work; it also means this prototype does not yet realize
the target-compute amortization expected from production MTP.

Zero, partial, and full acceptance use the same charged path. If exactly one
output remains, a target-only one-position tail step finishes the configured
window; this is not an assistant fallback and cannot be selected for the main
rounds.

Any exception after a block begins poisons the worker session. The worker
cannot continue from ambiguous cache state.

## Trusted block protocol

The trained worker accepts only:

- `mtp_decode_begin`: the target seed prompt.
- `mtp_decode_block`: request ID, last parent-committed token, and maximum block
  size.

Block requests contain no expected token, future oracle token, accepted count,
worker duration, score, denominator, prompt hash, or continuation history.
Request IDs must be monotonic. Responses carry a nonce and a nonempty token
block only. Unknown response fields such as `accepted_count`, `token_count`,
`seconds`, or `future_tokens` are rejected by the parent decoder.

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
A K-row target graph is mathematically causal but is not bit-identical to K
serial target graphs on this runtime. The fix verifies supplied target rows
using the exact serial K=1 graph and stops at the first rejection, avoiding
both divergent retained state and irreversible post-wrap rollback. A
runtime-gated public parity test preserves the failing seam.

### Final public/synthetic matrix

The corrected serial-equivalent verifier completed 12 thermal-gated M5 pairs:
three each for copy (N=255), prose (N=256), code (N=257), and reasoning
(N=256). All target tokens matched. The matrix exercised full, partial, and
zero draft acceptance plus one-token tails. Mean paired candidate/serial
speedups were 0.741x copy, 0.798x prose, 0.860x code, and 0.883x reasoning;
after the complete fixed K=1 warmup, a fresh copy triplet measured 0.860x.
Exact verification is therefore a correctness-complete control, not a
competitive target-amortizing MTP implementation.

Per-category paired CV was initially 7.8% copy, 1.9% prose, 4.0% code, and
1.1% reasoning. Phase diagnostics showed copy's variation came from seed and
first-block onset (up to 1.77s) while median block latency and thermal/frequency
telemetry stayed stable. The fixed BOS warmup had not guaranteed all four exact
target rows were compiled when the assistant rejected early. Explicitly warming
four deterministic K=1 target rows reduced a fresh three-pair copy run to 3.2%
paired CV without retaining allocator buffers or using prompt-specific work.
Across accepted runs, maximum temperature was 56.1C and minimum steady GPU
frequency was 1604 MHz.

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
  --require-trained-assistant
```

`--tokens` defaults to 128 and accepts `1...512`; the selected golden must
contain one seed token plus the requested number of decode tokens. The parent
owns this total, validates every token, and uses it as the fixed denominator.

For the opt-in model-backed validation:

```bash
MLXFAST_RUN_MTP_RUNTIME_TESTS=1 \
MLXFAST_MTP_WEIGHTS_PATH="${PWD}/mtp-weights" \
MLXFAST_MTP_ASSISTANT_DIR="${MLXFAST_MTP_ASSISTANT_DIR}" \
swift test -c release --filter trainedMTPArtifactValidationRuntimeGate
```

Do not claim model-backed success unless these commands run with the pinned
weights and an IT-target oracle.

## Rollout blockers and residual risks

- Public/synthetic IT-target M5 goldens have been exercised remotely, but no
  IT hidden behavior suite is checked in.
- No paired MTP reference baseline or component floors exist.
- Exact serial-shape target verification prioritizes correctness over target
  compute amortization. A future fast path needs K-row kernels proven
  bit-identical to K=1 hidden/KV and logits before it can replace this path.
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
