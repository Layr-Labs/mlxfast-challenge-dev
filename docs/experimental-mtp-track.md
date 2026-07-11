# Experimental Gemma 4 MTP Track Foundation

## Status

This is an **experimental, opt-in protocol probe**, not a ranked MTP track.
The existing `benchmark`, `--local-iterate`, and `--local-submit` paths retain
their frozen one-token behavior, baselines, score formula, and 512-seed +
128-step charged window.

Run the probe explicitly:

```bash
.build/release/mlxfast-swift mtp-probe \
  --weights weights \
  --golden correctness_prompts/public_longcopy_gate_english_512_256.json \
  --block-size 4
```

The command emits diagnostic JSON with no `score` or `speedup`. Its current
generator is `serial_target_fallback`: one ordinary target forward per emitted
token. It proves block and cache semantics but does not claim or provide MTP
speedup.

## Why no trained drafter is enabled

The shipped checkpoint is dense Gemma 4 31B **base**, 4-bit, and contains only
the text target tower. The transform selects `language_model.*`; there are no
assistant/drafter tensors or an assistant artifact manifest.

The pinned `mlx-swift-lm` revision does contain the software surfaces needed by
a future implementation:

- `Gemma4MTPTarget` for target prefill/verification and speculative rollback.
- `Gemma4AssistantDraftModel` for loading and binding assistant weights.
- `Gemma4MTPTokenIterator` and `runGemma4MTPRounds` for speculative rounds.

The public assistant examples and model pairs in that dependency target Gemma
4 **IT** checkpoints. Architecture-only binding does not establish that those
weights are the organizer-approved drafter for this base target. This
foundation therefore does not download, transform, or bind them. Model-side
availability reports:

```text
assistant weights unavailable/incompatible: the shipped Gemma 4 31B base
checkpoint contains no drafter weights, and the known public Gemma 4 assistant
targets are IT models
```

An organizer-provided assistant must identify the exact base target revision
and pass parity, acceptance, memory, and performance qualification before it
can replace the serial fallback.

## Protocol and trust boundary

The trusted harness adds `decode_block` beside the existing `decode_step`
request. It is used only by `mtp-probe`.

Request fields:

- `id`
- `kind = "decode_block"`
- `token`: the last token already committed and validated by the parent
- `max_block_size`: `1...4`

Response fields:

- The normal request `id`, worker nonce, and success status
- A nonempty `tokens` block no longer than the requested maximum

There is deliberately no expected-token, expected-prefix, oracle, or future
token field in the request. The first request sends the validated seed token.
Every later request sends only the last token from the previously validated
block.

The trusted parent:

1. Loads the golden and builds the oracle plan outside submitted model code.
2. Starts its wall timer before `decode_begin`, so seed prefill/setup is
   charged.
3. Requests bounded blocks until exactly 128 decode tokens are committed.
4. Rejects missing, empty, oversized, out-of-vocabulary, overrun, or
   out-of-order/mismatching blocks.
5. Uses the fixed expected count of 128 as the seconds-per-token denominator.

The timer therefore charges seed setup, target forwards, protocol
serialization and round trips, and—when a real implementation exists—all
drafting, target verification, rejection, and rollback work. Worker-reported
timing is never accepted.

The worker rejects malformed `decode_block` shapes, block sizes above four,
and requests that could exceed 128 total tokens. Protocol lines retain the
existing 4 MiB bound; the four-token request/response shape is tested against
that limit. If a block generator throws after a possible partial forward, the
worker poisons that decode cache rather than continuing from ambiguous state.

## Cache contract

The serial fallback consumes the previous committed token at the current
offset, emits the greedy target token, and repeats autoregressively. For a
returned block of length `N`:

- exactly `N` target forwards run;
- offsets advance by exactly `N`;
- the worker's decoded-token state advances by exactly `N`;
- the final returned token is sent back as the next request's previous
  committed token.

`Gemma4TargetBlockGenerating` is the model-side replacement point. Any future
speculative implementation must return only target-confirmed tokens and leave
the target cache at exactly the committed prefix. Multi-token verification
will temporarily advance KV state beyond that prefix on rejection, so explicit
rollback/commit behavior is mandatory. A real MTP adapter must not be enabled
until rollback is implemented and parity-tested across zero, partial, and full
draft acceptance.

## Artifact and memory budget

The transformed artifact cap remains 25 GiB. The current target text tower is
about 17 GiB and remains fully RAM-resident. A future assistant artifact,
metadata, and any transformed copies must fit within the same cap unless the
organizer deliberately changes the challenge contract. It must not duplicate
the target checkpoint or introduce prompt-dependent caches.

Runtime qualification must include peak unified-memory measurements for target
weights, assistant weights, target and drafter KV caches, verification
activations, and rollback buffers on the ranked M5 Max. Local success on a
different Apple Silicon generation is not sufficient.

## Security invariants

- No prompt-specific hashes, lookup tables, continuation caches, or repeated-
  prompt shortcuts.
- No future oracle token is sent to submitted worker/model code.
- Only the trusted parent validates output and measures elapsed time.
- Block size is bounded at four and total output is exactly 128 tokens.
- No prompt-dependent model work occurs before the parent timer.
- The worker cannot read the golden path under the normal sandbox profile.
- Default ranked and local benchmark commands do not dispatch `decode_block`.
- The experimental report cannot produce or overwrite an official score.

## Path to a real ranked MTP track

1. Organizer supplies a versioned assistant checkpoint trained for the exact
   frozen Gemma 4 31B base target, plus provenance and hashes.
2. Extend transform metadata and worker preflight to validate the assistant
   architecture, target binding, tensor allowlist, and combined artifact cap.
3. Adapt the runtime target to `Gemma4MTPTarget` (or an equivalent audited
   verifier) without changing normal one-token logits.
4. Implement draft, batched target verification, acceptance, and explicit KV
   rollback behind `Gemma4TargetBlockGenerating`.
5. Add runtime-gated parity tests against serial greedy for rejection,
   partial-acceptance, full-acceptance, final-short-block, and long-offset
   cases.
6. Measure acceptance and end-to-end parent-timed throughput on representative
   public and hidden-independent prompts.
7. Define a separate official MTP track and rebaseline it on the ranked M5.
   Update the benchmark-window freeze, score contract, security review, thermal
   procedure, and paired reference measurement together.
8. Only then consider enabling MTP in a ranked command. The current `benchmark`
   default must remain unchanged unless that explicit contract migration lands.
