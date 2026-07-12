# MLXFast Qwen3.6 Agent Guide

This branch is the Qwen-only conversion of the Swift MLXFast challenge.

## Frozen identity

- Checkpoint: `mlx-community/Qwen3.6-27B-4bit`
- Revision: `c000ac2c2057d94be3fa931000c31723aac53282`
- Internal architecture: `qwen3_5_text`
- Manifest: `fixtures/reference_qwen3_6_27b_4bit.sha256`
- Runtime inventory: exactly 1,847 text tensors

The model is dense and fully RAM-resident. Vision is excluded. The checkpoint
has no `mtp.*` tensors, so MTP is disabled even though config metadata declares
one MTP layer.

## Readiness boundary

The active runtime is the pinned library `Qwen35TextModel`. The editable custom
Qwen attention, Gated DeltaNet, MLP, RoPE, cache, and fast engine are an
inactive optimization scaffold.

Never flip either production gate without real-checkpoint parity and separate
operator approval:

```text
Qwen35FastPathReadiness.realCheckpointParityPassed = false
Qwen35FastPathReadiness.productionActivationApproved = false
```

This branch is intentionally not officially rankable. Protected M5 workflow
configuration and every checked-in/private correctness artifact are still
Gemma-specific. They must remain fail-closed until an operator completes the
Qwen migration. A Gemma golden must never be used as Qwen evidence.

## Editable and protected surfaces

Submission-editable paths:

```text
Sources/MLXFastModel/
Sources/MLXFastTransform/
```

Do not modify these operator/protected surfaces during normal Qwen model work:

- `correctness_prompts/**`
- `.github/workflows/benchmark.yml`
- `ops/m5-bench/**`
- private R2 paths, golden hashes, or byte counts
- M5 runner paths, thermal settings, baseline workspaces, or calibration
- operator-owned `/opt/bench` and `/opt/bench-runner` files

Tests and public documentation may change when the task explicitly asks for
them. Do not treat changes outside `editablePaths` as submission payload.

## Local workflow

```bash
./setup.sh

MLXFAST_OFFLINE_WRITABLE_PATHS="${PWD}/weights" \
  .github/scripts/run-offline.sh .build/release/mlxfast-swift transform \
  --reference \
  ".cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/snapshots/c000ac2c2057d94be3fa931000c31723aac53282" \
  --output weights

export MLXFAST_CORRECTNESS_GOLDEN_PATH="/absolute/path/to/qwen-correctness-golden.json"
.build/release/mlxfast-swift correctness \
  --weights weights \
  --golden "${MLXFAST_CORRECTNESS_GOLDEN_PATH}"

./benchmark.sh --local-iterate
./benchmark.sh --local-submit
```

The explicit golden must be Qwen-generated and identify
`"model_type": "qwen3_5_text"`. Do not instruct users to run the checked-in
Gemma fixtures. Qwen benchmark execution also requires both explicit positive
prefill/decode baseline values, either in the golden or as a complete trusted
paired override.

Do not run `./benchmark.sh --official` expecting a Qwen score. The protected
official workflow remains Gemma-wired.

Useful verification:

```bash
swift build -c release
swift test
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test

MLXFAST_RUN_QWEN_REFERENCE_PARITY=1 \
MLXFAST_QWEN_REFERENCE_WEIGHTS_PATH="/absolute/path/to/transformed-qwen-weights" \
swift test --filter Qwen35ReferenceParityTests
```

The parity path must be transform output, not a raw Hugging Face snapshot.

## Qwen architecture

- 64 layers: `[linear, linear, linear, full] × 16`
- hidden size 5,120; SwiGLU intermediate size 17,408
- 48 Gated DeltaNet layers: 48 value heads, 16 key heads, dimension 128
- 16 global-attention layers: 24 query heads, four KV heads, dimension 256
- partial RoPE: 64 rotated dimensions, theta 10,000,000
- affine 4-bit quantization, group size 64
- untied quantized LM head
- 48 `MambaCache` plus 16 `KVCacheSimple`

Focus optimization on quantized compute, global attention, recurrent linear
attention, cache handling, memory layout, and MLX scheduling. There is no
expert streaming or disk I/O on the hot path.

## Trusted request boundaries

Useful constructor warmup stays enabled. It is prompt-independent and compiles
the relevant graph shapes. It must not subsidize scored work with free
allocator buffers.

At every new correctness, prefill, and decode sequence, trusted worker code:

1. resets the phase-start cache limit;
2. calls `Memory.clearCache()`; and
3. fails closed unless `Memory.cacheMemory == 0`.

The parent starts timing before the request. Do not reset during
`correctness_step` or `decode_step`, because those requests legitimately reuse
state created inside the charged sequence.

Semantic GPQA capture must be collect-close-then-write. While submitted model
code is alive, collect generated tokens and non-sensitive status only. Close
and reap the worker before constructing or writing any bundle containing
hidden prompt text, answer keys, reference answers, or decoded candidate text.

Never restore teacher-forced `startStep` slicing or a phase-specific editable
decode-delay hook. The timed decode path must call the same editable entry
points used by correctness.

## Correctness and benchmark rules

- Never use `correctness_prompts/**` for Qwen; those token IDs and expected
  continuations are Gemma-generated.
- Benchmark preflight and execution must agree about the required Qwen oracle
  and baseline pair.
- Preserve teacher-forced and free-run cache semantics.
- Preserve the exact 1,847-tensor contract and reject vision/MTP/unexpected
  tensors.
- Keep the library model as oracle until real-checkpoint custom parity passes.
- Do not hardcode prompt IDs, GPQA answers, hidden fixtures, timing shortcuts,
  network access, or filesystem exfiltration.

Do not add caches or memos keyed on a request's input tokens whose only
possible hit is the benchmark harness repeating an identical computation.
Single-pass production inference is the contract. Input-independent weight,
kernel, RoPE, mask, and shape caches plus within-request KV/recurrent state
reuse remain valid.

## Scoring

The provisional formula remains:

```text
decode_speedup = baseline_decode_sec_per_token / decode_sec_per_token
prefill_speedup = baseline_prefill_sec_per_token / prefill_sec_per_token
score = decode_speedup^0.75 * prefill_speedup^0.25
```

Both components require speedup at least `0.95`. Current constants are
Gemma-derived local placeholders, not authenticated Qwen baselines. Qwen local
runs must carry explicit external baselines and are directional only.

`bandwidth_source=ram_resident_model` and `bandwidth_gb_per_token=0`.

## Operator blockers

Official Qwen ranking requires all of:

1. exact checkpoint provisioning and manifest verification on M5;
2. real-checkpoint parity;
3. new public/hidden Qwen goldens from unchanged prompt text;
4. Qwen GPQA references and semantic/TTFT calibration;
5. an immutable Qwen paired baseline identity and calibration/oracle;
6. protected workflow/R2/path/hash/layer-count updates; and
7. complete M5 correctness plus thermally gated paired timing.

See `docs/qwen3.6-operator-handoff.md`.

## Yukon CLI

```bash
export PATH="${HOME}/.local/bin:${PATH}"
mlxfast login <api-key> --api <url>
mlxfast clone <benchmark-id-or-name>
mlxfast submissions
```

Do not submit this branch for official ranking before the operator handoff is
complete.
