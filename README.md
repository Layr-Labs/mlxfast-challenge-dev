# mlxfast — Qwen3.6-27B 4-bit

This branch is the Swift-only Qwen3.6-27B text-tower conversion of the
MLXFast Apple Silicon inference challenge.

## Current status

- Frozen checkpoint:
  `mlx-community/Qwen3.6-27B-4bit@c000ac2c2057d94be3fa931000c31723aac53282`.
- Immutable internal architecture name: `qwen3_5_text`.
- Active runtime baseline: the pinned public `Qwen35TextModel` from
  `mlx-swift-lm`.
- Editable custom runtime: present as an optimization scaffold, but production
  activation remains hardcoded off until real-checkpoint parity and a separate
  operator approval both pass.
- Transform/runtime contract: Qwen text only, exactly 1,847 retained tensors,
  no vision tensors, no `mtp.*` tensors, and no weight streaming.

This branch is intentionally not officially rankable yet. The protected M5
workflow, checked-in `correctness_prompts/**`, hidden R2 goldens, GPQA
references, semantic threshold, and paired baseline are still Gemma-specific.
They are preserved unchanged so the incomplete Qwen migration fails closed.
Never use a checked-in Gemma golden to validate or score Qwen.

See [`TASK.md`](TASK.md) for the branch contract and
[`docs/qwen3.6-operator-handoff.md`](docs/qwen3.6-operator-handoff.md) for the
remaining M5 operator work.

## Local setup

```bash
./setup.sh

# setup.sh prints the canonical source path. The default is:
MLXFAST_OFFLINE_WRITABLE_PATHS="${PWD}/weights" \
  .github/scripts/run-offline.sh .build/release/mlxfast-swift transform \
  --reference \
  ".cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/snapshots/c000ac2c2057d94be3fa931000c31723aac53282" \
  --output weights
```

`setup.sh` verifies the source snapshot against
`fixtures/reference_qwen3_6_27b_4bit.sha256`. The transform output contains a
flattened Qwen text `config.json`, `model.safetensors.index.json`, all indexed
text-only shards, and tokenizer metadata.

Local correctness and benchmark commands require an explicit external
Qwen-generated golden. There is deliberately no fallback to the checked-in
Gemma fixtures:

```bash
export MLXFAST_CORRECTNESS_GOLDEN_PATH="/absolute/path/to/qwen-correctness-golden.json"

.build/release/mlxfast-swift correctness \
  --weights weights \
  --golden "${MLXFAST_CORRECTNESS_GOLDEN_PATH}"

./benchmark.sh --local-iterate
./benchmark.sh --local-submit
```

The golden must carry `"model_type": "qwen3_5_text"`. Any golden used for
benchmark execution must also carry both positive Qwen values
`baseline_prefill_seconds_per_token` and
`baseline_decode_seconds_per_token`, unless the trusted caller supplies both
paired-baseline environment values. Preflight and benchmark execution enforce
the same rule.

Do not run `./benchmark.sh --official` for Qwen on this branch. The protected
workflow still provisions Gemma artifacts and cannot produce a valid Qwen
ranking.

## Tests

```bash
swift build -c release
swift test
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test
```

Real-checkpoint parity is opt-in and consumes transformed Qwen weights, not the
raw Hugging Face snapshot:

```bash
MLXFAST_RUN_QWEN_REFERENCE_PARITY=1 \
MLXFAST_QWEN_REFERENCE_WEIGHTS_PATH="/absolute/path/to/transformed-qwen-weights" \
swift test --filter Qwen35ReferenceParityTests
```

That gate checks exact inventory, library load, deterministic synthetic-token
prefill, cached decode versus full context, chunked versus one-shot execution,
and the inactive custom engine against library logits/top token. It skips
cleanly when the opt-in flag or transformed path is absent.

## Runtime architecture

The frozen text tower has:

- 64 dense decoder layers in `[linear, linear, linear, full] × 16` order;
- hidden size 5,120 and dense SwiGLU intermediate size 17,408;
- 48 Gated DeltaNet layers with 48 value heads, 16 key heads, 128-dimensional
  key/value heads, and a four-token depthwise convolution;
- 16 gated global-attention layers with 24 query heads, four KV heads,
  256-dimensional heads, and 64 rotated dimensions;
- affine 4-bit weights with group size 64;
- an untied quantized LM head; and
- 48 `MambaCache` plus 16 `KVCacheSimple` entries.

The source config declares one MTP layer, but the frozen checkpoint has no MTP
tensors. MTP is therefore disabled. Vision and multimodal execution are out of
scope.

The library constructor performs useful prompt-independent graph warmup.
Trusted request handling then resets the MLX allocator at every new
correctness, prefill, or decode sequence after the parent timer starts. This
clears free constructor buffers so they cannot subsidize charged work while
preserving legitimate cache reuse across token-step requests.

## Editable surface

Only these paths are packaged as a submission:

- `Sources/MLXFastModel/`
- `Sources/MLXFastTransform/`

`Sources/MLXFastCore/`, `Sources/MLXFastHarness/`,
`Sources/MLXFastCLI/`, tests, scripts, docs, generated weights, checkpoints,
and golden files are trusted or operator-owned surfaces.

The custom Qwen engine is not selected by environment variables.
`Qwen35FastPathReadiness.realCheckpointParityPassed` and
`productionActivationApproved` both remain `false`.

## Local scoring

The score formula remains:

```text
decode_speedup = baseline_decode_sec_per_token / decode_sec_per_token
prefill_speedup = baseline_prefill_sec_per_token / prefill_sec_per_token
score = decode_speedup^0.75 * prefill_speedup^0.25
```

The component floors remain:

```text
decode_speedup >= 0.95
prefill_speedup >= 0.95
```

The current constants imply local ceiling values of
`0.14064626165296054` seconds/token for decode and
`0.011163191525904606` seconds/token for prefill. Those constants are retained
from the protected Gemma contract and are not authenticated Qwen baselines.
Qwen local runs therefore require explicit external baseline values and remain
directional only. The benchmark window checks 64 correctness steps and 128
decode steps.

The complete model is RAM-resident. There is no expert cache or streaming
bandwidth to optimize, so `bandwidth_source=ram_resident_model` and
`bandwidth_gb_per_token=0`.

## Remaining operator blockers

Before Qwen can rank on the M5, an operator must:

1. provision and manifest-verify the exact Qwen snapshot on the runner;
2. run real-checkpoint library/custom parity;
3. regenerate public and hidden Qwen goldens from the unchanged prompt text;
4. regenerate Qwen GPQA accepted sequences and calibrate semantic/TTFT gates;
5. establish an immutable Qwen paired baseline identity, calibration, and
   benchmark oracle;
6. replace the protected Gemma workflow paths, hashes, layer count, R2 objects,
   and M5 reference path; and
7. run the complete M5 correctness and thermally gated paired timing stack.

Gemma prompts, token IDs, expected continuations, GPQA references, semantic
thresholds, benchmark oracles, and baseline measurements are not Qwen evidence.

## Yukon CLI

The account/submission client is external to this repository:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
mlxfast login <api-key> --api <url>
mlxfast clone <benchmark-id-or-name>
mlxfast submissions
```

Do not submit this Qwen branch for official ranking until the operator blockers
above are complete.

## Requirements

- Apple Silicon with enough unified memory for the approximately 16 GB
  quantized text tower plus caches and intermediates
- macOS with Swift 6 and the Xcode Metal toolchain
- CMake for `tools/build-mlx-metallib.sh`
