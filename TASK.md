# mlxfast — Qwen3.6-27B Swift challenge

Optimize the frozen Qwen3.6-27B 4-bit text tower on Apple Silicon while
preserving the observable Qwen model behavior required by the eventual
correctness contract.

## Frozen target

- Repository: `mlx-community/Qwen3.6-27B-4bit`
- Revision: `c000ac2c2057d94be3fa931000c31723aac53282`
- Internal architecture: `qwen3_5_text`
- Runtime language: Swift with MLX
- Scope: text tower only

The transformed runtime inventory is exactly 1,847 tensors. Vision is dropped,
and no `mtp.*` tensors exist in the checkpoint. Although the source config
declares one MTP layer, MTP is disabled.

## Branch readiness

The active correctness baseline is the pinned library `Qwen35TextModel`.
An editable custom Qwen implementation exists for future optimization, but its
two production activation gates are hardcoded `false`.

This branch is intentionally not officially rankable. The protected
single-machine M5 workflow, checked-in `correctness_prompts/**`, private R2
objects, GPQA references, semantic threshold, and paired baseline still belong
to the Gemma benchmark. They remain untouched so an attempted Qwen official
run fails closed. Gemma goldens must never be used for Qwen.

## Local contract

Build and transform the pinned Qwen snapshot:

```bash
./setup.sh

MLXFAST_OFFLINE_WRITABLE_PATHS="${PWD}/weights" \
  .github/scripts/run-offline.sh .build/release/mlxfast-swift transform \
  --reference \
  ".cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/snapshots/c000ac2c2057d94be3fa931000c31723aac53282" \
  --output weights
```

Correctness and benchmark execution require an explicitly provisioned external
Qwen golden:

```bash
export MLXFAST_CORRECTNESS_GOLDEN_PATH="/absolute/path/to/qwen-correctness-golden.json"

.build/release/mlxfast-swift correctness \
  --weights weights \
  --golden "${MLXFAST_CORRECTNESS_GOLDEN_PATH}"

./benchmark.sh --local-iterate
./benchmark.sh --local-submit
```

Do not run the checked-in Gemma correctness prompts, and do not run
`./benchmark.sh --official` expecting a Qwen score.

The external golden must identify `"model_type": "qwen3_5_text"`. A benchmark
oracle must provide both positive explicit Qwen values
`baseline_prefill_seconds_per_token` and
`baseline_decode_seconds_per_token`, unless the trusted caller supplies both
paired-baseline overrides. Artifact preflight and the later benchmark use the
same resolution rule.

## Verification

```bash
swift build -c release
swift test
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test
```

When transformed real weights are available:

```bash
MLXFAST_RUN_QWEN_REFERENCE_PARITY=1 \
MLXFAST_QWEN_REFERENCE_WEIGHTS_PATH="/absolute/path/to/transformed-qwen-weights" \
swift test --filter Qwen35ReferenceParityTests
```

The optional parity test uses deterministic synthetic token IDs. It does not
read `correctness_prompts`.

## Architecture contract

The 64-layer dense model repeats three Gated DeltaNet layers and one global
attention layer:

- hidden size: 5,120
- dense SwiGLU intermediate size: 17,408
- global attention: 24 query heads, four KV heads, 256-dimensional heads
- partial RoPE: 64 of 256 dimensions, theta 10,000,000
- linear attention: 48 value heads, 16 key heads, 128-dimensional heads
- convolution kernel: four
- quantization: affine 4-bit, group size 64
- cache topology: 48 `MambaCache`, 16 `KVCacheSimple`
- LM head: untied and explicitly quantized

The model is fully RAM-resident after initialization. There is no expert
streaming, file I/O, or weight paging on the inference path.

## Trusted timing boundaries

Constructor warmup remains useful and prompt-independent. Before every new
correctness, prefill, and decode sequence, the trusted worker request handler:

1. applies the trusted phase-start MLX cache limit;
2. clears all free allocator buffers; and
3. verifies `Memory.cacheMemory == 0`.

The parent timer starts before the request, so allocator reset cost is charged.
No reset runs in `correctness_step` or `decode_step`; legitimate cache reuse
inside an already-started sequence remains intact.

Hidden semantic GPQA capture follows a collect-close-then-write boundary.
While the submitted worker is alive, trusted code collects only generated
tokens and a non-sensitive result. It closes and reaps the worker before
constructing or writing the bundle containing hidden prompts, answer keys, or
reference answers.

No `startStep` correctness slicing or editable decode-delay hook is permitted.
The scored decode path must invoke the same editable model entry points used by
correctness.

## Editable surface

Only:

```text
Sources/MLXFastModel/
Sources/MLXFastTransform/
```

Optimize Qwen attention, Gated DeltaNet, dense SwiGLU, quantized matmul,
cache handling, tensor layout, and MLX scheduling. Do not specialize for test
token IDs, prompts, or harness repetition.

## Score contract

The formula remains:

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
`0.011163191525904606` seconds/token for prefill. They are inherited
Gemma-calibrated constants, not Qwen baselines. Qwen benchmark execution
therefore requires explicit external values. The current windows contain 64
correctness steps and 128 decode steps.

The dense runtime reports `bandwidth_source=ram_resident_model` and
`bandwidth_gb_per_token=0`.

## Operator blockers

Official Qwen ranking remains unavailable until an M5 operator:

1. provisions and verifies the exact Qwen checkpoint;
2. runs real-checkpoint parity;
3. regenerates public and hidden Qwen goldens from unchanged prompt text;
4. regenerates Qwen GPQA references and calibrates semantic/TTFT thresholds;
5. pins a Qwen baseline ref/identity and regenerates calibration/oracle state;
6. updates protected workflow reference paths, manifests, R2 paths, hashes,
   byte counts, and the expected 64-layer identity; and
7. executes the full M5 correctness and thermally gated paired measurement.

See `docs/qwen3.6-operator-handoff.md` for exact stale protected values and the
ordered handoff.

## Submission tooling

The Yukon client is installed separately:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
mlxfast login <api-key> --api <url>
mlxfast clone <benchmark-id-or-name>
mlxfast submissions
```

Do not submit this branch for official ranking before the operator migration.
