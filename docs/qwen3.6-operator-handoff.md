# Qwen3.6 operator handoff

## Frozen identity

This branch targets only:

- checkpoint: `mlx-community/Qwen3.6-27B-4bit`
- checkpoint revision:
  `c000ac2c2057d94be3fa931000c31723aac53282`
- source manifest: `fixtures/reference_qwen3_6_27b_4bit.sha256`
- internal text architecture: `qwen3_5_text`
- pinned `mlx-swift-lm` revision:
  `bc1c0ee67d15798343be17c9f8f61f7c0d977149`
- pinned `mlx-swift` revision:
  `df1fdc5f7821a1fabe921fdefbc42ac74dcfb6bc`

The transformed checkpoint is text-only and contains exactly 1,847 tensors:
seven top-level tensors, 30 tensors for each of 48 linear-attention layers,
and 25 tensors for each of 16 full-attention layers. It contains no
`vision_tower.*` or `mtp.*` tensors. The source config declares one MTP layer,
but the frozen checkpoint does not provide an MTP head, so MTP remains
disabled.

## What is ready

- Local setup and transform defaults point at the frozen Qwen checkpoint and
  verify it with the Qwen manifest.
- The transform retains only `language_model.*`, writes a flattened text
  `config.json`, and produces the indexed safetensors layout consumed by the
  runtime.
- Runtime config and metadata validation pin the 64-layer
  `[linear, linear, linear, full] × 16` architecture and exact 1,847-tensor
  inventory.
- The active runtime loads the public library `Qwen35TextModel`, uses 48
  `MambaCache` plus 16 `KVCacheSimple` entries, and leaves MTP detached.
- The editable custom attention, Gated DeltaNet, MLP, RoPE, block, cache, and
  fast-engine implementation is compile- and synthetic-test covered, but it
  is inactive.
- `Tests/MLXFastTests/Qwen35ReferenceParityTests.swift` provides an opt-in
  real-checkpoint gate without reading `correctness_prompts`.

The pinned library passed the deterministic tiny-model normal-inference check:
one-shot full context matched prefill plus cached one-token decode, and
chunked prefill matched one-shot prefill. Therefore `Package.resolved` was not
advanced speculatively.

## Real-checkpoint parity input

Run the opt-in test only with a transformed Qwen text weights directory:

```bash
.build/release/mlxfast-swift transform \
  --reference /absolute/path/to/Qwen3.6-27B-4bit \
  --output /absolute/path/to/qwen-transformed-weights

MLXFAST_RUN_QWEN_REFERENCE_PARITY=1 \
MLXFAST_QWEN_REFERENCE_WEIGHTS_PATH=/absolute/path/to/qwen-transformed-weights \
swift test --filter Qwen35ReferenceParityTests
```

`MLXFAST_QWEN_REFERENCE_WEIGHTS_PATH` must name the transform output directory,
not the raw Hugging Face snapshot. It must contain:

- the runtime-authored, flattened text `config.json`;
- `model.safetensors.index.json`; and
- every and only the safetensors shard referenced by that index.

The test validates the exact inventory, absence of vision/MTP tensors, strict
library model load, deterministic synthetic-token prefill, cached one-token
decode against uncached full context, chunked against one-shot execution, and
the explicitly loaded but production-inactive custom engine against library
logits and top token. If either opt-in variable is absent, it prints the
missing requirement and exits cleanly without loading a model.

## Protected state intentionally untouched

The conversion deliberately did not modify:

- `correctness_prompts/**`;
- `.github/workflows/benchmark.yml`;
- `ops/m5-bench/**`;
- private R2 object paths or their pinned hashes/byte counts;
- public or hidden golden hashes;
- M5 runner paths, thermal policy, oracle cache, baseline workspace, or
  baseline calibration; and
- any operator-owned file under `/opt/bench` or `/opt/bench-runner`.

Those surfaces still describe Gemma. This branch is intentionally not
rankable. A Gemma-generated public golden, hidden golden, GPQA token reference,
benchmark oracle, semantic threshold, or paired baseline must never be used to
accept or score Qwen.

The custom production gates also remain hardcoded false:
`Qwen35FastPathReadiness.realCheckpointParityPassed` and
`productionActivationApproved`. Running a local synthetic test is not grounds
to change either gate.

## Required operator migration

Perform these steps on the M5 in order.

1. Provision and verify the frozen Qwen source snapshot at a new
   runner-owned path, preferably
   `/opt/bench-runner/cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/snapshots/c000ac2c2057d94be3fa931000c31723aac53282`.
   Verify every file against
   `fixtures/reference_qwen3_6_27b_4bit.sha256`; do not overwrite or alias the
   existing Gemma cache.
2. Build and transform on the M5, record the source revision, dependency
   revisions, binary hash, Metal library fingerprint, and transformed
   weights-directory hash, then run the opt-in parity command above. Keep the
   custom gates false unless that real-checkpoint run passes and a separate
   activation review approves them.
3. Keep the checked-in prompt text unchanged. Retokenize it with the frozen
   Qwen tokenizer and generate new M5 Qwen outputs for both public fixtures:
   `correctness_prompts/public_longcopy_gate_english_512_256.json` and
   `correctness_prompts/public_longcopy_gate_english_512_1024.json`. Generate
   a new private base golden from the unchanged hidden prompt text as well.
   Record the observed Qwen token counts, SHA-256 values, and byte counts. If
   Qwen tokenization changes a configured token count, update the contract to
   that observed count without changing the prompt text.
4. Rebuild every hidden GPQA `accepted_token_sequences` entry with the Qwen
   tokenizer and Qwen library baseline, using the unchanged question/prompt
   text and answer keys. Capture Qwen semantic answers on the M5, run repeated
   judge baselines, and calibrate a new
   `MLXFastConstants.semanticGPQAMinPassCount` plus matching workflow
   `MLXFAST_SEMANTIC_GPQA_MIN_PASS`. Revalidate the GPQA token budget, case
   count, max-new-token budget, and first-token TTFT limits. None of the
   current Gemma-calibrated value `1` or its observations are Qwen evidence.
5. Choose and record one immutable trusted Qwen baseline ref and commit
   identity. Build `/opt/bench-runner/baseline/current` from that exact ref
   with the library backend and frozen identities above. Regenerate
   `/opt/bench-runner/state/baseline-calibration.json` from cool M5 runs and
   invalidate any Gemma binary-keyed benchmark-oracle cache. The candidate and
   paired denominator must resolve to the same Qwen checkpoint, tokenizer,
   dependency pins, and harness contract.
6. Upload new, Qwen-named private R2 objects only after their hashes are
   recorded. Update the protected workflow and its tests atomically. The
   following current values are Gemma-only and must be replaced:
   - `.github/workflows/benchmark.yml`
     `MLXFAST_REFERENCE_DIR` and
     `MLXFAST_REFERENCE_MANIFEST_PATH`;
   - `MLXFAST_CORRECTNESS_GOLDEN_R2_PATH`
     (`correctness_prompts/golden_prompt_benchmark_transcription_gate_english_512_256-gemma.json`);
   - `MLXFAST_GPQA_R2_PATH`
     (`correctness_prompts/gpqa_reference_cases-gemma.json`);
   - `MLXFAST_RAW_CORRECTNESS_GOLDEN_SHA256`
     (`56c282dcaac433543ef0eecb625cd99bc20f1ae1f7b9415efe32a71e6eb4eae9`)
     and byte count `38162`;
   - `MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_SHA256`
     (`182a7f98d24cc8f26e8b08505fe7a8b6d825702f99d0b78a83f49dd42f1b2aea`)
     and byte count `11140`; and
   - the semantic/GPQA calibration fields described above.
7. Audit the remaining protected operator surfaces in the same change:
   `.github/workflows/reference-cache-probe.yml` still names the Gemma cache
   and manifest, `.github/scripts/download-reference-cache-scope.sh` still
   defaults to the Gemma manifest,
   `.github/scripts/validate-benchmark-artifacts.sh` still requires
   `num_layers == 60`, and `ops/m5-bench/m5-bench-timing.yml` still names the
   Gemma reference cache. Update the corresponding source-string tests and
   operator-owned `/opt/bench-runner/measure-job.sh` identity/oracle
   configuration.
8. Run the complete M5 stack: release build, ordinary and MLX-enabled tests,
   transform/hash audit, opt-in real-checkpoint parity, regenerated public
   gate, full hidden base and behavior gates, exact GPQA prefixes, semantic
   GPQA judging, GPQA TTFT, benchmark oracle, and finally thermally gated
   paired timing. Rank only after every Qwen artifact and identity is sealed
   and no Gemma artifact appears in the run provenance.

Until all of those operator steps are complete, failures caused by the stale
protected workflow are expected fail-closed behavior, not permission to reuse
Gemma artifacts.
