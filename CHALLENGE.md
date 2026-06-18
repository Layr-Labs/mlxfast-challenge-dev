# mlxfast — DeepSeek V4 Flash Swift Challenge

Optimize DeepSeek V4 Flash inference on Apple Silicon while preserving exact
greedy output for the supplied correctness prompts.

## Contract

Submissions are evaluated through the Swift harness:

```bash
./setup.sh
./benchmark.sh
```

The benchmark entrypoint:

1. Builds `mlxfast-swift` when needed.
2. Runs the Swift transform if `weights/` is missing or `MLXFAST_FORCE_TRANSFORM=1`.
3. Runs the correctness gate against `correctness_golden.json`.
4. Validates the benchmark prefill/decode tokens against the hidden benchmark
   oracle in `correctness_golden.json`.
5. Measures prefill latency, 512-step greedy decode latency, MLX peak memory, and
   `mactop` hardware DRAM bandwidth.
6. Writes `score.json` in the Darkbloom-compatible schema, plus
   `score.json.sha256` and `benchmark-integrity.json` audit sidecars.

If required artifacts are missing, the harness writes a failed `score.json`
rather than producing a ranked score.

## Model Artifacts

Place the frozen reference checkpoint here unless overriding with
`MLXFAST_REFERENCE_DIR`:

```text
reference_weights/DeepSeek-V4-Flash-4bit/
```

By default `setup.sh` downloads `mlx-community/DeepSeek-V4-Flash-4bit` directly
from Hugging Face with resumable `curl` requests when that directory is missing.
The safetensors payload is about 141 GiB across 33 shards; `setup.sh` requires
170 GiB free by default before starting. Set `MLXFAST_REFERENCE_DIR` to a larger
local or mounted SSD when the repo disk is too small, or set
`MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1` when the checkpoint is provisioned externally.

The Swift transform writes benchmark-ready weights here:

```text
weights/
  config.json
  model.safetensors.index.json
  experts/manifest.json
```

The generated `weights/` tree is a compact runtime artifact set, not a second
full copy of the checkpoint. It stores dense/shared tensors plus metadata, while
the baseline runtime streams routed expert tensors from the frozen reference
checkpoint. Submissions may adjust this overlay by changing both
`Sources/MLXFastTransform/` and `Sources/MLXFastDeepSeek/`; correctness and
benchmark results are the authority, not byte equality with the baseline
layout.

Correctness cases and the timed benchmark token oracle are supplied by the
benchmark operator and are intentionally not committed to the public repo:

```text
correctness_golden.json
```

Use `MLXFAST_CORRECTNESS_GOLDEN_PATH=/path/to/correctness_golden.json` when the
file is provisioned outside the repository root.

## Editable Surface

The active editable surface is Swift-only and is defined by `benchmark.json`:

| Path | Scope |
|---|---|
| `Sources/MLXFastDeepSeek/` | DeepSeek V4 Flash runtime, attention, MoE, expert streaming, correctness, benchmark timing. |
| `Sources/MLXFastTransform/` | Offline safetensors transform and expert manifest generation. |

`Sources/MLXFastCore/`, `Sources/MLXFastHarness/`, `Sources/MLXFastCLI/`,
scripts, tests, `benchmark.json`, generated `weights/`, reference checkpoints,
golden fixtures, and local scores are harness/operator files, not submission
surface. `mlxfast-swift submit` packages only `editablePaths`, rejects symlinks
and generated/model artifact paths, skips macOS metadata files, and applies a
256 MiB default source archive input cap. Override the cap with
`MLXFAST_MAX_SUBMISSION_BYTES` or `mlxfast-swift submit --max-bytes`. The submit
report includes the generated zip SHA-256 hash.

`mlxfast-swift verify-transform` is an organizer/debug check for deterministic
transform output. It re-runs the submitted transform and compares the generated
`weights/` tree against that fresh run. It is not a baseline-layout requirement.
The default transformed-output cap is 50 GiB; override it with
`MLXFAST_MAX_WEIGHTS_BYTES` or `--max-bytes` when running the verifier.

There is no Python harness path.

## Correctness Gate

Correctness is a hard gate. For each golden case, the harness runs cached greedy
generation for 256 tokens with temperature-zero behavior and compares token IDs
exactly. The first mismatch records the case, step, expected token, and actual
token in the failed report.

The gate is intended as a first-stage filter: an implementation that fails it is
not eligible for the longer benchmark.

The gate intentionally does not port the earlier Python hidden-state or top-K
logit comparison layers. The benchmark contract cares about the externally
observable greedy token stream for a text-to-text DeepSeek V4 Flash run. Exact
token-oracle checks are cleaner here because they validate the same output path
that is timed by the benchmark, avoid ambiguous internal tensor choices around
normalization/head-combination, and keep the hidden golden fixture small enough
to manage privately.

VLM/image inputs and speculative/MTP draft decoding are also out of scope for
this challenge. They should only be added if the official benchmark contract
changes to score those paths.

The hidden golden file also includes a benchmark oracle. The benchmark validates
the greedy token after the fixed 512-token prefill prompt, the greedy token
after the fixed 32-token decode seed, and all 512 tokens produced inside the
timed decode window before accepting a score.

## Score

```text
score = peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token
```

Lower is better.

`bandwidth_GB_per_token` is measured with `mactop` hardware DRAM counters during
the decode window. `setup.sh` installs `mactop` with Homebrew when needed; set
`MLXFAST_MACTOP_BIN=/path/to/mactop` to use a local binary instead.

## Useful Commands

```bash
swift test
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test
swift build -c release
.build/release/mlxfast-swift transform
.build/release/mlxfast-swift correctness
.build/release/mlxfast-swift preflight
.build/release/mlxfast-swift benchmark --score-path score.json
.build/release/mlxfast-swift make-golden --prompt-file private_prompts.json --output correctness_golden.json
.build/release/mlxfast-swift verify-transform
.build/release/mlxfast-swift clone
.build/release/mlxfast-swift submit --output mlxfast-submission.zip
```
