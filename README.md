# mlxfast — DeepSeek V4 Flash

A benchmark arena for memory-bandwidth-optimal LLM inference on Apple Silicon.
Run DeepSeek V4 Flash without loading all 256 experts into RAM — and beat the baseline score.

See [CHALLENGE.md](CHALLENGE.md) for the full problem statement, scoring formula, and approach space.

## Quickstart

```bash
# Build the Swift harness, MLX metallib, install tools, and fetch weights if needed
./setup.sh

# Split dense weights into weights/ and write the expert streaming manifest
.build/release/mlxfast-swift transform

# Run the Darkbloom-compatible benchmark entrypoint.
# Requires the organizer-supplied correctness_golden.json.
./benchmark.sh

# Or call the Swift CLI directly
.build/release/mlxfast-swift preflight
.build/release/mlxfast-swift benchmark --score-path score.json
.build/release/mlxfast-swift submit --output mlxfast-submission.zip

# If required model artifacts are missing, the benchmark emits a valid failed
# score.json instead of a ranked score.
```

The benchmark writes `score.json` in the format consumed by Darkbloom.
`score.json` is a generated local output and is not tracked. The fixed
`correctness_golden.json` is also not tracked in the public repo; the benchmark
operator supplies it, or points the harness at it with
`MLXFAST_CORRECTNESS_GOLDEN_PATH=/path/to/correctness_golden.json`.

Full model setup needs a large local or mounted SSD. The reference checkpoint is
`mlx-community/DeepSeek-V4-Flash-4bit`, with 33 safetensors shards totaling about
141 GiB. `setup.sh` downloads it directly from Hugging Face with resumable
`curl` requests when `reference_weights/` is missing and checks for at least
170 GiB free by default. Use
`MLXFAST_REFERENCE_DIR=/Volumes/ssd/DeepSeek-V4-Flash-4bit` to point at a larger
volume, or `MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh` when the checkpoint will
be supplied separately. The Swift CLI also honors `MLXFAST_REFERENCE_DIR`,
`MLXFAST_WEIGHTS_PATH`, `MLXFAST_CORRECTNESS_GOLDEN_PATH`, and
`MLXFAST_SCORE_PATH` as defaults; explicit CLI flags take precedence.

## Why this challenge exists

DeepSeek V4 Flash has 256 routed experts per layer, 6 activated per token.
The checkpoint is too large to keep fully resident on typical Apple Silicon
machines. The baseline ships with SSD streaming: expert tensors stay on disk and
only the routed tensors needed for the current forward pass are materialized.

That baseline is functional but naive. Expert reads block the forward pass,
there is no prefetching, no cross-layer reuse, and the weights are stored in
their original 4-bit form. Every one of these is an optimisation target.

## The modifiable surface

Unlike typical inference benchmarks, the entire model execution pipeline is
in scope. Submissions should focus on the Swift targets listed in
`benchmark.json`:

| Path | What it controls |
|---|---|
| `Sources/MLXFastDeepSeek/` | DeepSeek V4 Flash runtime, MLX Swift array bridge, dense/expert loading, SSD streaming, decode/prefill logic. **Primary target.** |
| `Sources/MLXFastTransform/` | Offline weight transform from frozen reference safetensors into benchmark-ready `weights/`. |

The repository is Swift-only: setup, transform, correctness, and benchmark all
run through the Swift package.

`mlxfast-swift submit` reads `benchmark.json` and archives only the paths listed
in `editablePaths` for the Yukon backend. Generated `weights/`, reference
checkpoints, golden files, and local scores are not submitted.

## Scoring

```
score = peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token
```

Bandwidth is measured via **mactop hardware DRAM counters** — not a software model.
Correctness is a hard gate. See CHALLENGE.md for the full correctness specification.

**Baseline (TBD — reference M5 Max 128 GB):**

| Peak RAM | Bandwidth | Decode | Prefill | Score |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD |

## Architecture

```
Sources/
  MLXFastCLI/                Swift command-line entrypoint
  MLXFastCore/               score.json, golden cases, shared contracts
  MLXFastHarness/            trusted benchmark/provenance helpers
  MLXFastTransform/          Swift offline weight transform
  MLXFastDeepSeek/           DeepSeek V4 Flash Swift runtime
weights/                     transformed weights (harness loads from here)
  experts/
    manifest.json            byte ranges for streamed expert tensors
reference_weights/           original 4-bit checkpoint (frozen, read-only)
correctness_golden.json      hidden correctness cases and benchmark token oracle
score.json                   written after each benchmark run
```

For stricter organizer-side provenance, set `MLXFAST_VERIFY_TRANSFORM=1` when
running `benchmark.sh`. That re-runs the Swift transform into a clean temporary
directory and fails unless the submitted `weights/` tree is byte-equal to the
clean transform output.

Organizer golden files can be generated from a private prompt manifest:

```bash
.build/release/mlxfast-swift make-golden \
  --weights weights \
  --prompt-file private_prompts.json \
  --output correctness_golden.json
```

The manifest contains correctness prompts plus a dedicated benchmark prompt
(arrays shown abbreviated):

```json
{
  "version": 1,
  "cases": [
    {"name": "hidden-0", "prompt_tokens": [1, 2, 3]}
  ],
  "benchmark": {
    "name": "timed-hidden",
    "prompt_tokens": [1, 2, 3]
  }
}
```

The benchmark prompt must contain at least 512 token IDs. The generated golden
file stores exact expected tokens for the correctness gate, the 512-token
prefill check, the 32-token decode seed, and the timed 512-token decode window.

## Requirements

- Apple Silicon Mac, 24 GB+ unified memory (M2 or newer)
- macOS Sequoia or later
- Swift 6 / Xcode command line tools
- Xcode Metal Toolchain, installable with `xcodebuild -downloadComponent MetalToolchain`
- CMake, used by `tools/build-mlx-metallib.sh` to build `mlx.metallib`
- [mactop](https://github.com/metaspartan/mactop) — installed by `./setup.sh` via Homebrew when missing, or supplied with `MLXFAST_MACTOP_BIN=/path/to/mactop`
