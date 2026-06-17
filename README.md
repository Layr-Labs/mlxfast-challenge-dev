# mlxfast — DeepSeek V4 Flash

A benchmark arena for memory-bandwidth-optimal LLM inference on Apple Silicon.
Run DeepSeek V4 Flash without loading all 256 experts into RAM — and beat the baseline score.

See [CHALLENGE.md](CHALLENGE.md) for the full problem statement, scoring formula, and approach space.

## Quickstart

```bash
# Build the Swift harness, MLX metallib, and install mactop if needed
./setup.sh

# Split dense weights into weights/ and write the expert streaming manifest
.build/release/mlxfast-swift transform

# Run the Darkbloom-compatible benchmark entrypoint
./benchmark.sh

# Or call the Swift CLI directly
.build/release/mlxfast-swift preflight
.build/release/mlxfast-swift benchmark --score-path score.json

# If required model artifacts are missing, the benchmark emits a valid failed
# score.json instead of a ranked score.
```

The benchmark writes `score.json` in the format consumed by Darkbloom.

## Why this challenge exists

DeepSeek V4 Flash has 256 routed experts per layer, 6 activated per token.
At 4-bit quantisation the full expert stack is ~30 GB — more than most Apple
Silicon machines can hold. The baseline ships with SSD streaming: only the 6
activated experts per token are loaded into Metal memory, keeping peak RAM
under ~6 GB.

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
  MLXFastTransform/          Swift offline weight transform
  MLXFastDeepSeek/           DeepSeek V4 Flash Swift runtime
weights/                     transformed weights (harness loads from here)
  experts/
    manifest.json            byte ranges for streamed expert tensors
reference_weights/           original 4-bit checkpoint (frozen, read-only)
correctness_golden.json      fixed greedy-token correctness cases
score.json                   written after each benchmark run
```

## Requirements

- Apple Silicon Mac, 24 GB+ unified memory (M2 or newer)
- macOS Sequoia or later
- Swift 6 / Xcode command line tools
- Xcode Metal Toolchain, installable with `xcodebuild -downloadComponent MetalToolchain`
- CMake, used by `tools/build-mlx-metallib.sh` to build `mlx.metallib`
- [mactop](https://github.com/metaspartan/mactop) — installed by `./setup.sh` via Homebrew when missing, or supplied with `MLXFAST_MACTOP_BIN=/path/to/mactop`
