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
.build/release/mlxfast-swift submit --dry-run --output mlxfast-submission.zip

# If required model artifacts are missing, the benchmark emits a valid failed
# score.json instead of a ranked score.
```

The benchmark writes `score.json` in the format consumed by Darkbloom.
`score.json` is a generated local output and is not tracked. The fixed
`correctness_golden.json` is also not tracked in the public repo; the benchmark
operator supplies it, or points the harness at it with
`MLXFAST_CORRECTNESS_GOLDEN_PATH=/path/to/correctness_golden.json`.
`benchmark.sh` also writes `score.json.sha256` and `benchmark-integrity.json`,
which record the score file hash, golden hash, transformed `weights/` hash, and
transform source hash for run auditing.

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
The generated `weights/` tree is expected to stay small: it is a runtime
artifact overlay on top of the frozen reference checkpoint, not a second full
model copy. Submissions may change both the Swift transform and Swift runtime
to adjust metadata, caching, or streaming strategy, as long as the generated
runnable artifacts pass the hidden correctness and benchmark checks.

## The modifiable surface

Unlike typical inference benchmarks, the entire model execution pipeline is
in scope. Submissions should focus on the Swift targets listed in
`benchmark.json`:

| Path | What it controls |
|---|---|
| `Sources/MLXFastDeepSeek/` | DeepSeek V4 Flash runtime, MLX Swift array bridge, dense/expert loading, SSD streaming, decode/prefill logic. **Primary target.** |
| `Sources/MLXFastTransform/` | Offline weight transform from frozen reference safetensors into benchmark-ready `weights/`. |

The repository is Swift-only: setup, transform, correctness, and benchmark all
run through the Swift package. Correctness, scoring, timing, provenance, and
submission packaging are trusted harness code outside `editablePaths`, matching
the original Python harness split between editable model code and non-editable
judge code.

`mlxfast-swift submit --dry-run` reads `benchmark.json` and archives only the
paths listed in `editablePaths`. Generated `weights/`, reference checkpoints,
golden files, local scores, repository metadata, symlinks, and macOS metadata
files are not submitted. The default source archive input cap is 256 MiB;
override it with `MLXFAST_MAX_SUBMISSION_BYTES` or
`mlxfast-swift submit --max-bytes`. The dry-run report includes the generated
zip SHA-256 hash.

For Yukon upload, first store an API key:

```bash
.build/release/mlxfast-swift login <api-key> --api https://yukon-api.fly.dev
.build/release/mlxfast-swift link <benchmark-id-or-name>
.build/release/mlxfast-swift submit <benchmark-id-or-name> \
  --note "Changed expert streaming prefetch policy."
.build/release/mlxfast-swift submissions <benchmark-id-or-name>
```

The upload path packages the same editable paths as `submission.tar.gz` and
POSTs it to Yukon with `Authorization: Bearer <api-key>` and an idempotency key.
`YUKON_API_URL`, `YUKON_API_TOKEN`, `MLXFAST_API_URL`, `MLXFAST_API_KEY`, and
`MLXFAST_BENCHMARK_REF` can be used in CI or scripted runs. Use `--dry-run` to
force local packaging even when credentials are configured. `mlxfast-swift clone
<benchmark>` fetches the benchmark source repository from Yukon metadata and
writes local `yukon.*` git config; `mlxfast-swift link <benchmark>` writes the
same config into an existing checkout. Pass `--idempotency-key KEY` to make a
live submit retry use a stable backend idempotency key.

## Scoring

```
score = peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token
```

Bandwidth is measured via **mactop hardware DRAM counters** — not a software model.
Correctness is a hard gate. See CHALLENGE.md for the full correctness specification.
The score payload also includes audit-only fields for wall-clock benchmark time,
preflight time, correctness time, timed benchmark time, final process RSS, expert
streaming counters, and transformed-weights digest. These fields are for
operator review and are not additional scoring factors.

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
  MLXFastDeepSeek/           editable DeepSeek V4 Flash Swift runtime
  MLXFastDeepSeekHarness/    trusted correctness, golden, and benchmark runner
  MLXFastSubmission/         trusted Yukon login/submit integration
weights/                     transformed weights (harness loads from here)
  experts/
    manifest.json            baseline byte ranges for streamed expert tensors
reference_weights/           original 4-bit checkpoint (frozen, read-only;
                              baseline runtime also uses it as expert backing)
correctness_golden.json      hidden correctness cases and benchmark token oracle
score.json                   written after each benchmark run
```

The standard preflight/benchmark path enforces a default 50 GiB cap on the
generated `weights/` tree before correctness or timing runs. Change it with
`MLXFAST_MAX_WEIGHTS_BYTES`; use `0`, `none`, or `unlimited` only for organizer
debugging. For stricter organizer-side provenance, set
`MLXFAST_VERIFY_TRANSFORM=1` when running `benchmark.sh`. That re-runs the
submitted Swift transform into a clean temporary directory and fails unless
`weights/` is byte-equal to that fresh run. This checks determinism and stale
files; it does not require the baseline `weights/` layout. `verify-transform`
uses the same default cap and can also be changed with
`mlxfast-swift verify-transform --max-bytes N`.

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
