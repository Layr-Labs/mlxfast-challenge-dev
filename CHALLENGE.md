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
5. Measures prefill latency, 256-step greedy decode latency, MLX peak memory, and
   bandwidth. Bandwidth uses `mactop` hardware DRAM counters when available and
   falls back to expert-streaming read bytes when the runner does not expose
   IOReport DRAM channels.
6. Writes `score.json` in the Darkbloom-compatible schema, plus
   `score.json.sha256` and `benchmark-integrity.json` audit sidecars.

If required artifacts are missing, the harness writes a failed `score.json`
rather than producing a ranked score.

After transform, local users can run the checked-in public correctness gate with
`.build/release/mlxfast-swift correctness --weights weights`. For benchmark
iteration, `./benchmark.sh --quick` requires a local golden file that includes
the benchmark oracle; it checks only the first 64 correctness tokens and times
only 64 decode tokens. It still writes and prints `score.json`; it is a
directional local signal, not the official ranking run.

## Model Artifacts

By default, `setup.sh` stores the frozen reference checkpoint in a repo-local
Hugging Face-style cache:

```text
.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/main/
```

It also creates this compatibility symlink unless the path already exists:

```text
reference_weights/DeepSeek-V4-Flash-4bit/
```

By default `setup.sh` downloads `mlx-community/DeepSeek-V4-Flash-4bit` from the
configured mirror with resumable `curl` requests. It checks cached files against
the pinned SHA256 manifest and redownloads only missing, truncated, or
hash-mismatched files. The safetensors payload is about 141 GiB across 33
shards; `setup.sh` requires 170 GiB free by default before starting. After a
full verification, setup writes `.mlxfast-reference-cache.lock`; later setup
runs use cheap size/mtime checks from that lock and skip the full checkpoint
hash pass when the cache is unchanged. Set
`MLXFAST_REFERENCE_CACHE_DIR` or `MLXFAST_REFERENCE_DIR` to a larger local or
mounted SSD when the repo disk is too small, or set
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
`Sources/MLXFastTransform/` and `Sources/MLXFastModel/`; correctness and
benchmark results are the authority, not byte equality with the baseline
layout.

The public correctness-only prompt and golden are committed under
`correctness_prompts/` so participants can run a local correctness smoke test.
The timed benchmark token oracle is supplied by the benchmark operator and is
intentionally not committed to the public repo:

```text
correctness_golden.json
```

Use `MLXFAST_CORRECTNESS_GOLDEN_PATH=/path/to/correctness_golden.json` when the
file is provisioned outside the repository root.
Benchmark CI consumes the checked-in public golden for correctness-only runs and
downloads the private precomputed golden from protected storage for full
benchmark runs. Private prompt manifests and hidden benchmark goldens are not
committed to the public repository. The workflow does not generate goldens;
organizers regenerate them offline and upload the resulting file to protected
storage.

## Editable Surface

The active editable surface is Swift-only and is defined by `benchmark.json`:

| Path | Scope |
|---|---|
| `Sources/MLXFastModel/` | DeepSeek V4 Flash model implementation: attention, MoE, expert streaming, caches, weight loading, and prefill/decode execution. |
| `Sources/MLXFastTransform/` | Offline safetensors transform and expert manifest generation. |

`Sources/MLXFastCore/`, `Sources/MLXFastHarness/`,
`Sources/MLXFastCLI/`, `Sources/MLXFastSubmission/`, scripts, tests,
`benchmark.json`, generated
`weights/`, reference checkpoints, golden fixtures, and local scores are
harness/operator files, not submission surface. Correctness, scoring, timing,
golden generation, benchmark-oracle validation, provenance checks, and
submission packaging live in that trusted harness layer. `mlxfast-swift submit`
packages only `editablePaths`, rejects symlinks and generated/model artifact
paths, skips macOS metadata files, and applies a 256 MiB default source archive
input cap. Override the cap with `MLXFAST_MAX_SUBMISSION_BYTES` or
`mlxfast-swift submit --max-bytes`. Before packaging or upload, submit checks
the local Git diff against the trusted base ref and rejects any committed,
staged, unstaged, or untracked source changes outside `editablePaths`. The base
ref normally comes from `mlxfast-swift clone`/`link`; submit fails if no base ref
can be resolved, so pass `--base-ref REF` for manual checkouts.

Use `mlxfast-swift submit --dry-run --output mlxfast-submission.zip` for local
inspection. For Yukon upload, run `mlxfast-swift login <api-key> --api <url>`
once, then `mlxfast-swift link <benchmark-id-or-name>` for an existing checkout
or `mlxfast-swift clone <benchmark-id-or-name>` for a fresh checkout. Upload
with `mlxfast-swift submit <benchmark-id-or-name> --note "..."`. Uploads are
sent as a gzip tar archive with bearer-token auth; the backend applies the
archive to the frozen benchmark checkout and runs hidden validation. Use
`mlxfast-swift submissions <benchmark-id-or-name>` to inspect submitted jobs.
Pass `--idempotency-key KEY` when a live submit should be safely retried with a
stable backend idempotency key.

`mlxfast-swift verify-transform` is an organizer/debug check for deterministic
transform output. It re-runs the submitted transform and compares the generated
`weights/` tree against that fresh run. It is not a baseline-layout requirement.
The normal preflight/benchmark path also rejects generated `weights/` above the
default 50 GiB transformed-output cap before correctness or timing runs.
Override it with `MLXFAST_MAX_WEIGHTS_BYTES`; `verify-transform` additionally
accepts `--max-bytes`.

There is no Python harness path.

## Correctness Gate

Correctness is a hard gate. For each golden case, the prompt must contain
exactly 512 token IDs. The harness runs cached greedy generation for 256
tokens with temperature-zero behavior and compares token IDs exactly. The first
mismatch records the case, step, expected token, and actual token in the failed
report.

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
after the fixed 32-token decode seed, and all 256 tokens produced inside the
timed decode window before accepting a score.

## Score

```text
cost = peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token
score = 1 / cost
```

Higher is better. The component metrics remain in `score.json`, so operators can
still inspect the raw cost factors that produced the score.

`bandwidth_GB_per_token` prefers `mactop` hardware DRAM counters during the
decode window. `setup.sh` installs `mactop` with Homebrew when needed; set
`MLXFAST_MACTOP_BIN=/path/to/mactop` to use a local binary instead. If mactop
cannot collect IOReport DRAM samples, the score records
`bandwidth_source=expert_streaming_reads` and uses the measured expert-streaming
file bytes. Set `MLXFAST_REQUIRE_MACTOP_BANDWIDTH=1` to fail instead of falling
back.
`score.json` also carries audit-only wall-clock phase timings, final process RSS,
expert streaming counters, and transformed-weights digest fields. These values
help operators review runs but do not change the score formula.

## Useful Commands

```bash
swift test
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test
swift build -c release
MLXFAST_OFFLINE_WRITABLE_PATHS="${PWD}/weights" .github/scripts/run-offline.sh .build/release/mlxfast-swift transform --output weights
.build/release/mlxfast-swift correctness --weights weights
.build/release/mlxfast-swift preflight
.build/release/mlxfast-swift benchmark --score-path score.json
.build/release/mlxfast-swift benchmark --quick --score-path score.json
.build/release/mlxfast-swift verify-transform
.build/release/mlxfast-swift clone
.build/release/mlxfast-swift link <benchmark-id-or-name>
.build/release/mlxfast-swift submit --dry-run --output mlxfast-submission.zip
.build/release/mlxfast-swift submissions <benchmark-id-or-name>
```
