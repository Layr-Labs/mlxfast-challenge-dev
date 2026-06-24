# mlxfast — DeepSeek V4 Flash

A benchmark arena for memory-bandwidth-optimal LLM inference on Apple Silicon.
Run DeepSeek V4 Flash without loading all 256 experts into RAM — and beat the baseline score.

See [CHALLENGE.md](CHALLENGE.md) for the full problem statement, scoring formula, and approach space.

## Quickstart

```bash
# Check local tools, build the Swift harness/MLX metallib, and fetch weights if needed
./setup.sh

# Optional: split dense weights into weights/ and write the expert streaming
# manifest. setup.sh prints this command with the exact reference path it used.
MLXFAST_OFFLINE_WRITABLE_PATHS="${PWD}/weights" \
  .github/scripts/run-offline.sh .build/release/mlxfast-swift transform \
  --reference .cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/main \
  --output weights

# Run the checked-in public correctness gate.
.build/release/mlxfast-swift correctness --weights weights

# Run the Darkbloom-compatible benchmark entrypoint.
# Official benchmark runs use the organizer-supplied hidden correctness_golden.json.
./benchmark.sh

# Faster benchmark iteration when a local golden with a benchmark oracle is
# available: checks 64 correctness tokens, measures 64 decode tokens, writes
# score.json, and prints score.json to stdout.
./benchmark.sh --quick

# Or call the Swift CLI directly
.build/release/mlxfast-swift correctness --weights weights
.build/release/mlxfast-swift preflight
.build/release/mlxfast-swift benchmark --score-path score.json
.build/release/mlxfast-swift benchmark --quick --score-path score.json
.build/release/mlxfast-swift submit --dry-run --output mlxfast-submission.zip

# If required model artifacts are missing, the benchmark emits a valid failed
# score.json instead of a ranked score.
```

The benchmark writes `score.json` in the format consumed by Darkbloom.
`score.json` is a generated local output and is not tracked. Public
correctness-only workflow runs use the checked-in
`correctness_prompts/public_longcopy_gate_english_512_256.json` golden and
matching prompt text. Official benchmark runs use a hidden
`correctness_golden.json` supplied by the benchmark operator, or a harness path
set with `MLXFAST_CORRECTNESS_GOLDEN_PATH=/path/to/correctness_golden.json`.
`benchmark.sh` also writes `score.json.sha256` and `benchmark-integrity.json`,
which record the score file hash, golden hash, transformed `weights/` hash, and
transform source hash for run auditing.

Full model setup needs a large local or mounted SSD. The reference checkpoint is
`mlx-community/DeepSeek-V4-Flash-4bit`, with 33 safetensors shards totaling about
141 GiB. `setup.sh` downloads the checkpoint from the fast Darkbloom/R2 mirror by
default into a repo-local Hugging Face-style cache under
`.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/main/`.
It verifies cached files against `fixtures/reference_deepseek_v4_flash_4bit.sha256`
and redownloads only files that are missing, truncated, or hash-mismatched. A
compatibility symlink is created at `reference_weights/DeepSeek-V4-Flash-4bit`
for older commands, but current setup and CI pass the canonical cache directory
to transform explicitly. The downloader uses resumable `curl` requests, prints
numbered shard progress with elapsed time, and checks for at least 170 GiB free
by default. After a full SHA-256 verification, setup writes
`.mlxfast-reference-cache.lock` next to the checkpoint; later setup runs use
cheap size/mtime checks against that lock and skip the full 141 GiB hash pass
when the cache is unchanged. Use
`MLXFAST_REFERENCE_CACHE_DIR=/Volumes/ssd/hf-cache/.../snapshots/main` or
`MLXFAST_REFERENCE_DIR=/Volumes/ssd/DeepSeek-V4-Flash-4bit` to point at a larger
volume, or `MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh` when the checkpoint will
be supplied separately. If you use a custom cache path, either copy the exact
transform command printed by `setup.sh` or set `MLXFAST_REFERENCE_DIR` before
running `transform` or `benchmark.sh`. The Swift CLI also honors
`MLXFAST_REFERENCE_DIR`, `MLXFAST_WEIGHTS_PATH`,
`MLXFAST_CORRECTNESS_GOLDEN_PATH`, and `MLXFAST_SCORE_PATH` as defaults;
explicit CLI flags take precedence. Set `MLXFAST_REFERENCE_BASE_URL` to use
another HTTP checkpoint prefix, including Hugging Face. Run `./setup.sh --help`
for the full local setup knobs.

For manual GitHub Actions benchmark runs, dispatch `benchmark.yml` on a macOS
Blacksmith runner. Set `reference_base_url` to an HTTP prefix containing the
reference checkpoint files, such as an R2 public bucket or Worker route. The
workflow downloads the reference checkpoint into the same repo-local
Hugging Face-style cache path used by local setup, passes that path explicitly
to the offline transform, then prepares the correctness golden after transform
completes. Correctness-only workflow runs use the checked-in public
`correctness_prompts/public_longcopy_gate_english_512_256.json` fixture. Full
benchmark runs require a precomputed hidden `correctness_golden.json` through
the `correctness_golden_url` input, `MLXFAST_CORRECTNESS_GOLDEN_URL`
repository secret, or the private R2 object
`correctness_prompts/golden_prompt_benchmark_transcription_gate_english_512_256.json`.
Full benchmark runs also require the private R2
`correctness_prompts/gpqa_reference_cases.json` object. The workflow tokenizes
9 token-budget-valid hidden GPQA multiple-choice prompts locally and attaches
them as short-answer behavior gates before correctness runs. Each private GPQA
case must include reference-calibrated `accepted_token_sequences` or
`accepted_responses`; GPQA answer keys are metadata, not an exact-token oracle.
Generate those accepted token sequences on the official runner with
`mlxfast-swift calibrate-gpqa-gates --gpqa PATH --weights weights --tokenizer weights --output PATH`.
The official workflow checks the first generated GPQA answer token for each
case, using the stable prefix of any longer calibrated reference sequence.
Because this gate is exact-token based, calibrate it on the official Blacksmith
runner with the manual `calibrate_gpqa_reference` workflow input; M-series local
calibration can differ from the official runner even at temperature zero.
If none of those is configured, a full benchmark fails; it will not use a
committed prompt, committed golden, or Actions cache fallback for ranked
scoring. Final hidden goldens should come from protected storage. Private
endpoints can pass headers through
`MLXFAST_REFERENCE_AUTH_HEADER` and `MLXFAST_CORRECTNESS_GOLDEN_AUTH_HEADER`
repository secrets. Private R2 golden downloads use the `R2_ACCESS_KEY_ID`,
`R2_BUCKET_ENDPOINT`, and `R2_SECRET_ACCESS_KEY` secrets. See
[`docs/private-benchmark-security.md`](docs/private-benchmark-security.md) for
the private prompt and artifact handling model.

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
| `Sources/MLXFastModel/` | DeepSeek V4 Flash runtime, MLX Swift array bridge, dense/expert loading, SSD streaming, decode/prefill logic. **Primary target.** |
| `Sources/MLXFastTransform/` | Offline weight transform from frozen reference safetensors into benchmark-ready `weights/`. |

The repository is Swift-only: setup, transform, correctness, and benchmark all
run through the Swift package. Correctness, scoring, timing, provenance, and
submission packaging are trusted harness code outside `editablePaths`; only
the model and transform targets are contestant-editable.

`mlxfast-swift submit --dry-run` reads `benchmark.json` and archives only the
paths listed in `editablePaths`. Generated `weights/`, reference checkpoints,
golden files, local scores, repository metadata, symlinks, and macOS metadata
files are not submitted. The default source archive input cap is 256 MiB;
override it with `MLXFAST_MAX_SUBMISSION_BYTES` or
`mlxfast-swift submit --max-bytes`. The dry-run report includes the generated
zip SHA-256 hash. Before packaging or upload, submit also checks the local Git
diff against the trusted base ref and fails if any committed, staged, unstaged,
or untracked source changes are outside `editablePaths`. The base ref defaults
to Yukon metadata from `mlxfast-swift clone`/`link`, then `origin/main`; submit
fails if no base ref can be resolved, so pass `--base-ref REF` for a manually
prepared checkout.

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
cost = peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token
score = 1 / cost
```

Higher score is better. The cost product is still recoverable from the
component metrics in `score.json`, but the top-level `score` is an up-only value
so leaderboards and local comparisons read naturally.
Bandwidth prefers **mactop hardware DRAM counters**. On macOS virtualized runners
that do not expose IOReport DRAM channels, the harness records
`bandwidth_source=expert_streaming_reads` and uses measured expert-streaming file
bytes as a fallback. Set `MLXFAST_REQUIRE_MACTOP_BANDWIDTH=1` to fail instead of
falling back.
Correctness is a hard gate. See CHALLENGE.md for the full correctness specification.
The official run checks 256 correctness positions and times a 256-token decode
window. Public local correctness uses the checked-in correctness fixture. When
a local golden with a benchmark oracle is available, `--quick` shortens
correctness and decode to 64 token checks and prints the resulting `score.json`.
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
  MLXFastTransform/          Swift offline weight transform
  MLXFastModel/              editable DeepSeek V4 Flash Swift runtime
  MLXFastHarness/            trusted correctness, golden, and benchmark runner
  MLXFastSubmission/         trusted Yukon login/submit integration
weights/                     transformed weights (harness loads from here)
  experts/
    manifest.json            baseline byte ranges for streamed expert tensors
.cache/huggingface/hub/...   canonical frozen 4-bit reference checkpoint cache
reference_weights/...        compatibility symlink to the reference cache
correctness_prompts/         public correctness prompt and checked-in golden
correctness_golden.json      hidden benchmark correctness cases and token oracle
score.json                   written after each benchmark run
```

The baseline runtime loads dense/shared tensors from `weights/` and streams
routed expert tensors from the frozen reference checkpoint named by
`MLXFAST_REFERENCE_DIR`, falling back to the compatibility symlink when that
environment variable is not set.

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

The public correctness-only prompt and golden live in `correctness_prompts/`.
Private prompt manifests and hidden benchmark golden files are not committed or
generated by the benchmark workflow. In private benchmark CI, the normal path
downloads the precomputed
`correctness_prompts/golden_prompt_benchmark_transcription_gate_english_512_256.json`
object from R2, then downloads
`correctness_prompts/gpqa_reference_cases.json` and merges it into the local
golden as 9 hidden one-token GPQA behavior checks. Generate
final hidden benchmark goldens outside the public repository and provide the
resulting file to benchmark CI with R2, `correctness_golden_url`, or
`MLXFAST_CORRECTNESS_GOLDEN_URL`. The benchmark workflow stores its local golden
copy under `$RUNNER_TEMP`, not the repository workspace, and uploads only hash
and byte-count sidecars.

The Swift `make-golden` generator has been removed from the public harness so CI
only consumes precomputed fixtures. The last commit on this branch containing
that generator is `bcc9438fabf95a9b371d5749dd64f2f5ccc60fd5`.

Each base correctness prompt must contain exactly 512 token IDs. The benchmark
prompt must contain at least 512 token IDs. The precomputed golden file stores
exact expected tokens for each 512-token correctness prompt and its 256-token
greedy continuation, the 512-token prefill check, the 512-token decode seed, and
the timed 256-token decode window. During correctness, the harness checks those
continuation positions teacher-forced: after each accepted step it feeds the
golden previous token back into the model. This keeps the gate stable across
Apple GPU/software differences by preventing one earlier mismatch from
cascading into unrelated later-token failures. A token is accepted only when it
matches the expected token, except for a true top-logit tie within the tiny
`1e-6` logit tolerance used by the harness.

Private fixtures can also include a `correctness_gates` object with hidden
anchor logits, short free-run prefixes, and answer-token behavior checks.
Those gates are additive: public local correctness still works with the
checked-in fixture, while official benchmark fixtures can cover more adversarial
behavior without exposing prompt or answer data. Behavior checks compare
accepted answer prefixes against up to `max_new_tokens` generated tokens, which
lets hidden GPQA questions require only a one-letter answer while tolerating
tokenizer whitespace variants.

## Requirements

- Apple Silicon Mac, 24 GB+ unified memory (M2 or newer)
- macOS Sequoia or later
- Swift 6 through Xcode or Xcode Command Line Tools
- Xcode Metal Toolchain for `mlx.metallib`; `./setup.sh` tries
  `xcodebuild -downloadComponent MetalToolchain`, but users with only Command
  Line Tools may need full Xcode installed, opened once, and licensed with
  `sudo xcodebuild -license accept`
- CMake, installed by `./setup.sh` via Homebrew when missing and used by `tools/build-mlx-metallib.sh` to build `mlx.metallib`
- [mactop](https://github.com/metaspartan/mactop) — installed by `./setup.sh` via Homebrew when missing, or supplied with `MLXFAST_MACTOP_BIN=/path/to/mactop`; hardware bandwidth requires macOS IOReport DRAM channels
