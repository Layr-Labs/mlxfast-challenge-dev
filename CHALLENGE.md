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
4. Measures prefill latency, 512-step greedy decode latency, MLX peak memory, and
   `mactop` hardware DRAM bandwidth.
5. Writes `score.json` in the Darkbloom-compatible schema.

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

Correctness cases are supplied by the benchmark operator and are intentionally
not committed to the public repo:

```text
correctness_golden.json
```

Use `MLXFAST_CORRECTNESS_GOLDEN_PATH=/path/to/correctness_golden.json` when the
file is provisioned outside the repository root.

That hidden fixture also contains the scored benchmark oracle: the 512-token
prefill prompt, expected prefill token, 32-token decode seed, expected seed
token, and expected tokens emitted by the timed decode loop. The benchmark only
accepts a speed score when those hidden token IDs match.

The organizer can generate that hidden fixture from a private prompt manifest:

```bash
.build/release/mlxfast-swift make-golden \
  --prompt-file private_prompts.json \
  --output correctness_golden.json
```

The private manifest contains named correctness prompts and one named benchmark
prompt:

```json
{
  "version": 1,
  "cases": [
    {"name": "hidden-0", "prompt_tokens": [1, 2, 3]}
  ],
  "benchmark": {"name": "timed-hidden", "prompt_tokens": [1, 2, 3]}
}
```

The arrays above are abbreviated. The benchmark prompt must contain at least 512
token IDs.

For local development only, participants can generate a self-consistent fixture
from their current transformed weights:

```bash
.build/release/mlxfast-swift make-golden --output local_correctness_golden.json
MLXFAST_CORRECTNESS_GOLDEN_PATH=local_correctness_golden.json ./benchmark.sh
```

This local fixture is useful for exercising the full harness and catching
obvious regressions, but it is not the organizer-owned hidden golden used for
scoring.

## Editable Surface

The active implementation is Swift-only:

| Path | Scope |
|---|---|
| `Sources/MLXFastModel/` | Participant-editable DeepSeek V4 Flash runtime, attention, MoE, expert streaming, and weight loading. |
| `Sources/MLXFastTransform/` | Offline safetensors transform and expert manifest generation. |
| `Sources/MLXFastHarness/` | Frozen correctness gate, benchmark timing, mactop/RAM measurement, and score assembly. |
| `Sources/MLXFastCore/` | Frozen shared constants, score schema, safetensors, and golden-case loading. |
| `Sources/MLXFastCLI/` | Frozen command-line entrypoint. |
| `tools/build-mlx-metallib.sh` | Local MLX Metal library build helper. |

There is no Python harness path.

The default Swift transform uses an expert byte-range manifest rather than
rewriting all routed experts into new expert-major files. This is deliberate: on
Blacksmith's 250 GB Apple-Silicon runners, keeping the reference checkpoint and a
full rewritten expert copy at the same time would exceed available disk. The
manifest baseline is still part of the optimization surface. Submissions may
extend both `MLXFastTransform` and `MLXFastModel` to produce and consume a custom
expert layout, provided the generated `weights/` are deterministic from the
frozen reference checkpoint and pass the trusted correctness gate.

## Correctness Gate

Correctness is a hard gate. For each golden case, the harness runs cached greedy
generation for 256 tokens with temperature-zero behavior and compares token IDs
exactly. The first mismatch records the case, step, expected token, and actual
token in the failed report.

The gate is intended as a first-stage filter: an implementation that fails it is
not eligible for the longer benchmark.

The timed benchmark path is validated separately against the hidden benchmark
oracle in the same golden fixture. The harness checks the prefill token, decode
seed token, and tokens generated inside the timed decode loop before accepting a
score.

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
.build/release/mlxfast-swift make-golden --output local_correctness_golden.json
.build/release/mlxfast-swift correctness
.build/release/mlxfast-swift preflight
.build/release/mlxfast-swift benchmark --score-path score.json
```
