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

The Swift transform writes benchmark-ready weights here:

```text
weights/
  config.json
  model.safetensors.index.json
  experts/manifest.json
```

Correctness cases live at:

```text
correctness_golden.json
```

## Editable Surface

The active implementation is Swift-only:

| Path | Scope |
|---|---|
| `Sources/MLXFastDeepSeek/` | DeepSeek V4 Flash runtime, attention, MoE, expert streaming, correctness, benchmark timing. |
| `Sources/MLXFastTransform/` | Offline safetensors transform and expert manifest generation. |
| `Sources/MLXFastCore/` | Shared constants, score schema, safetensors, golden-case loading. |
| `Sources/MLXFastCLI/` | Command-line entrypoint. |
| `tools/build-mlx-metallib.sh` | Local MLX Metal library build helper. |

There is no Python harness path.

## Correctness Gate

Correctness is a hard gate. For each golden case, the harness runs cached greedy
generation for 256 tokens with temperature-zero behavior and compares token IDs
exactly. The first mismatch records the case, step, expected token, and actual
token in the failed report.

The gate is intended as a first-stage filter: an implementation that fails it is
not eligible for the longer benchmark.

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
```
