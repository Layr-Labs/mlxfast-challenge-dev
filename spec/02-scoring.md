# Scoring

## Formula

```
score = peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token
```

All four values are non-negative. Lower score is better.

## Axis Definitions

### `peak_ram_GB`

Peak unified memory used by the Metal GPU allocator during the measurement window.

- Measured by `mx.reset_peak_memory()` / `mx.get_peak_memory()` (MLX API).
- Reset occurs **before** the warmup forward pass.
- Warmup allocations (KV cache creation, first expert loads, parameter materialization)
  are included in peak — this is the true allocation footprint.
- Returned as raw integer bytes from `mx.get_peak_memory()`. Divided by `1024³`
  for the score formula. No intermediate float conversion.
- See [08-peak-ram.md](08-peak-ram.md) for the exact measurement window.

### `bandwidth_GB_per_token`

Mean DRAM bandwidth consumed per decoded token, measured by hardware IOReport counters.

- Measured by `mactop --headless --format json --interval 100` running as a background
  subprocess during the 512-token decode loop.
- Uses `soc_metrics.dram_bw_combined_gbs` (combined read + write in GB/s).
- Idle baseline (measured before model load) is subtracted per sample.
- Formula: `mean(max(sample - idle, 0)) × decode_elapsed_seconds / 512`
- This is a hardware measurement — it cannot be gamed from Python.
- See [06-bandwidth.md](06-bandwidth.md) for the full measurement spec.

### `decode_sec_per_token`

Wall-clock seconds per autoregressive decode token over a 512-token window.

- Single warm-up call, then 512-token AR decode, timed with `time.perf_counter`.
- Includes: GPU syncs (43 per token for routing materialization), SSD I/O for
  expert loads, attention, MoE dispatch, HyperConnection.
- Does NOT include: mactop sampling overhead (separate process), model loading.
- See [07-latency.md](07-latency.md) for timing window details.

### `prefill_sec_per_token`

Wall-clock seconds per token to process a 512-token prompt in a single forward pass.

- 1 warmup run + 2 timed runs; result is the mean of the 2 timed runs.
- Divided by 512 (the prompt length) to get per-token rate.
- Includes: per-layer `mx.eval()` + `mx.clear_cache()` (mandatory to prevent OOM).
- See [07-latency.md](07-latency.md) for timing window details.

## Score Semantics

### Passing vs Failing

- If the correctness gate fails: `score = inf`, `passed = false`.
- If mactop produces no samples: run fails with `RuntimeError` (no score written).
- If any metric is non-finite (NaN, inf): score is null in `score.json`.

### Leaderboard Ranking

The leaderboard uses the best (lowest) score from all **passing** runs submitted
by a participant. Non-passing runs are not ranked but are stored.

### Interpreting the Axes

The formula is a product — all four dimensions matter equally in log space.
A 2× improvement on any single axis halves the score:

| Axis | Current | Reduction target | Score impact |
|---|---|---|---|
| `peak_ram_GB` | 7.51 | 4.0 GB (50%) | ×0.53 |
| `bandwidth_GB_per_token` | 11.78 | 6.0 GB/tok (50%) | ×0.51 |
| `decode_sec_per_token` | 0.458 | 0.230 s/tok (50%) | ×0.50 |
| `prefill_sec_per_token` | 0.048 | 0.024 s/tok (50%) | ×0.50 |

Current best score: **1.9464** (7.51 × 11.78 × 0.458 × 0.048)

## Output Files

### `score.json`

Written after every run (pass or fail) in the following schema:

```json
{
  "score": 1.9464,          // null if failed or non-finite
  "passed": true,
  "metrics": {
    "peak_ram_gb": 7.51,
    "bandwidth_gb_per_token": 11.78,
    "decode_seconds_per_token": 0.458,
    "prefill_seconds_per_token": 0.048,
    "passed_correctness": true,
    "num_layers": 43,
    "first_failing_layer": null,  // decode step index (NOT layer index) if failed
    "max_abs_diff": 0.0,
    "bandwidth_source": "mactop_hardware",
    "error": "",
    "commit": "abc1234",
    "timestamp": "2026-06-16T00:00:00Z",
    "harness_hash": "abc123..."
  }
}
```

**Note**: `first_failing_layer` is misleadingly named. It is the decode **step** index
(0–255) at which the correctness check first failed, not a transformer layer index.

### `results.tsv`

Appended after every run. Columns:

```
timestamp  commit  note  peak_ram_gb  bandwidth_gb_per_tok  decode_sec_per_tok
prefill_sec_per_tok  score  passed  num_layers  first_failing_layer
max_abs_diff  bandwidth_source  harness_hash
```
