# Bandwidth Measurement

## Overview

DRAM bandwidth is measured via Apple's IOReport hardware counters, read by `mactop`.
This is a hardware-level measurement — it captures all DRAM traffic regardless of
what generated it, and cannot be gamed from Python.

## Hardware Requirement

`mactop` is required. It only works on Apple Silicon (M1/M2/M3/M4).
The harness raises `RuntimeError` if mactop is not installed and no samples are produced.

Install: `brew install mactop`

Binary search order (first found wins):
1. `$MACTOP_BINARY` env var (if non-empty and file exists)
2. `shutil.which("mactop")` — PATH lookup
3. `/opt/homebrew/bin/mactop` — Homebrew default (ARM)
4. `/usr/local/bin/mactop` — Homebrew default (Intel)

**Risk**: if a non-Apple-Silicon `mactop` or a wrong binary is earlier on PATH,
it takes priority over the Homebrew binary. See [13-issues.md §7](13-issues.md).

## mactop Invocation

### Idle Baseline (blocking, before model load)

```bash
mactop --headless --count 30 --interval 100 --format json
```

- 30 samples × 100ms interval = 3 seconds
- Subprocess.run with `timeout=duration_s + 5`
- Computes `mean(dram_bw_combined_gbs)` across all positive samples
- Result stored as `idle_gbps`

### Decode Phase (streaming, background process)

```bash
mactop --headless --interval 100 --format json
```

- Started as a background subprocess (`Popen`) after the warmup forward pass
- Runs concurrently with the 512-token decode loop
- Terminated with `SIGTERM` after decode loop completes; 3-second drain timeout
- If SIGTERM is ignored: `SIGKILL` sent, then `communicate()` blocks until exit

## mactop Output Format

mactop v2.1.2 outputs a **JSON array**:
```json
[
  {"soc_metrics": {"dram_bw_combined_gbs": 11.78, ...}, ...},
  {"soc_metrics": {"dram_bw_combined_gbs": 11.82, ...}, ...},
  ...
]
```

Parsing logic (`bandwidth.py`):
```python
# Try JSON array first (mactop v2.x)
try:
    items = json.loads(stdout)
    if isinstance(items, list):
        for obj in items:
            bw = obj.get("soc_metrics", {}).get("dram_bw_combined_gbs", 0.0)
            if bw > 0.0:
                samples.append(float(bw))
        return samples  # early return on success
except (json.JSONDecodeError, TypeError, AttributeError):
    pass

# Fall back to NDJSON (one JSON object per line)
for line in stdout.splitlines():
    ...
```

The NDJSON fallback handles future mactop versions that may change format.
If both parse attempts fail to produce any samples: returns `[]`.

## Net Bandwidth Formula

```python
net_samples = [max(s - idle_gbps, 0.0) for s in samples if s > 0]
mean_gbps   = sum(net_samples) / len(net_samples)
total_gb    = mean_gbps × decode_elapsed_seconds
gb_per_tok  = total_gb / num_tokens      # num_tokens = 512
```

`decode_elapsed_seconds` comes from `mactop_session._elapsed`, set in `run.py`
immediately after the decode loop and before `bandwidth.measure()` is called.

### Sample Count

At 0.458 s/tok × 512 tokens = 234 seconds total decode time, and 100ms sample
interval → ~2340 samples. Excellent statistical coverage.

### Idle Subtraction

Per-sample subtraction rather than a single mean subtraction. Each sample has the
idle baseline removed independently, then floored at 0.0. This accounts for
momentary spikes in the idle baseline without distorting the mean.

## What Bandwidth Captures

The `dram_bw_combined_gbs` field measures all DRAM read+write traffic at the SoC level:

| Source | Approx GB/tok | Notes |
|---|---|---|
| Model params (attention, HC, gates, shared experts) | ~4.0 | In Metal, re-read every token |
| Expert reads (6 × 13.37 MB × 43 layers, cache miss) | up to 3.45 | Via page cache from SSD |
| KV cache traffic (512-step cache × MLA dims) | ~1.7 | Grows during decode |
| Activations + attention scores + residuals | ~2.6 | Intermediate GPU tensors |
| **Total** | **~11.78 GB/tok** | Matches mactop measurement |

Expert reads from SSD pass through the OS page cache into DRAM before GPU reads them.
All of this is captured by IOReport.

## Failure Mode

If `net_samples` is empty (mactop not found, returned no data, or all samples ≤ 0):
```python
raise RuntimeError(
    "mactop produced no usable bandwidth samples. "
    "Ensure mactop is installed (brew install mactop) and the machine is Apple Silicon."
)
```

The run fails; no score is written. There is no software fallback.
*(Earlier versions had a software estimate fallback; it was removed because it
produced systematically wrong values.)*

## Idle Baseline Risk

If another process spikes DRAM bandwidth during the 3-second idle window, `idle_gbps`
is inflated. Since it is subtracted per sample:
- Inflated idle → lower reported `bandwidth_gb_per_token` → artificially better score locally
- The competition server measures on a clean, dedicated machine

Always run `mlxfast run` with no other background workloads.
