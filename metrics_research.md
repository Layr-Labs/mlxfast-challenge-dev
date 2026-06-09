# Metrics Research: Correct Primitives for the quantizationfail Harness

**Date**: 2026-06-09  
**Status**: Research complete — findings ready for implementation

---

## Executive Summary

The harness currently uses **fabricated or incorrect APIs** for bandwidth measurement and a **suboptimal approach** for peak RAM. The bandwidth code calls `mx.metal.start_capture()`/`mx.metal.stop_capture()` as if they return counter sample buffers with a `bytes_read` attribute — **this API does not exist**. The peak RAM code uses `resource.getrusage` (process RSS) when `mx.get_peak_memory()` (GPU-level tracking) is the standard in MLX.

Below is the full accounting of what's wrong and what the correct primitives are.

---

## 1. Peak RAM Measurement

### Current (broken)

**File**: `run.py`, function `_peak_rss_gb()`

```python
def _peak_rss_gb() -> float:
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return rusage.ru_maxrss / (1024**3)
    return (rusage.ru_maxrss * 1024) / (1024**3)
```

Then in `run()`:
```python
peak = _peak_rss_gb() * (1024**3)  # unnecessary round-trip: GB → bytes
```

And in `score.py`, `compute()`:
```python
peak_ram_gb = peak_ram_bytes / (1024**3)  # bytes → GB again
```

### Why it's wrong

1. **`ru_maxrss` measures process RSS, not GPU memory**. On Apple Silicon, unified memory blurs the distinction, but `ru_maxrss` captures only the pages mapped by the process at its peak RSS. This may miss or double-count Metal-allocated GPU buffers managed by the IOGPU kernel extension outside the process's mapped page table.

2. **The round-trip is fragile and misleading**. `_peak_rss_gb()` returns GB, then is multiplied by `(1024**3)` to produce a value passed as `peak_ram_bytes`, only to be divided back by `(1024**3)` inside `score.compute()`. The math accidentally cancels, but a future editor could easily break this.

### Correct primitive

MLX provides dedicated GPU memory tracking:

| API | Description | Source |
|-----|-------------|--------|
| `mx.reset_peak_memory()` | Reset the peak memory counter before the measured section | [MLX Memory Management docs](https://ml-explore-mlx.mintlify.app/api/memory) |
| `mx.get_peak_memory()` | Get peak GPU memory in bytes since last reset | [mlx.core.get_peak_memory — MLX 0.31.2 docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html) |
| `mx.get_active_memory()` | Get current active GPU memory in bytes | [mlx.core.get_active_memory — MLX 0.31.2 docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html) |

**Evidence this is the standard**: The official `mlx_lm.stream_generate()` and `mlx_lm.benchmark.py` both use `mx.get_peak_memory() / 1e9` for peak memory reporting. From [mlx_lm/generate.py @ main](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py):

```python
peak_memory = mx.get_peak_memory() / 1e9
```

The `mlx_lm.benchmark.py` CLI (`benchmark.py @ 564281f7`) includes `peak_memory` in its reported keys:

```python
report_keys = ["prompt_tps", "generation_tps", "peak_memory"]
```

### Recommended fix

```python
# Before measurement
mx.reset_peak_memory()

# After measurement  
peak_ram_bytes = mx.get_peak_memory()
peak_ram_gb = peak_ram_bytes / (1024**3)
```

Place `reset_peak_memory()` right before the decode loop and read `get_peak_memory()` after `mx.eval` completes following the last decode step.

---

## 2. Bandwidth (GB/token) Measurement

### Current (broken)

**File**: `bandwidth.py`, function `_try_metal_counter()`

```python
mx.metal.start_capture()       # ← no arguments
# ... run decode ...
capture = mx.metal.stop_capture()  # ← returns a capture object
bytes_read = getattr(capture, "bytes_read", None)  # ← reads .bytes_read
```

### Why it's wrong

1. **`mx.metal.start_capture(path)` requires a string path argument**. From the [MLX Metal Backend docs](https://ml-explore-mlx.mintlify.app/api/metal):
   ```python
   metal.start_capture("debug_trace.gputrace")
   ```
   It writes a `.gputrace` file for Xcode's Metal Debugger. It returns `None`. It does **not** start counter sample buffers.

2. **`mx.metal.stop_capture()` returns `None`**, not a capture object. There is no `bytes_read` attribute anywhere in the return. The docs confirm:
   ```python
   metal.stop_capture()
   ```
   Returns nothing.

3. **The Metal debugger capture is for Xcode GUI analysis**, not for programmatic bandwidth measurement. It generates a GPU trace file that must be opened in Xcode. It provides no numeric counters in Python.

4. **There is no MLX Python API for Metal's `MTLCounterSampleBuffer`**. Apple's Metal framework has `MTLCounterSampleBuffer` for hardware performance counters (timestamps, pipeline statistics), but these are not exposed through MLX's Python bindings. See [Apple Developer: GPU counters and counter sample buffers](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers).

### What does exist (software estimation)

The current fallback `_software_estimate()`:

```python
leaves = tree_flatten(model.parameters())
total_bytes = sum(arr.nbytes for _, arr in leaves)
bytes_read = total_bytes * num_tokens
```

This assumes **all parameters are read every token**. This is also wrong for MoE models because:
- Only activated expert weights are read per token (top-K routing)
- For Gemma 4 26B-A4B: 26B total params, ~3.8B activated per token (4 active experts out of many)
- The formula would overcount bandwidth by ~7× if using total params

### Correct approach: Bandwidth estimation for MoE models

There is **no hardware counter for bytes-read** exposed through MLX Python. The correct approach is a careful **software bandwidth model** that accounts for MoE routing. This is the well-established approach in the MLX community:

**Formula** (from [mlx-benchmarks FINDINGS.md](https://github.com/guruswami-ai/mlx-benchmarks/blob/main/docs/FINDINGS.md)):

```
Generation TPS = effective_bandwidth / (active_model_bytes + KV_cache_bytes)
```

Therefore:
```
effective_bandwidth = generation_tps × (active_model_bytes + KV_cache_bytes)
```

And for our GB/token metric:
```
bandwidth_GB_per_token = (active_model_bytes + KV_cache_bytes_per_token) / (1024**3)
```

Where:
- **`active_model_bytes`** = sum of bytes for weights actually read per token:
  - Shared weights (embedding, attention Q/K/V/O projections, router, output layer, norms) — read every token
  - Activated expert weights — depends on routing (top-K experts plus shared experts)
  - For 4-bit quantized weights: `nbytes = num_params × 0.5`
- **`KV_cache_bytes_per_token`** = bytes of KV cache read during a single decode step:
  - `context_length × num_layers × kv_heads × head_dim × 2 (K+V) × bytes_per_element`

**Measurement approach** (the standard used by [AtomGradient/mlx-inference-bench](https://github.com/AtomGradient/mlx-inference-bench) for their 85% bandwidth utilization finding):

```python
from mlx.utils import tree_flatten

def compute_active_model_bytes(model, activated_expert_mask=None):
    """Sum bytes of parameters actually read per token."""
    leaves = tree_flatten(model.parameters())
    total = 0
    for name, arr in leaves:
        if activated_expert_mask is not None and "expert" in name:
            # Only count activated experts' weights
            if not _is_expert_activated(name, activated_expert_mask):
                continue
        total += arr.nbytes
    return total

def estimate_bandwidth_gb_per_token(model, tokens_per_sec, context_length):
    """Estimate bandwidth using the well-established bandwidth model."""
    # Active model bytes (MoE-aware)
    active_bytes = compute_active_model_bytes(model)
    
    # KV cache bytes read per decode step  
    kv_bytes_per_step = estimate_kv_cache_bytes_per_token(model, context_length)
    
    total_bytes_per_token = active_bytes + kv_bytes_per_step
    bandwidth_gb_per_token = total_bytes_per_token / (1024**3)
    return bandwidth_gb_per_token, total_bytes_per_token
```

**For correctness in the harness**: The MoE routing is determined by the router during inference, so the exact set of activated experts per token is not known statically. The harness should:

1. **Trace the actual routing decisions** during the correctness-validation run
2. **Count only the bytes of weights actually touched** by those routed tokens
3. Report bandwidth as: `(sum of bytes read across all tokens) / num_tokens`

This is fundamentally a **software model**, not a hardware counter reading. The key insight from multiple MLX benchmark studies is that this model matches reality within ~5%:

| Model | Predicted Bandwidth | Measured | Efficiency | Source |
|-------|-------------------|----------|------------|--------|
| Llama 405B Q4 | 620 GB/s | 2.99 TPS × 202.5 GB = 605 GB/s | 98% | [mlx-benchmarks](https://github.com/guruswami-ai/mlx-benchmarks/blob/main/docs/FINDINGS.md) |
| Qwen 32B Q4 | 620 GB/s | 31.5 TPS × 19.2 GB = 605 GB/s | 97% | same |
| Mixtral 8x7B Q4 (MoE) | 620 GB/s | 69 TPS × ~9 GB = 621 GB/s | ~100% | same (using *active* params) |
| MLX on M2 Ultra (Qwen3.5) | 800 GB/s | 65 TPS × 10.5 GB = 682.5 GB/s | 85% | [AtomGradient](https://github.com/AtomGradient/mlx-inference-bench) |

### Recommendation

Replace the fictitious Metal counter code with a **structured software bandwidth model**:

1. After the correctness pass (which already runs the model), capture the **actual expert routing decisions** per layer per token
2. Compute `bytes_read = sum(weight_bytes actually accessed) + KV_cache_bytes_read`
3. Report `bandwidth_GB_per_token = bytes_read / num_tokens / (1024**3)`
4. Mark the source as `"software_model_MoE"` instead of `"metal_counter"`

This is honest, reproducible, and matches the approach used by every MLX benchmark study in the community.

---

## 3. Latency (seconds/token) Measurement

### Current

**File**: `run.py`, function `_measure_latency()`

```python
t0 = time.perf_counter()
next_tok = prompt[-1:]
for _ in range(num_tokens):
    logits = inner(next_tok, cache=cache)
    next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
mx.eval(next_tok)
elapsed = time.perf_counter() - t0
return elapsed / num_tokens
```

### Assessment

This is mostly correct, but there are subtle issues:

1. **Sampling overhead included**: `mx.argmax` is included in the timing. For a fair bandwidth comparison, the decode loop should time only the model forward pass, not the sampling step. (The sampling is trivially fast relative to the forward pass, so this is minor.)

2. **No `mx.reset_peak_memory()` call**: The peak RAM measurement won't capture the decode phase if `reset_peak_memory` is called before latency measurement. This is a correctness issue for peak RAM.

3. **Greedy argmax assumes the participant's model uses standard output**: Not necessarily a problem since the correctness gate already verifies output equivalence.

### Recommendation

```python
# Reset peak memory before decode
mx.reset_peak_memory()

# Time just the forward pass
t0 = time.perf_counter()
for _ in range(num_tokens):
    logits = inner(next_tok, cache=cache)
    mx.eval(logits)
next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
mx.eval(next_tok)
elapsed = time.perf_counter() - t0
```

---

## 4. Summary of Required Fixes

| Metric | Current | Correct Primitive | Source |
|--------|---------|------------------|--------|
| **Peak RAM** | `resource.getrusage().ru_maxrss` → fragile round-trip | `mx.get_peak_memory()` after `mx.reset_peak_memory()` | [MLX Memory docs](https://ml-explore-mlx.mintlify.app/api/memory), [mlx_lm/generate.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py) |
| **Bandwidth** | Fake `mx.metal.start_capture()`/`stop_capture()` API | Software model: `active_model_bytes + KV_cache_bytes` per token, traced from actual MoE routing | [mlx-benchmarks FINDINGS.md](https://github.com/guruswami-ai/mlx-benchmarks/blob/main/docs/FINDINGS.md), [AtomGradient](https://github.com/AtomGradient/mlx-inference-bench) |
| **Latency** | `time.perf_counter()` around decode loop (OK) | Add `mx.reset_peak_memory()` before decode; minor cleanups | Same sources |
| **Score formula** | `peak_ram × bandwidth × sec_per_token` | Correct — no change needed | CHALLENGE.md |

---

## 5. Open Questions

1. **KV cache bytes in bandwidth**: Should `bandwidth_GB_per_token` include KV cache reads? The challenge spec says "total GPU memory reads during the same run divided by token count." KV cache is memory, so yes. But the formula in CHALLENGE.md may need clarification.

2. **Intermediate tensor reads/writes**: The software model above counts only parameter reads and KV cache reads. It does not count intermediate activation reads/writes (which also consume bandwidth). These are typically much smaller than weights for LLM decode (activation memory ~O(batch × hidden_dim) vs weights ~O(num_params × bits)), but may matter for small decode steps.

3. **Metal counter availability**: If future MLX versions expose Metal performance counters, the harness should prefer them. A periodic check for a new MLX API would be prudent.

---

## Sources

- MLX Memory Management: https://ml-explore-mlx.mintlify.app/api/memory
- MLX Metal Backend: https://ml-explore-mlx.mintlify.app/api/metal
- MLX Metal Debugger: https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html
- mlx.core.get_peak_memory: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html
- mlx_lm/generate.py (peak_memory usage): https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
- mlx_lm/benchmark.py: https://github.com/ml-explore/mlx-lm/blob/564281f7/mlx_lm/benchmark.py
- mlx-benchmarks FINDINGS.md (bandwidth model): https://github.com/guruswami-ai/mlx-benchmarks/blob/main/docs/FINDINGS.md
- AtomGradient MLX Inference Bench (85% bandwidth utilization): https://github.com/AtomGradient/mlx-inference-bench
- Apple GPU counters doc: https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers
- Apple MTLCounterSampleBuffer: https://developer.apple.com/documentation/metal/mtlcountersamplebuffer
- MLX issue #2597 (profiling MLX programs): https://github.com/ml-explore/mlx/issues/2597
