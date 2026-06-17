# Peak RAM Measurement

## API

```python
mx.reset_peak_memory()   # reset the high-water mark to 0
# ... run the benchmark ...
peak_bytes = mx.get_peak_memory()   # returns int: bytes
```

MLX tracks the peak total allocated Metal memory since the last `reset_peak_memory()`.
This is the unified memory used by the GPU Metal allocator — the same memory
shared between CPU and GPU on Apple Silicon.

## Measurement Window

`mx.reset_peak_memory()` is called **before the warmup forward pass**, not before
the timed decode loop. The full sequence:

```
mx.reset_peak_memory()           ← reset
[warmup] make_cache()            ← allocates KV cache objects
[warmup] model(seed, cache)      ← allocates expert slot tensors, intermediate buffers
[warmup] mx.eval(model.parameters()) ← materializes all dense params in Metal
[decode] mactop.start()
[decode] make_cache()            ← NEW cache (reuses Metal allocator buffers)
[decode] 512-token AR loop       ← peak tracked here
peak_bytes = mx.get_peak_memory()
```

**Warmup allocations are included in peak.** The peak is NOT "clean decode peak" —
it reflects the initial allocation spike from the warmup call. However, the MLX
allocator reuses Metal buffers, so the decode loop does not allocate significantly
above the warmup peak. In practice, peak is dominated by warmup.

## What Contributes to Peak

| Component | Approx GB | Notes |
|---|---|---|
| Dense model params (attention, HC, gates, shared experts, embeddings, LM head) | ~5.8 GB | Materialized in warmup by `mx.eval(model.parameters())` |
| KV cache — 512 steps × MLA dims × 43 layers | ~1.7 GB | Grows during decode; peak = full 512-step state |
| Expert slot bank (128 slots × 13.37 MB) | ≈ 0 initially | Lazy mx.arrays; Metal allocated only when consumed by `gather_qmm` |
| Intermediate activation buffers | ~0.1 GB | Reused via MLX allocator; typically reuse warmup allocations |
| **Total (current)** | **~7.51 GB** | |

## Expert Slot Bank and Peak RAM

`ExpertSlotBank` stores lazy `mx.array`s (not numpy arrays). Metal is not allocated
until the array is consumed by `gather_qmm`. This means:
- Loading a record into the LRU does NOT immediately increase Metal peak
- Metal is allocated only when the expert is actually used in a forward pass
- After use, MLX retains the Metal buffer in its caching allocator (does not free to OS)
- LRU eviction (dropping the Python reference) does not free Metal — the allocator
  retains it for reuse
- Peak grows until the allocator's high-water mark stabilizes

At `SLOT_BANK_SIZE=128`: the first 128 unique experts ever activated across all layers
contribute their Metal allocations to peak. Subsequent re-activations (LRU hits) reuse
already-allocated buffers.

## mactop Start vs Peak RAM

mactop starts **after** the warmup but the peak RAM window starts **before** the warmup.
There is no conflict — these are independent measurements:
- Peak RAM: `mx.reset_peak_memory()` to `mx.get_peak_memory()` (includes warmup)
- Bandwidth: mactop session during decode only (excludes warmup)

## Exact Return Value

`peak_bytes = mx.get_peak_memory()` returns a Python `int` (exact bytes).
This is passed directly to `score.compute(peak_ram_bytes=peak_bytes)` which
divides by `1024**3` internally to get `peak_ram_gb`.

No intermediate float conversion. No truncation. Exact to the byte.

## Participant Impact

To reduce `peak_ram_GB`:
- **KV cache quantization** — `QuantizedKVCache` or `TurboQuantKVCache` reduce
  the ~1.7 GB KV component
- **Smaller `SLOT_BANK_SIZE`** — fewer expert Metal allocations (tradeoff: more
  LRU misses → more disk reads → higher bandwidth)
- **Smaller expert quantization** — 2-bit experts = 2× smaller Metal footprint
  per expert slot; requires `transform.py` changes
- **Weight offloading** — keep some dense params in system memory and page them
  in (advanced; may raise bandwidth)

The ~5.8 GB dense param floor is hard to reduce without changing quantization
of the non-expert weights (attention, HC, embeddings).
