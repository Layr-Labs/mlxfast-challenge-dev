# Latency Measurement

## Prefill Latency (`prefill_sec_per_token`)

### What Is Measured

Time to process a 512-token prompt in a single forward pass, divided by 512.

### Prompt

512 random token IDs drawn from the same seed as the correctness prompt
(same SHA256 of commit hash, different byte offset). Generated at startup,
reused across all timed calls.

### Procedure

```python
def _prefill_once():
    cache = model.make_cache()           # fresh cache every call
    out = model(prompt, cache=cache)
    mx.eval(out.logits)                  # force GPU completion

# 1 warmup (discarded)
_prefill_once()

# 2 timed runs
t0 = time.perf_counter(); _prefill_once(); t1 = time.perf_counter()
t2 = time.perf_counter(); _prefill_once(); t3 = time.perf_counter()

prefill_spt = mean(t1-t0, t3-t2) / 512
```

### Cache State

Each call creates a new `make_cache()`. This means 43 × (1–3 cache objects) are
created and garbage-collected per call. If `PoolingCache` allocates Metal arrays
on construction, this adds allocation overhead to the measured time.

### Timing Window

Includes:
- All 43 transformer layer forward passes
- 43 × `mx.eval(result)` + `mx.clear_cache()` calls per pass (mandatory for OOM prevention)
- Final `mx.eval(out.logits)`

`mx.clear_cache()` is called 43 times per prefill pass. It releases all currently-
cached but unreferenced Metal buffers back to the OS. Without it, 43 layers × ~3 GB
stacked tensors ≈ 129 GB would accumulate. This is mandatory but expensive (~43
OS interactions per pass), directly inflating `prefill_sec_per_token`.

### When It Runs

Prefill is measured **before** the decode phase. This means:
- Expert slot bank is cold (post-model-load state)
- Prefill pays the true disk-read cost for experts
- After prefill completes, the LRU is partially warm — decode benefits from this

This ordering is intentional: prefill latency reflects a cold-start scenario.

---

## Decode Latency (`decode_sec_per_token`)

### What Is Measured

Wall-clock time for 512 autoregressive decode steps, divided by 512.

### Prompt (Seed)

32 random token IDs (subset of the correctness prompt). Used as the initial
context. The 32-token prefill is included in decode step 1.

### Procedure

```python
# Warmup (not timed)
mx.reset_peak_memory()          # ← resets here, BEFORE warmup
cache = model.make_cache()
_ = model(32-token seed, cache=cache)
mx.eval(model.parameters())

# Start mactop background process
mactop = MactopSession(); mactop.start()

# Timed decode (NEW cache, 512 steps)
cache = model.make_cache()
t0 = time.perf_counter()

next_tok = mx.argmax(model(32-token seed, cache=cache).logits[0, -1:], axis=-1, keepdims=True)
mx.eval(next_tok)               # ← step 1: processes 32 tokens, not 1

for _ in range(511):
    out = model(next_tok, cache=cache)
    next_tok = mx.argmax(out.logits[0, -1:], axis=-1, keepdims=True)
    mx.eval(next_tok)           # ← steps 2–512: true single-token decode

elapsed = time.perf_counter() - t0
decode_spt = elapsed / 512
```

### Step 1 Inflation

Step 1 processes the full 32-token seed prompt (not 1 token). The extra ~31 tokens
add roughly `(step1_time - single_token_time) / 512` to the mean — approximately
negligible (< 0.1% at current throughput).

### Per-Step GPU Syncs

Each decode step has exactly **1 mandatory GPU→CPU sync** per transformer layer:
`sorted_idx.tolist()` in `StreamingSwitchGLU.__call__`. Over 43 layers:
- 43 GPU syncs × 512 steps = 22,016 total syncs per decode run
- Each sync drains the GPU command queue before disk I/O can begin
- This is the hard floor on decode latency — routing requires knowing
  which expert records to load before any `os.pread` call

The second `.tolist()` call in the same method reuses the already-materialized
`sorted_idx` (MLX caches the value), so it is O(1) and not an additional sync.

### Per-Step I/O

For each of 43 layers, up to 6 unique expert records are loaded via `os.pread`.
With `SLOT_BANK_SIZE=128` and autoregressive routing being fairly stable, most
are LRU cache hits (no I/O). Cold-start decode has higher miss rate.

### Timing Window

Includes:
- 43 GPU syncs per token (routing materialization)
- Expert slot bank LRU lookup + `os.pread` on cache miss
- Three `gather_qmm` calls per layer (gate, up, down projections)
- Attention (MLA: compression, KV cache update, attention scores)
- HyperConnection expand/collapse (Metal kernel)
- `mx.argmax` + `mx.eval` per step

Does NOT include:
- Warmup (before `t0`)
- mactop subprocess overhead
- Model loading
- `mx.clear_cache()` (not called during decode for N=1)

### Lazy Decode

During decode (N=1), `mx.eval` is called once per step on `next_tok`.
The entire 43-layer forward pass is a single lazy computation graph that is
executed atomically at `mx.eval(next_tok)`. No intermediate `mx.eval` or
`mx.clear_cache` per layer — this is the "lazy decode" optimization.

This contrasts with prefill (N=512) which must eval + clear per layer to
prevent ~129 GB accumulation.
