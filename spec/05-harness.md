# Harness Pipeline

## Entry Point

`mlxfast run` → `cli.py:run()` → `_harness_runner.run_in_subprocess()` →
spawns a fresh Python subprocess → `harness/run.py:run()`

The subprocess isolation ensures:
- Fresh Python interpreter (no state from previous runs)
- Independent `getrusage`-tracked peak RSS
- MLX patch state from prior imports does not leak
- `sys.path` is clean except for the participant's working directory

## CLI Phase (`cli.py`)

1. **Self-hash check** (`_self_hash.verify()`)
   Computes SHA-256 of all `.py` files under `mlxfast/` plus version pin strings.
   Compares against `MLXFAST_EXPECTED_HARNESS_HASH` env var.
   If env var is empty (always the case locally and until server sets it): **skipped**.

2. **Transform verification** (unless `--skip-transform-verify`)
   If `transform.py` exists:
   - Calls `_sandbox.verify_transform(transform.py, reference_weights/, weights/)`
   - Re-runs `transform.py` in a sandboxed subprocess with audit hooks
   - Compares SHA-256 of sandbox output to current `weights/` contents
   - Fails if they differ
   - Prints elapsed time and hash prefix on success

3. **Spawn subprocess**
   `_harness_runner.run_in_subprocess(weights_path, note, secret)`:
   - Formats an inline Python script string
   - Spawns `python -c <script>` with 1-hour hard timeout
   - Parses `__MLXFAST_RESULT__<JSON>` from stdout
   - Returns a dict of 14 TSV columns on success; error dict on failure

4. **Write outputs**
   - Appends row to `results.tsv`
   - Writes `score.json`
   - Prints rich table

## Subprocess Phase (`harness/run.py`)

### Step 1: Architecture Invariant Check

```python
_check_architecture_invariants(weights_path)
```

Reads `weights/config.json` and asserts:
- `num_hidden_layers == 43`
- `n_routed_experts == 256`
- `num_experts_per_tok == 6`
- `vocab_size == 129280`

Raises `ValueError` on mismatch. **Only 4 of 11+ architecture fields are checked.**
See [11-integrity.md §invariants](11-integrity.md).

### Step 2: Idle Bandwidth Baseline

```python
idle_gbps = bandwidth.measure_idle_bandwidth(duration_s=3)
```

Runs `mactop --headless --count 30 --interval 100 --format json` (3 seconds, 30 samples).
Computes mean `dram_bw_combined_gbs` across samples. This is subtracted from each
decode-phase sample later.

**Critical**: run on a quiet machine. A spiking background process here inflates
`idle_gbps` and artificially lowers reported bandwidth (since it is subtracted).

### Step 3: Model Load

```python
sub_model, sub_tokenizer, ref_model = _load_models(weights_path)
```

Execution:
1. `importlib.import_module("mlx_models.deepseek_v4.deepseek_v4")` — loads participant's Model class
2. `sys.modules["mlx_vlm.models.deepseek_v4"] = participant_mod` — monkey-patches mlx_vlm's model registry
3. `_force_sanitize_load(mlx_vlm.load, ...)` — wraps `safe_open` to strip `"format": "mlx"` metadata,
   forces `sanitize()` key remapping, also patches `load_config` for quant key prefixes
4. `model.eval()` + `mx.eval(model.parameters())` — materializes all dense params in Metal
5. Returns `(sub_model, sub_tokenizer, sub_model)` — **the same object for both ref and sub**

**Critical gap**: ref_model and sub_model are the same Python object. Local correctness
check is self-consistency only. See [09-correctness.md](09-correctness.md) and
[13-issues.md §1](13-issues.md).

### Step 4: Correctness Gate

```python
result = correctness.check(ref_model, sub_model, prompt_tokens)
```

Prompt construction:
```python
seed = SHA256(f"|{git_commit_hash}")  # XOR with server secret if provided
prompt = [seed_bytes[i] % vocab_size for i in range(32)]  # 32 random token IDs
```

Three-layer check (256 autoregressive steps):
1. **Token match**: greedy argmax tokens from ref and sub must match at every step
2. **Hidden state diff**: `max(abs(h_ref - h_sub))` ≤ `CORRECTNESS_EPSILON = 5e-3`
3. **Top-K logits**: `set(top-10 logits ref) == set(top-10 logits sub)`

If any layer fails: `score = inf`, `passed = false`, `first_failing_layer = step_index`.

See [09-correctness.md](09-correctness.md) for full details.

### Step 5: Prefill Latency

```python
prefill_spt = _measure_prefill_latency(sub_model, prefill_prompt)
```

- Prompt: 512 random tokens (seeded from same commit hash, different offset)
- 1 warmup + 2 timed `model(prompt, cache=make_cache())` calls
- Each timed call creates a fresh cache to avoid cache-state interference
- `mx.eval(out.logits)` after each call to force completion
- `prefill_spt = mean(t1, t2) / 512`

Prefill runs with a **cold** expert slot bank (post-model-load state).
This loads many experts into the LRU, warming it for the subsequent decode phase.
This is intentional — decode benefits from prefill-warmed slots.

### Step 6: Decode Latency + Peak RAM

```python
decode_spt, peak_bytes, mactop_session = _measure_latency_and_memory(sub_model, prompt)
```

Exact sequence:
```
mx.reset_peak_memory()                        # ← reset BEFORE warmup

# Warmup (not timed, not bandwidth-measured)
cache = model.make_cache()
_ = model(32-token prompt, cache=cache)
mx.eval(model.parameters())                   # ← materializes all dense params

# Start mactop AFTER warmup
mactop = MactopSession()
mactop.start()                                # ← background subprocess

# Timed decode (512 tokens AR, NEW cache)
cache = model.make_cache()
t0 = time.perf_counter()
next_tok = mx.argmax(model(32-token prompt, cache=cache).logits[0,-1:])
mx.eval(next_tok)                             # ← step 1: includes 32-token prefill
for _ in range(511):
    out = model(next_tok, cache=cache)
    next_tok = mx.argmax(out.logits[0,-1:])
    mx.eval(next_tok)
elapsed = time.perf_counter() - t0

mactop._samples = mactop.stop()               # ← SIGTERM + 3s drain
mactop._elapsed = elapsed

peak_bytes = mx.get_peak_memory()             # ← raw int bytes
decode_spt = elapsed / 512
```

**Note**: Step 1 of the decode loop processes the 32-token seed prompt (not 1 token).
This slightly inflates `decode_spt` for the first step. Over 512 tokens the effect
is (step1_extra / 512) ≈ negligible.

### Step 7: Bandwidth Measurement

```python
bw = bandwidth.measure(mactop_session, num_tokens=512, decode_elapsed=elapsed, idle_gbps=idle_gbps)
```

See [06-bandwidth.md](06-bandwidth.md) for the full formula.

### Step 8: Score Computation

```python
sr = score.compute(
    peak_ram_bytes=peak_bytes,
    bandwidth_gb_per_token=bw,
    decode_seconds_per_token=decode_spt,
    prefill_seconds_per_token=prefill_spt,
    passed=result.passed,
)
```

`score.compute` divides `peak_ram_bytes / 1024³` internally to get `peak_ram_gb`.

### Step 9: Return Result

```python
report = RunReport(
    timestamp=..., commit=..., note=...,
    peak_ram_gb=..., bandwidth_gb_per_tok=...,
    decode_sec_per_tok=..., prefill_sec_per_tok=...,
    score=..., passed=...,
    num_layers=43, first_failing_layer=..., max_abs_diff=...,
    bandwidth_source="mactop_hardware",
    harness_hash=compute_harness_hash(),
)
print("__MLXFAST_RESULT__" + json.dumps(report.to_tsv_row().split("\t")))
```

The harness runner (parent process) scans stdout for the `__MLXFAST_RESULT__` prefix
and parses the JSON list into a dict keyed by the 14 TSV column names.
