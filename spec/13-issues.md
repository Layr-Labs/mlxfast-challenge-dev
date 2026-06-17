# Open Issues — Must Resolve Before Hosting

Each issue is categorized: **BLOCKER** (competition broken without fix),
**HIGH** (significant fairness or integrity risk), **MEDIUM** (correctness
or UX risk), **LOW** (cosmetic or edge case).

---

## BLOCKER Issues

### 1. Local correctness check is self-consistency only

**File**: `mlxfast/harness/run.py:_load_models()`
**What happens**: `_load_models()` returns `(sub_model, tokenizer, sub_model)` — the
same Python object as both reference and submission model. `correctness.check(ref, sub)`
runs the same model twice. All three check layers trivially pass (`max_abs_diff = 0.0`).

**Impact**: A participant can completely remove the model's computation (return zeros),
and the local harness reports `passed: true, max_abs_diff: 0.0`. There is no local
correctness validation.

**Required fix**: The server must load the frozen reference model separately and run
`correctness.check(frozen_ref, submitted_sub, server_seeded_prompt)`. Locally the
self-consistency behavior is acceptable as a dev convenience, but the server must
do a real check.

**Additionally**: the server must not share the correctness prompt secret before
the submission deadline, to prevent prompt hardcoding.

---

### 2. `mlxfast submit` is a stub — no server exists

**File**: `mlxfast/cli.py:submit()`
**What happens**: If `MLXFAST_API_URL` is unset, `submit` prints the payload and exits.
If set, it exits without POSTing (the httpx call is a comment marked "out of scope").
No leaderboard server exists.

**Impact**: Participants cannot submit. The competition cannot run.

**Required fix**: Implement the server API (`POST /submit`, leaderboard) per the
spec in [12-submission.md](12-submission.md). Wire `mlxfast submit` to actually
POST to the server.

---

### 3. `EXPECTED_HARNESS_HASH` is never set — harness integrity is inert

**File**: `mlxfast/_self_hash.py`, `mlxfast/harness/constants.py`
**What happens**: `EXPECTED_HARNESS_HASH = os.environ.get("MLXFAST_EXPECTED_HARNESS_HASH", "")`.
Empty string → `_self_hash.verify()` returns immediately. No participant has ever
had their harness hash verified.

**Impact**: Any participant can modify `mlxfast/` code (alter measurement logic,
e.g., halve the decode loop count) without detection.

**Required fix**: Before distributing the wheel, compute `compute_harness_hash()`
and embed the value. See [12-submission.md §harness-hash](12-submission.md).

---

## HIGH Issues

### 4. Architecture invariant check covers only 4 of 11+ fields

**File**: `mlxfast/harness/run.py:_check_architecture_invariants()`
**What happens**: Only `num_hidden_layers`, `n_routed_experts`, `num_experts_per_tok`,
`vocab_size` are verified. All other architecture fields are unchecked.

**Impact**: A participant could set `moe_intermediate_size=512` (quarter-size experts,
4× smaller records) while keeping the 4 checked fields at their required values.
Expert weights would be 4× smaller, bandwidth 4× lower — but it's a fundamentally
different model.

**Fields that should be added to the invariant check**:
```
hidden_size          = 4096
num_attention_heads  = 64
head_dim             = 512
q_lora_rank          = 1024
num_key_value_heads  = 1
n_shared_experts     = 1
moe_intermediate_size = 2048
intermediate_size     = 18432
hc_mult               = 4
num_hash_layers       = 3
sliding_window        = 128
```

---

### 5. Correctness prompt is uniform-random tokens, not natural language

**File**: `mlxfast/harness/run.py:_build_prompt()`
**What happens**: Prompt is 32 random token IDs (`secret_bytes[i] % vocab_size`).
Random token IDs do not form valid UTF-8, have no linguistic structure, and do not
stress attention, MoE routing, or precision on real inputs.

**Impact**: A participant could implement a model that behaves correctly only on
random-token inputs but produces garbage on natural language (e.g., by hardcoding
a lookup table). The spec called for "5 NL + 3 code + 2 adversarial" prompts.

**Required fix**: Replace the random-token prompt with 5–10 actual text prompts
encoded with the DeepSeek V4 tokenizer. Include at least one adversarial prompt
designed to catch common shortcuts (e.g., repeated tokens, very long context,
mixed language).

---

### 6. No server-side correctness reference run is specified

**Depends on issue #1.** There is no documented protocol for how the server should
run the correctness check: which Python environment, which hardware, which version
of the reference model, what tolerance to use, whether to use the same 3-layer
check or a stricter one.

**Required fix**: Document the server's correctness check procedure, including:
- Hardware (Apple Silicon, which chip)
- Python/MLX/mlx-lm/mlx-vlm versions
- Whether the check uses the same `CORRECTNESS_EPSILON = 5e-3` or stricter
- Whether `num_layers` and `num_steps` (currently 43 and 256) are sufficient

---

### 7. `mactop` is a hard runtime dependency with no setup documentation

**File**: `mlxfast/harness/bandwidth.py`
**What happens**: If `mactop` is not installed, bandwidth measurement raises
`RuntimeError`. There are no setup docs, no `requirements.txt` entry, no graceful
error message pointing to `brew install mactop`.

**Impact**: First-time participants cannot run the harness. The error message is
cryptic.

**Required fix**:
- Add `mactop` to setup docs / README as a required install
- Improve the error message to include install instructions
- Consider: `_find_mactop_binary()` could print a helpful error before raising

---

## MEDIUM Issues

### 8. Per-layer hidden state check not implemented

**File**: `mlxfast/harness/correctness.py`
**What happens**: Only the **final** hidden state (after layer 42) is checked
against `CORRECTNESS_EPSILON`. The spec described checks at all 43 layer
boundaries.

**Impact**: A participant could introduce per-layer numerical drift that stays
below epsilon at the final layer but accumulates to real divergence on longer
sequences or different prompts.

**Recommended fix**: Check the hidden state at every layer boundary (or at least
at a sample of them). This requires the model to support per-layer hidden state
output, or the correctness check to use model hooks.

---

### 9. Decode step 1 includes 32-token prompt processing

**File**: `mlxfast/harness/run.py:_measure_latency_and_memory()`
**What happens**: The timed decode window starts at `t0 = time.perf_counter()`.
Step 1 calls `model(32-token seed, cache)` — processing 32 tokens, not 1.
Steps 2–512 are true single-token decode.

**Impact**: `decode_spt = elapsed / 512` includes the cost of one 32-token prefill
call, slightly inflating the reported decode latency. At current throughput, the
extra cost is roughly `(prefill_32_tokens - one_token) / 512 ≈ 0.002 s/tok` (< 0.5%).

**Recommended fix**: Use a pre-tokenized prompt of exactly 1 token as the seed, or
start timing after the first (seed prefill) step.

---

### 10. Transform sandbox is not adversarially secure

**File**: `mlxfast/_sandbox.py`
**What happens**: `sys.audit` hooks only intercept Python-level calls. Pre-loaded
C extensions that directly call libc can read any file, open sockets, etc., without
triggering the audit hooks.

**Impact**: A determined attacker could write `transform.py` that loads a pre-existing
C extension (e.g., NumPy's C code) and uses ctypes-style tricks to perform arbitrary
I/O. In practice, the sandbox blocks naive non-determinism and obvious network calls.

**Acceptable for now**: The sandbox is a deterrent, not a security boundary. Document
this limitation clearly for participants. The server's re-run on isolated hardware
is the real enforcement.

---

### 11. `switch_layers.py` is not included in the submission payload

**File**: `mlxfast/cli.py:_build_submission_payload()`
**What happens**: `modifiable_hashes` only covers the four files in
`mlx_models/deepseek_v4/`. `mlx_models/mlx_lm_shims/switch_layers.py` — the primary
optimization target — is not hashed.

**Impact**: The server cannot verify changes to `switch_layers.py` from the payload.
The server would need to re-run the model to detect changes indirectly via benchmark
results.

**Recommended fix**: Add `switch_layers.py` (and the entire `mlx_models/` directory)
to the hashed file list in `_build_submission_payload`. Consider hashing all `.py`
files under `mlx_models/` rather than a hardcoded list.

---

### 12. `--note` with `{` or `}` characters crashes the harness runner

**File**: `mlxfast/_harness_runner.py:SUBPROCESS_SCRIPT.format(...)`
**What happens**: `SUBPROCESS_SCRIPT` is a Python format string. The `note` parameter
is inserted via `{note!r}`. If `note` itself contains `{` or `}` (e.g., `--note "test {foo}"`),
Python's `.format()` attempts to interpret the braces and raises `KeyError`.

**Workaround**: Use plain text notes (no curly braces).

**Fix**: Escape the format string template or use a different injection mechanism
(e.g., environment variable, temp file, JSON-encoded argument).

---

### 13. `first_failing_layer` field name is misleading

**File**: `mlxfast/harness/run.py:RunReport`, `mlxfast/cli.py`
**What happens**: `first_failing_layer` stores the **decode step index** (0–255) at
which correctness first failed, not the transformer layer index (0–42).

**Impact**: When debugging a correctness failure, `first_failing_layer=5` means the
model diverged at decode step 5, not at transformer layer 5. The name misleads.

**Recommended fix**: Rename to `first_failing_step` in `RunReport`, `results.tsv`
header, `score.json`, and the CLI table.

---

## LOW Issues

### 14. Prefill measurement runs before decode — slot bank is warm for decode

The prefill measurement (512-token forward pass) is run before the decode measurement.
This loads many (layer, expert) pairs into the LRU, warming the slot bank for decode.
Decode performance is slightly better than cold-start because of this ordering.

This is **intentional by design** (the comment in the code confirms it), but it means
reported `decode_sec_per_token` reflects a slot-bank-warm scenario, not a cold-start.
For consistency across participants and runs on the server, this is fine as long as
the ordering is always the same.

---

### 15. `results.tsv` header is not validated against existing file

If a participant upgrades the harness and the TSV column set changes, `results.tsv`
will have mixed headers. No validation or migration is performed.

---

### 16. 43 file descriptors opened, never closed during run

`ExpertSlotBank._open_fd()` opens one FD per layer `.bin` file (43 total). The
`close()` method exists but is never called by the harness. FDs are released when
the subprocess exits. Not a problem for single-run benchmarks; would be an issue
for long-lived processes.

---

### 17. Global singleton slot bank breaks multi-model scripts

`_configure_streaming()` creates/reuses a module-level global `ExpertSlotBank`.
If a participant's custom script loads two models in the same process (not done
by the harness), they share the same slot bank and the second model loads wrong
expert records.

---

### 18. `MactopSession._elapsed` set after `stop()` is called

In `run.py`, `mactop._elapsed = elapsed` is set after `mactop._samples = mactop.stop()`.
This is a fragile ordering — the `_elapsed` attribute must be set before
`bandwidth.measure()` is called, and it is, but the pattern is unusual and
easy to accidentally break when refactoring.
