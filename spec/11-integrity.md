# Integrity Mechanisms

## 1. Harness Self-Hash

### What It Does

On every `mlxfast run` and `mlxfast submit`, the CLI calls `_self_hash.verify()`.

```python
def compute_harness_hash() -> str:
    h = hashlib.sha256()
    h.update(
        f"mlx={MLX_VERSION}\n"
        f"mlx-lm>={MLX_LM_MIN_VERSION}\n"
        f"mlx-lm<{MLX_LM_MAX_VERSION}\n"
        f"mlx-vlm={MLX_VLM_VERSION}\n"
        .encode()
    )
    for path in sorted(harness_root().rglob("*.py")):
        h.update(path.read_bytes())
    return h.hexdigest()
```

Covers: all `.py` files under `mlxfast/` (harness/, cli.py, _harness_runner.py,
_sandbox.py, _self_hash.py), plus all four version pin strings.

If the computed hash does not match `EXPECTED_HARNESS_HASH`, `verify()` raises.

### Local Bypass

`EXPECTED_HARNESS_HASH` is read from `os.environ.get("MLXFAST_EXPECTED_HARNESS_HASH", "")`.
If empty (always the case locally), the check returns immediately:

```python
if not expected:
    return   # ← always reached locally
```

Additionally: `if os.environ.get("MLXFAST_SKIP_HASH_CHECK") == "1": return` provides
a second bypass.

**Before hosting**: the server must compute `compute_harness_hash()` for the
distributed wheel and inject the result into `MLXFAST_EXPECTED_HARNESS_HASH`
in the environment of every participant's run. Without this, any participant can
modify `mlxfast/` freely.

### `.venv` Files

`harness_root()` is `mlxfast/`. The `.venv/` directory is at the repo root and
is NOT under `mlxfast/` — it is not included in the hash. However: any `.py` file
accidentally placed under `mlxfast/` (e.g., a temp file) changes the hash.

---

## 2. Architecture Invariant Check

Performed at the start of every run. Reads `weights/config.json` and asserts:

| Field | Required Value |
|---|---|
| `num_hidden_layers` | 43 |
| `n_routed_experts` | 256 |
| `num_experts_per_tok` | 6 |
| `vocab_size` | 129280 |

Raises `ValueError` on any mismatch.

### Gap: Only 4 of 11+ Fields Are Checked

The following architecture fields are NOT checked and can be silently changed:

| Field | Current Value | Risk if Changed |
|---|---|---|
| `hidden_size` | 4096 | Different model capacity; unfair comparison |
| `num_attention_heads` | 64 | Attention structure change |
| `num_key_value_heads` | 1 | MLA structure change |
| `head_dim` | 512 | Attention dims change |
| `q_lora_rank` | 1024 | Query compression change |
| `n_shared_experts` | 1 | Shared expert count change |
| `moe_intermediate_size` | 2048 | Expert hidden dim change |
| `intermediate_size` | 18432 | Shared expert hidden dim change |
| `hc_mult` | 4 | HyperConnection streams change |
| `num_hash_layers` | 3 | Routing behavior change |
| `sliding_window` | 128 | Attention window change |

A participant could reduce `moe_intermediate_size` to halve expert sizes while
maintaining the 256-expert count, passing the 4 invariant checks. This should
be blocked. See [13-issues.md §4](13-issues.md).

---

## 3. Transform Sandbox

### Purpose

Re-run `transform.py` in isolation to verify it produces the same output as the
submitted `weights/`. Prevents non-deterministic or side-effecting transforms.

### Sandbox Mechanism

`_sandbox.py` spawns a fresh subprocess with audit hooks installed via `sys.audit`:

| Hook | What It Blocks |
|---|---|
| `os.open`, `open`, `io.open` | Writes/appends outside `OUTPUT_DIR` (writable output is `weights/`) |
| `socket.*` | All network operations |
| `subprocess.*` | All subprocess spawning |
| `os.fork`, `os.forkpty` | All fork operations |
| `os.putenv`, `os.unsetenv` | Environment variable mutations |
| `ctypes.dlopen`, `ctypes.CDLL` | New native library loads |
| `time.*` | All time functions (frozen for determinism) |

Reads are allowed from: the reference weights dir, all `sys.path` entries, and
the participant's working directory.

### Residual Risk

**Pre-loaded native extensions bypass all hooks.** Python's `sys.audit` only
intercepts Python-level calls. A C extension that calls `libc` functions directly
(e.g., `open(2)`, `socket(2)`) is not audited. Any package imported by `transform.py`
that pre-loaded a native extension can perform arbitrary I/O.

This means the sandbox is trust-based, not adversarially secure. It protects
against accidental non-determinism and straightforward cheating, not a determined
adversary.

### Hash Comparison

```python
sandbox_hash = content_hash(sandbox_output_dir)   # SHA-256 of all files in weights/
submitted_hash = content_hash(submitted_weights)   # SHA-256 of current weights/

if sandbox_hash != submitted_hash:
    return False, f"Transform not reproducible: {sandbox_hash} != {submitted_hash}"
```

`content_hash` is a deterministic SHA-256 over all files and their paths in the
directory, sorted. If any byte differs, verification fails.

**Important**: if you manually edit `weights/` after running `transform.py`, the hash
will not match. Always re-run `python transform.py` after any change to the transform
logic.

---

## 4. Weight Provenance

The combination of transform verification + harness hash + architecture invariants
is intended to ensure:

1. The submitted weights are the deterministic output of the submitted `transform.py`
2. The `transform.py` ran on the reference checkpoint with no side channels
3. The harness code was not modified

Gaps in current implementation:
- Transform sandbox is not adversarially secure (pre-loaded extensions)
- Architecture invariants cover only 4 fields
- Harness hash enforcement requires server to set `MLXFAST_EXPECTED_HARNESS_HASH`
- No server-side correctness check against the frozen reference (see [09-correctness.md](09-correctness.md))
