# Correctness Gate

## Purpose

Verify that the participant's model produces numerically equivalent outputs
to the reference implementation. If the gate fails, `score = inf` and the
run is not ranked.

## Prompt Construction

```python
secret_bytes = hashlib.sha256(f"|{git_commit_hash}".encode()).digest()
# On the server: XOR secret_bytes with a server-supplied secret per-submission

prompt_tokens = [secret_bytes[i % 32] % vocab_size for i in range(PROMPT_SEED_PREFIX_LENGTH)]
# PROMPT_SEED_PREFIX_LENGTH = 32
```

The prompt is 32 token IDs derived from the SHA256 of the git commit hash.
On the competition server, a server-supplied secret is mixed in to prevent
hardcoding of the correctness prompt.

**Issue**: the prompt is 32 uniform-random token IDs, not natural language.
The spec described 5 NL + 3 code + 2 adversarial prompts. Random tokens do not
stress real-world model behavior. See [13-issues.md §5](13-issues.md).

## Three-Layer Check

Runs for `CORRECTNESS_STEPS = 256` autoregressive steps.

### Layer 1: Greedy Token Match

At every step:
```python
ref_next = mx.argmax(ref_logits[0, -1])
sub_next = mx.argmax(sub_logits[0, -1])
assert ref_next == sub_next
```

Both models must produce the same greedy next token at every step.
If they diverge: `passed = false`, `first_failing_layer = step_index`.

The decode loop feeds `ref_next` to both models (not `sub_next`). This means
both paths follow the reference token sequence even if the sub model would diverge.

### Layer 2: Hidden State Diff

At every step:
```python
h = hidden_list[-1]           # (B, L, hc_mult=4, H=4096) or (B, L, H)
if h.ndim == 4:
    h = h.mean(axis=2)        # (B, L, H) — mean over hc_mult streams
hidden = h[0, -1]             # (H=4096,)

max_diff = max(abs(h_ref - h_sub))
assert max_diff <= CORRECTNESS_EPSILON  # 5e-3
```

**Important details**:
- The hidden state is extracted **pre-norm** — before `self.norm(self.hc_head(h))`.
  The state that goes into `lm_head` is post-norm; this check is on the earlier state.
- The 4D HyperConnection hidden state `(B, L, hc_mult=4, H)` is **mean-reduced**
  over the `hc_mult=4` axis. Changes that affect individual HC streams but preserve
  their mean pass this check.
- Only the **final hidden state** (after the last transformer layer) is checked.
  The spec described per-layer hidden state checks at all 43 layer boundaries;
  this is not implemented. See [13-issues.md §8](13-issues.md).

### Layer 3: Top-K Logit Set

At every step:
```python
ref_topk = set(mx.argsort(-ref_logits[0, -1])[:CORRECTNESS_TOP_K].tolist())
sub_topk = set(mx.argsort(-sub_logits[0, -1])[:CORRECTNESS_TOP_K].tolist())
assert ref_topk == sub_topk
```

`CORRECTNESS_TOP_K = 10`. The set of the 10 highest-logit token IDs must match
exactly. Order within the set is not checked.

Uses `argsort` over the full 129,280-element logit vector (O(V log V)) rather
than `argpartition` (O(V)). Over 256 steps this is 256 full sorts. Fast enough
in practice but inefficient.

## Local vs Server Behavior

### LOCAL (current implementation)

`_load_models()` returns:
```python
return sub_model, sub_tokenizer, sub_model   # same object twice
```

`correctness.check(ref_model, sub_model, ...)` receives the **same Python object**
as both ref and sub. Separate KV caches are created for each call but both use
the same model parameters.

**Result**:
- All three checks trivially pass (the same computation produces identical outputs)
- `passed = true` always, regardless of actual model correctness
- `max_abs_diff = 0.0` always
- **A completely broken model passes locally**

This is the most critical pre-hosting issue.

### SERVER (required, not yet implemented)

The server must:
1. Load the frozen reference model from `mlxfast/reference_weights/DeepSeek-V4-Flash-4bit/`
   using the original (unmodified) `mlx_models/deepseek_v4/` code
2. Load the participant's submitted model
3. Run `correctness.check(frozen_ref, submitted_sub, server_seeded_prompt)`
4. Only accept the run if all three layers pass

The server prompt must use a server-supplied secret (XOR into the seed) that is
unknown to the participant before the submission deadline, to prevent prompt hardcoding.

## Numerical Tolerance

`CORRECTNESS_EPSILON = 5e-3`

This is described in the code as "generous to account for non-deterministic GPU
reduction order". The spec allows bfloat16 floating-point non-associativity
(reordering of operations) but not lossy approximation.

In practice, with self-consistency checking, `max_abs_diff` is always 0.0 locally.

## RunReport Fields

| Field | Type | Notes |
|---|---|---|
| `passed` | bool | True if all three layers pass for all 256 steps |
| `first_failing_layer` | int or None | Decode **step** index (0–255) of first failure; null if passed |
| `max_abs_diff` | float | Maximum hidden state diff across all steps and checked positions |
| `num_layers` | int | Always 43 (transformer layers, not correctness layers) |
