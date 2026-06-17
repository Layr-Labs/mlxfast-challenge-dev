# Competition Overview

## Goal

Run DeepSeek V4 Flash (a 256-expert mixture-of-experts LLM) on Apple Silicon
as efficiently as possible along four simultaneously measured axes:
peak memory, memory bandwidth, decode latency, and prefill latency.

## Scoring

```
score = peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token
```

Lower is better. All four axes are measured by the frozen harness in a single run.
Optimizing one axis at the expense of another is fully penalized — the product
means no free lunches.

See [02-scoring.md](02-scoring.md) for the full formula definition.

## What Participants Submit

1. **`transform.py`** — an offline weight conversion script that runs once on the
   reference checkpoint and writes participant-formatted weights to `weights/`.
2. **`weights/`** — the output of `transform.py`. All dense layers plus expert
   binary files in the format the participant's model code expects.
3. **`mlx_models/deepseek_v4/`** — the model implementation. Participants may
   replace any of the four modifiable files:
   - `deepseek_v4.py` — top-level Model class
   - `language.py` — all layer logic (attention, MoE, HyperConnection)
   - `config.py` — ModelConfig dataclass
   - `hyper_connection.py` — HyperConnection residual mixing + Metal kernel
4. **`mlx_models/mlx_lm_shims/switch_layers.py`** — expert streaming engine
   (`ExpertSlotBank` LRU + `StreamingSwitchGLU`). Primary optimization target.

## What Participants Cannot Change

Everything under `mlxfast/` is the frozen harness:
- `mlxfast/harness/` — measurement pipeline (run, bandwidth, correctness, score)
- `mlxfast/cli.py` — the `mlxfast run` command
- `mlxfast/_harness_runner.py` — subprocess spawner
- `mlxfast/_sandbox.py` — transform.py verification sandbox
- `mlxfast/_self_hash.py` — harness integrity check

The harness SHA-256 hashes its own `.py` files at startup. Any modification
causes `mlxfast run` to refuse (enforcement is server-side; the local check
is advisory — see [11-integrity.md](11-integrity.md)).

## Constraints on Participant Code

1. **Correctness gate must pass** — the model's outputs must match the reference
   implementation within numerical tolerance. Score = ∞ if it fails.
   See [09-correctness.md](09-correctness.md).

2. **`transform.py` must be deterministic** — the harness re-runs it in a sandbox
   and compares SHA-256 of the output to the submitted `weights/`. Non-deterministic
   transforms are rejected.

3. **`transform.py` must not access the network** — the sandbox blocks all network
   calls, subprocess spawns, and environment mutations.

4. **Architecture invariants must be preserved** — `num_hidden_layers=43`,
   `n_routed_experts=256`, `num_experts_per_tok=6`, `vocab_size=129280` are
   checked by the harness. Changing them causes immediate failure.
   *(See [13-issues.md §4](13-issues.md) — only 4 of 11+ invariants are checked.)*

5. **`transform.py` input is the reference checkpoint** — participants receive the
   reference checkpoint at `mlxfast/reference_weights/DeepSeek-V4-Flash-4bit/`.

## Hardware Requirements

- Apple Silicon Mac (M1/M2/M3/M4 family) — required for Metal + mactop IOReport
- Unified memory: minimum 24 GB recommended (current optimized run uses 7.5 GB peak;
  the reference (unoptimized) run requires all model weights in memory)
- `mactop` v2.x installed (Homebrew: `brew install mactop`) — **required** for
  bandwidth measurement; harness raises `RuntimeError` if absent
- Python 3.11+, mlx 0.31.2, mlx-lm ≥0.31.2 <0.32.0, mlx-vlm 0.6.3

## Run Command

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/mlxfast run --note "description" [--skip-transform-verify]
```

`--skip-transform-verify` skips the sandbox re-run of `transform.py` (saves ~30 min
when weights are unchanged). Always run without this flag before submitting.

## Leaderboard

*(Not yet implemented — see [12-submission.md](12-submission.md) and
[13-issues.md §2](13-issues.md).)*

Score is defined as the best passing (correctness gate = pass) run submitted.
