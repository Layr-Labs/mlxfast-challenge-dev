# Submission Flow

## Current Status: STUB

`mlxfast submit` is not implemented. It prints the submission payload to stdout
but does not POST to any server. No leaderboard exists yet.

See [13-issues.md §2](13-issues.md).

---

## CLI Command

```bash
mlxfast submit --note "description" [--weights weights/]
```

### Current Behavior

1. Self-hash check (`_self_hash.verify()`)
2. Re-runs `transform.py` verification (same as `mlxfast run`)
3. Builds submission payload dict
4. If `MLXFAST_API_URL` is unset: **prints payload to stdout and exits** (stub)
5. If `MLXFAST_API_URL` is set: intended to POST to `{api_url}/submit` via httpx
   (not implemented — raises `typer.Exit(0)` after the print)

---

## Submission Payload Schema

Built by `_build_submission_payload(weights_path, note)`:

```python
{
    "transform_source": str,          # contents of transform.py (or null)
    "transform_hash": str,            # SHA-256 of transform.py (or null)
    "modifiable_hashes": {            # SHA-256 of each modifiable file
        "mlx_models/deepseek_v4/deepseek_v4.py": str,
        "mlx_models/deepseek_v4/language.py": str,
        "mlx_models/deepseek_v4/config.py": str,
        "mlx_models/deepseek_v4/hyper_connection.py": str,
        # Note: switch_layers.py is not included. See 13-issues.md §11
    },
    "weights_hashes": {               # SHA-256 of each .safetensors file in weights/
        "model-00001-of-00003.safetensors": str,
        ...
    },
    "harness_hash": str,              # compute_harness_hash()
    "note": str,
    "timestamp": float,               # unix timestamp
}
```

---

## Required Server Implementation

The following must be built before hosting:

### `POST /submit`

Accepts the payload above. Server responsibilities:

1. **Authenticate** the participant (API key in header)
2. **Verify harness hash** — reject if `harness_hash` does not match the expected value
   for the distributed wheel
3. **Verify transform reproducibility** — re-run `transform.py` (from `transform_source`)
   in the server's sandbox, compare SHA-256 of output to `weights_hashes`
4. **Verify architecture invariants** — check all 11 fields (not just 4)
   from the submitted `weights/config.json`
5. **Run correctness gate against frozen reference** — load frozen reference model
   and submitted model, run `correctness.check(frozen_ref, submitted_sub, server_prompt)`
   with a server-seeded secret unknown to the participant
6. **Run benchmark** — execute `mlxfast run` in an isolated environment with known
   hardware configuration (clean machine, no background workloads)
7. **Record result** — store all metrics, harness hash, transform hash, modifiable hashes
8. **Update leaderboard** — use best passing score per participant

### `GET /leaderboard`

Returns ranked list of best passing scores per participant.

### Harness Hash Distribution

Before distributing the `mlxfast` wheel to participants:
1. Build the wheel
2. Install it in a clean environment
3. Run `python -c "from mlxfast.harness.constants import compute_harness_hash; print(compute_harness_hash())"`
4. Set `MLXFAST_EXPECTED_HARNESS_HASH=<result>` in every participant's run environment
   and in the server's execution environment

---

## Leaderboard Ranking Rules

- Only **passing** runs (correctness gate = pass) are ranked
- Best (lowest) score per participant
- Tie-breaking: earlier submission wins
- Score must be finite (not `inf`, not `nan`)
- Non-passing runs are stored but not ranked; participants can see their own failed runs

---

## What `switch_layers.py` Is Not Included in Payload

`mlx_models/mlx_lm_shims/switch_layers.py` is not hashed in `modifiable_hashes`
despite being a primary optimization target. The server cannot verify changes to
it via the current payload. This should be fixed before hosting.
See [13-issues.md §11](13-issues.md).
