# mlxfast Challenge — Spec Index

This folder is the authoritative spec for every dimension of the competition.
Use it to audit implementation gaps before hosting.

## Files

| File | Covers |
|---|---|
| [01-competition.md](01-competition.md) | Rules, participant surface, what is scored |
| [02-scoring.md](02-scoring.md) | Exact formula, axis definitions, edge cases |
| [03-model.md](03-model.md) | DeepSeek V4 Flash architecture, all constants |
| [04-weights.md](04-weights.md) | Reference + transformed weight format, manifest contract |
| [05-harness.md](05-harness.md) | Full measurement pipeline, step by step |
| [06-bandwidth.md](06-bandwidth.md) | mactop integration, net bandwidth formula |
| [07-latency.md](07-latency.md) | Prefill and decode timing windows, warmup |
| [08-peak-ram.md](08-peak-ram.md) | Peak RAM measurement, what is and isn't included |
| [09-correctness.md](09-correctness.md) | 3-layer correctness gate, thresholds, local vs server |
| [10-surface.md](10-surface.md) | Modifiable surface, participant contracts |
| [11-integrity.md](11-integrity.md) | Harness hash, arch invariants, transform sandbox |
| [12-submission.md](12-submission.md) | Submit flow, server API contract, payload schema |
| [13-issues.md](13-issues.md) | All open issues — must-fix before hosting |

---

## Critical Issues Before Hosting (summary)

These are the highest-severity gaps. Full details in [13-issues.md](13-issues.md).

### Must Fix

1. **Correctness check is self-consistency only** — local harness passes the same model
   object as both ref and sub. Participants can break the model and still pass locally.
   The server must run a real comparison against the frozen reference.

2. **`mlxfast submit` is a stub** — no server endpoint is wired. Participants cannot
   actually submit. The server API and leaderboard are not implemented.

3. **`EXPECTED_HARNESS_HASH` is never set** — the self-hash check is inert everywhere.
   The server must sign the harness wheel with this value before distributing to participants.

4. **Architecture invariant check covers only 4 of 11+ fields** — participants can
   silently change `hidden_size`, `head_dim`, `num_attention_heads`, etc. without
   the harness detecting it.

5. **Correctness prompt is uniform random tokens** — spec calls for NL prompts
   (5 natural language + 3 code + 2 adversarial). Random tokens do not stress
   real-world correctness.

6. **Server-side reference model not specified** — the server must re-run the
   correctness gate against the frozen reference checkpoint. This is not implemented
   and the protocol is not defined.

### Should Fix

7. **mactop is a hard runtime dependency** — not installed by default; no participant
   setup docs; harness raises `RuntimeError` if absent. Must be in requirements.

8. **Per-layer hidden state check not implemented** — spec says all 43 layers;
   code only checks the final hidden state.

9. **Decode step 1 includes 32-token prompt prefill** — timing window starts
   before the AR loop, slightly inflating decode latency.

10. **Transform sandbox is trust-based** — pre-loaded native extensions bypass
    all audit hooks. Not adversarially secure.
