# MTP decode benchmark — Qwen 3.6 27B native multi-token prediction

Reproducible benchmark for the `Qwen36MTPDriver` exploratory target: decode
speedup, draft acceptance, and prefill throughput at draft depths k = 1, 2, 3
across three workloads. Every cell also asserts exactness (`exact-vs-serial:
MATCH`) — a cell that mismatches is a correctness regression, not a benchmark
result.

## Files

| file | purpose |
|---|---|
| `run-matrix.sh` | 9-cell matrix (3 workloads × k=1,2,3) + warm-cache probe → `matrix.log` |
| `run-prefill.sh` | prefill throughput at ~480–3840 prompt tokens |
| `parse_matrix.py` | `matrix.log` → JSON rows + human-readable summary (`./parse_matrix.py [log]`) |
| `baselines/2026-08-12-m5max-128gb.json` | recorded baseline (cells + environment metadata) |
| `report-2026-08-12.html` | full written report for the baseline run |

## Running

```bash
swift build                       # plain build — `--target` compiles but does not link
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
MTP_BIN=.build/arm64-apple-macosx/debug/Qwen36MTPDriver \
MTP_WEIGHTS=../weights-cache/qwen36-mtp-merged \
  benchmarks/mtp-decode/run-matrix.sh
benchmarks/mtp-decode/parse_matrix.py matrix.log
```

On a shared machine, set `MTP_LOCK_DIR` to a mkdir-mutex path so model runs
serialize with other users. Run warm: discard a cold-page-cache first process
(~23% slower serial). Do not benchmark on a machine that pages — the 24 GB
boxes produce meaningless timings with this 15 GB checkpoint.

## Baseline — 2026-08-12, M5 Max 128 GB (debug build, driver `fd4f04dd`, fork `e0c1d54`)

Workloads: W1 code (chat, binary-search prompt, 256 tok) · W2 prose (poem
prompt, 256 tok) · W3 Q/A (raw capital-of-France, 32 tok). All 11 runs MATCH.

| cell | k | accept % | tok/fwd | serial tok/s | MTP tok/s | speedup |
|---|---|---|---|---|---|---|
| W1 code | 1 | 88.2 | 1.882 | 24.0 | 36.2 | 1.51× |
| W1 code | 2 | 77.5 | 2.535 | 23.9 | 36.6 | **1.53×** |
| W1 code | 3 | 60.9 | 2.813 | 23.9 | 29.4 | 1.23× |
| W2 prose | 1 | 63.1 | 1.631 | 23.9 | 26.1 | 1.10× |
| W2 prose | 2 | 49.4 | 1.984 | 23.8 | 24.1 | 1.01× |
| W2 prose | 3 | 38.1 | 2.133 | 23.8 | 20.8 | 0.87× |
| W3 Q/A | 1 | 72.2 | 1.684 | 22.2 | 27.4 | 1.23× |
| W3 Q/A | 2 | 73.1 | 2.462 | 22.2 | 31.7 | 1.42× |
| W3 Q/A | 3 | 57.1 | 2.667 | 22.4 | 29.1 | 1.30× |

Prefill ≈ 960 tok/s (flat to ~2k context, 873 tok/s at 3.8k); model load 1.60 s.

Cost model (matches all cells to ±0.005): `speedup = tok/fwd ÷ round-cost`,
each depth increment costing ~+0.4–0.5 serial-forward-equivalents per round
(includes the per-rejecting-round repair forward of the MTPLX-style loop — a
future DFlash tape graft removes that term). Read: **k=2 default** (peak on
structured output, never below break-even), k=1 the never-loses floor, k=3
never optimal in this matrix.

## Comparing against the baseline

- **Acceptance and tok/fwd are deterministic** under greedy decode — bit-identical
  across replications. Any change in these means the model/driver *behavior*
  changed, not noise.
- **Timings carry ~1% warm spread** (replication probe in the baseline JSON).
  Treat speedup deltas under ~3% as noise; the cost-model decomposition is the
  more sensitive detector of real regressions.
- **Any MISMATCH is a stop-the-line correctness bug.** The driver prints the
  serial top-2 logit margin at the divergent index: near-zero margin late in a
  run is the known bf16 chunk-geometry tie-flip class; a large margin or a
  mismatch at a rejection boundary is a logic bug.
