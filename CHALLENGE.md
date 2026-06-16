# mlxfast — DeepSeek V4 Flash: beat the baseline score

> A benchmark arena for memory-bandwidth-optimal LLM inference on Apple Silicon.
> Run DeepSeek V4 Flash without loading all 256 experts into RAM — and beat the baseline score.

---

## The problem

DeepSeek V4 Flash has 256 routed experts per MoE layer, 6 activated per token, across 43 layers.
At 4-bit quantisation the full expert stack is ~30 GB — more than most Apple Silicon machines can hold in unified memory.

The baseline ships with **SSD streaming**: only the 6 activated experts per token are ever loaded into Metal memory. Peak RAM stays under ~6 GB. But the baseline is deliberately naive:

- Expert reads **block** the forward pass
- **No prefetching** — the next layer's experts are never pre-loaded
- **No cross-layer reuse** — each (layer, expert) pair is an independent slot with no sharing across time steps
- Weights are stored in their **original 4-bit form** — the transform is a no-op beyond splitting the file

Every one of these is an optimisation target. The challenge is to lower the score as far as possible.

---

## Current frontier

| Submission | Peak RAM (GB) | Bandwidth (GB/tok) | Decode (s/tok) | Prefill (s/tok) | Score |
|---|---|---|---|---|---|
| **Baseline** — SSD streaming, no prefetch | TBD | TBD | TBD | TBD | **TBD** |

Score = `peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token`. Lower is better.
Correctness is a hard gate — submissions that fail it do not appear on the leaderboard.

Baseline numbers will be published once measured on the reference M5 Max 128 GB machine.

---

## How it works

### What you submit

A fork of this repo with modifications to the files listed under [Modifiable surface](#modifiable-surface). Your submission must include:

1. **Modified inference code** — changes under `mlx_models/` that run faster, use less RAM, or read fewer bytes.
2. **A `transform.py`** (optional) — an offline weight conversion script that produces `weights/` from `reference_weights/`. If not modified, the baseline transform runs unchanged.
3. **A `results.tsv` entry** — appended automatically by `mlxfast run`.

### What the harness measures

```
mlxfast run --note "what I tried"
```

This single command:

1. Runs your `transform.py` if it has changed since last run (or skips if weights are already present)
2. Loads your model and the frozen reference model from the same `weights/` directory
3. Runs correctness validation — three independent layers (see [Correctness gate](#correctness-gate))
4. Measures peak unified memory via `mx.get_peak_memory()`, isolated to the decode phase
5. Measures memory bandwidth via **mactop hardware DRAM counters** — real IOReport byte counts, not a software model
6. Measures decode latency — wall-clock seconds per token, averaged over the decode run
7. Measures prefill latency — wall-clock seconds per token for the prefill phase
8. Computes score — `peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token`
9. Appends one row to `results.tsv` and writes `score.json`

To submit to the leaderboard:

```bash
mlxfast submit
```

---

## Correctness gate

Correctness is a **hard gate** enforced at three independent layers. A submission must pass all three or it does not appear on the leaderboard.

### Layer 1 — Greedy token sequence

The submission model must produce the exact same greedy token sequence as the reference model for 256 autoregressive steps. This catches any approximation that shifts the argmax.

### Layer 2 — Hidden state tolerance

For each transformer layer `l`, the absolute deviation between submission and reference hidden states must be within tolerance:

```
max( abs( submission_hidden[l] - reference_hidden[l] ) ) < ε = 5e-3
```

This is tighter than layer-1 alone: a submission could produce matching greedy tokens while accumulating hidden-state drift that would cause divergence on longer sequences or different prompts.

### Layer 3 — Top-10 logit set

The set of the 10 highest-probability tokens at each step must match exactly between submission and reference. This prevents submissions that produce the correct greedy token through coincidental logit shifts.

All three layers are evaluated on prompts seeded by a runtime-generated value unknown until `mlxfast run` executes. The seed is derived from a server-side secret XORed with your submission commit hash — preventing any hardcoding against a fixed eval set.

---

## Bandwidth measurement

Bandwidth is measured via **mactop**, which reads Apple's hardware IOReport DRAM counters directly:

```bash
mactop --headless --count 20 --interval 100 --format json
```

The harness runs mactop in the background during the decode phase, collects `soc_metrics.dram_bw_combined_gbs` samples, and computes the mean non-zero value. This is a real hardware measurement — it counts actual bytes transferred between DRAM and SoC, not a software model of what should have been read.

**Why this matters:** Software bandwidth models are gameable. Any approach that claims to read fewer bytes but actually reads them via a different code path still shows up in the hardware counters. You cannot fake it.

---

## Modifiable surface

You may modify any file under `mlx_models/` and `transform.py`. Everything else is frozen.

### Core model — `mlx_models/deepseek_v4/`

| File | What it controls |
|---|---|
| `language.py` | All layer logic: MoE routing, attention, shared experts, hyper connections. |
| `deepseek_v4.py` | Top-level model: forward dispatch, streaming configuration. |
| `hyper_connection.py` | HyperConnection residual mixing + fused Metal kernel. |
| `config.py` | Model configuration. Shape parameters are frozen; runtime knobs are open. |

### Expert streaming — `mlx_models/mlx_lm_shims/`

| File | What it controls |
|---|---|
| `switch_layers.py` | **Primary target.** Expert slot bank, SSD loading, dispatch. `SLOT_BANK_SIZE`, prefetching logic, async I/O, cross-layer reuse — all here. |
| `mla.py` | MultiLinear / QuantizedMultiLinear — MLA attention projections. |

### KV cache — `mlx_models/`

| File | What it controls |
|---|---|
| `cache.py` | KV cache implementations: `RotatingKVCache`, `QuantizedKVCache`, and others. |
| `turboquant.py` | TurboQuant KV cache Metal kernels. |

### Speculative decoding — `mlx_models/speculative/`

| File | What it controls |
|---|---|
| `drafters/deepseek_v4_mtp/deepseek_v4_mtp.py` | MTP speculative decoding drafter. |
| `drafters/deepseek_v4_mtp/config.py` | Drafter configuration. |

### Offline transform

| File | What it controls |
|---|---|
| `transform.py` | Offline weight conversion. Default: splits experts into per-layer binary files. May be replaced with any deterministic function of `reference_weights/`. |

### Frozen (do not modify)

```
harness/                  # measurement and validation code
reference_weights/        # original 4-bit checkpoint (read-only)
pyproject.toml            # harness environment is fixed
```

The frozen set is enforced by content hashing at submission time. Any modification to a frozen file causes the submission to be rejected.

---

## The baseline in detail

The baseline `switch_layers.py` implements a minimal SSD streaming path:

```
weights/experts/
  manifest.json           expert record layout (dtype, shape, byte offsets)
  layer_00.bin            256 expert records, expert-major, fixed record size
  layer_01.bin
  ...
  layer_42.bin
```

Each `layer_NN.bin` stores 256 fixed-size records. Record `j` contains the packed arrays for expert `j`:

```
[gate_proj.weight][gate_proj.scales][up_proj.weight][up_proj.scales][down_proj.weight][down_proj.scales]
```

At inference time, `StreamingSwitchGLU`:

1. Sorts routing indices ascending — contiguous expert accesses for Metal
2. Identifies unique activated experts (at most `N × K`, typically 6–12 per batch)
3. Loads each from the `ExpertSlotBank` LRU cache (or from disk on miss via `os.pread`)
4. Stacks into small dense `(num_unique, out, in_packed)` tensors
5. Calls `mx.gather_qmm` — the same fused kernel used by the original mlx-lm SwitchGLU

The `ExpertSlotBank` is a fixed-capacity LRU (`SLOT_BANK_SIZE = 32` slots default, ~400 MB wired). Raising this reduces disk reads at the cost of wired memory.

This baseline is intentionally simple. The optimization targets are explicit:

| Gap | Approach |
|---|---|
| Reads block forward pass | Async I/O — overlap SSD reads with GPU compute |
| No prefetching | Predict next layer's experts from current routing; pre-load before needed |
| No cross-layer reuse | Cache (layer, expert) pairs across decode steps when routing is stable |
| No weight transform | Store experts in a form that requires fewer bytes to express the same computation |
| No prefill seeding | Use routing pattern during prefill to warm the decode slot bank |

---

## Approach space

The scoring formula penalises all four dimensions simultaneously:

- **Lower bandwidth** without increasing latency: store experts in a more compact representation (different quantisation, delta coding, structured sparsity).
- **Lower peak RAM** without increasing bandwidth: keep fewer tensors live simultaneously; stream through smaller working sets.
- **Lower decode latency** without increasing bandwidth: async I/O, prefetching, better Metal kernel utilisation.
- **Lower prefill latency**: batch expert loads across the prompt sequence; use the known routing pattern to load all needed experts before the forward pass begins.

Some directions worth exploring:

**Async prefetching.** While the GPU computes layer N, load layer N+1's experts from SSD. The forward pass never blocks on I/O.

**Routing-aware reuse.** Track which experts are activated across consecutive decode steps. Stable routing (common during greedy decode) means the same 6–12 experts are needed repeatedly — keep them pinned rather than cycling through LRU.

**Prefill seeding.** During prefill the full routing pattern is known in advance. Load all needed experts before the decode loop starts.

**Weight transforms.** Replace the baseline no-op transform with a representation that is cheaper to load or execute: lower-bit quantisation, weight sharing across experts, delta compression between similar experts.

**KV cache compression.** `QuantizedKVCache` and `TurboQuant` are in the modifiable surface. Reducing KV cache bandwidth shows up directly in the hardware DRAM counters.

**Speculative decoding.** The MTP drafter in `mlx_models/speculative/` is modifiable. Accepted speculative tokens reduce the number of full expert loads per generated token.

---

## Getting started

### Requirements

- Apple Silicon Mac, 24 GB+ unified memory (M2 or newer recommended)
- macOS Sequoia or later
- Python 3.11+
- `mlx>=0.31.2`, `mlx-vlm==0.6.3`, `mlx-lm>=0.31.3`
- [mactop](https://github.com/metaspartan/mactop) — installed by `./setup.sh` via Homebrew when missing

### Install

```bash
./setup.sh
```

Downloads `mlx-community/DeepSeek-V4-Flash-4bit` (~30 GB) to `reference_weights/`. Do not substitute a different checkpoint — it will fail the correctness gate.

### Run the baseline

```bash
python transform.py           # split expert weights into weights/experts/ (one-time)
mlxfast run --note "baseline"
```

You should see a score matching the published baseline. Your absolute numbers will vary by hardware; the leaderboard notes hardware tier.

### Iterate

```bash
vim mlx_models/mlx_lm_shims/switch_layers.py   # primary target
python transform.py                              # only if you changed weight layout
mlxfast run --note "async prefetch v1"
```

Results append to `results.tsv`:

```bash
column -t -s $'\t' results.tsv
```

---

## Scoring formula

```
score = peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token
```

- `peak_ram_GB` — peak Metal memory allocation (MLX high-water mark) during decode
- `bandwidth_GB_per_token` — mean DRAM bandwidth from mactop hardware counters during decode, divided by tokens generated
- `decode_sec_per_token` — wall-clock decode latency per token
- `prefill_sec_per_token` — wall-clock prefill latency per token

Lower is better. The formula cannot be gamed on a single axis:

- Compress bandwidth by requiring expensive dequant → pays in `decode_sec_per_token`
- Stream everything to save RAM → pays in `bandwidth_GB_per_token` (still shows in DRAM counters)
- Cache experts in RAM to save bandwidth → pays in `peak_ram_GB`
- Skip correctness-critical computation → fails the hard gate before scoring

---

## FAQ

**Can I change the model architecture?**
You may change how existing layers compute their outputs within `mlx_models/`. You may not add new trained parameters or fine-tune the weights.

**Can I use async I/O?**
Yes. The baseline deliberately does not. Overlapping SSD reads with GPU compute is one of the clearest optimisation targets.

**Can my transform be lossy?**
No. The three-layer correctness gate rejects any approximation. The bandwidth improvement must come from representation and compute — not from degrading the model.

**Does the transform have to apply to all experts equally?**
No. Per-expert transforms are permitted. The correctness gate applies uniformly but the schema for each expert can differ.

**Does the transform have to be fast?**
No. Transform time is not scored. It runs once offline.

**Can I change `SLOT_BANK_SIZE`?**
Yes — it is the first tunable constant in `switch_layers.py` and is explicitly documented. Raising it keeps more experts wired (fewer disk reads, higher peak RAM).

**What if my approach requires weights that don't fit in 30 GB?**
Your transformed `weights/` directory can be larger than `reference_weights/`. There is no size limit on the transform output — only the runtime metrics are scored.

**Can I use the reference model's routing pattern to cheat the correctness check?**
No. Correctness is checked on prompts seeded at runtime from a server-side secret. The routing pattern is not available ahead of time.

---

## Hardware

Reference machine for official scoring:

- **Apple M5 Max**, 14-core CPU, 32-core GPU, 128 GB unified memory
- macOS Sequoia 15.x
- `mlx>=0.31.2` · `mlx-vlm==0.6.3`

Community leaderboard entries from other Apple Silicon hardware are accepted with hardware tier noted. Only scores run on the reference machine via `mlxfast submit` appear on the official frontier.

---

## License

The harness and benchmark code are MIT licensed. The DeepSeek V4 Flash model weights are released under the [DeepSeek Model License](https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-MODEL). Your submissions belong to you.
