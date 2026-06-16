# mlxfast — DeepSeek V4 Flash

A benchmark arena for memory-bandwidth-optimal LLM inference on Apple Silicon.
Run DeepSeek V4 Flash without loading all 256 experts into RAM — and beat the baseline score.

See [CHALLENGE.md](CHALLENGE.md) for the full problem statement, scoring formula, and approach space.

## Quickstart

```bash
# Install Homebrew/mactop if needed, then Python deps and weights
./setup.sh

# Split expert weights onto SSD (baseline transform, runs once)
python transform.py

# Run the baseline — should match the published baseline score
mlxfast run --note "baseline"

# Edit the modifiable surface and iterate
vim mlx_models/mlx_lm_shims/switch_layers.py
python transform.py   # only if you changed weight layout
mlxfast run --note "my approach v1"
```

Results append to `results.tsv`:

```bash
column -t -s $'\t' results.tsv
```

## Why this challenge exists

DeepSeek V4 Flash has 256 routed experts per layer, 6 activated per token.
At 4-bit quantisation the full expert stack is ~30 GB — more than most Apple
Silicon machines can hold. The baseline ships with SSD streaming: only the 6
activated experts per token are loaded into Metal memory, keeping peak RAM
under ~6 GB.

That baseline is functional but naive. Expert reads block the forward pass,
there is no prefetching, no cross-layer reuse, and the weights are stored in
their original 4-bit form. Every one of these is an optimisation target.

## The modifiable surface

Unlike typical inference benchmarks, the entire model execution pipeline is
in scope. You can modify any file under `mlx_models/`:

| Path | What it controls |
|---|---|
| `mlx_models/mlx_lm_shims/switch_layers.py` | Expert streaming: slot bank, loading, dispatch. **Primary target.** |
| `mlx_models/deepseek_v4/language.py` | All layer logic: MoE routing, attention, shared experts, hyper connections. |
| `mlx_models/deepseek_v4/deepseek_v4.py` | Top-level model: forward dispatch, streaming configuration. |
| `mlx_models/deepseek_v4/hyper_connection.py` | HyperConnection + fused Metal kernel. |
| `mlx_models/cache.py` | KV cache implementations (RotatingKVCache, QuantizedKVCache, …). |
| `mlx_models/turboquant.py` | TurboQuant KV cache Metal kernels. |
| `mlx_models/speculative/drafters/deepseek_v4_mtp/` | MTP speculative decoding drafter. |
| `mlx_models/mlx_lm_shims/mla.py` | MultiLinear / QuantizedMultiLinear (MLA attention projections). |

Plus:

- `transform.py` — offline weight transform. Deterministic function of the reference weights.
- `weights/` — output of `transform.py`. The harness loads from here.

## Scoring

```
score = peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token
```

Bandwidth is measured via **mactop hardware DRAM counters** — not a software model.
Correctness is a hard gate. See CHALLENGE.md for the full correctness specification.

**Baseline (TBD — reference M5 Max 128 GB):**

| Peak RAM | Bandwidth | Decode | Prefill | Score |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD |

## Architecture

```
mlx_models/                  modifiable surface — the full DS4-Flash pipeline
  deepseek_v4/               core model (language, attention, MoE, hyper connections)
  mlx_lm_shims/              expert dispatch + MLA primitives (SwitchGLU, MultiLinear)
  speculative/               MTP speculative decoding drafter
  base.py / cache.py         shared model infrastructure and KV cache
  turboquant.py              TurboQuant KV cache kernels
transform.py                 offline weight transform (optional)
weights/                     transformed weights (harness loads from here)
  experts/
    manifest.json            expert record layout
    layer_NN.bin             per-layer expert binaries (expert-major, fixed record size)
reference_weights/           original 4-bit checkpoint (frozen, read-only)
harness/                     frozen measurement and validation code
results.tsv                  local experiment log
score.json                   written after each finite passing run
```

## Requirements

- Apple Silicon Mac, 24 GB+ unified memory (M2 or newer)
- macOS Sequoia or later
- Python 3.11+
- `mlx>=0.31.2`, `mlx-vlm==0.6.3`, `mlx-lm>=0.31.3`
- [mactop](https://github.com/metaspartan/mactop) — installed by `./setup.sh` via Homebrew when missing
