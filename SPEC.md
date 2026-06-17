# Challenge Spec — DeepSeek V4 Flash on mlx-vlm

> Internal design document. Describes the finished challenge: constraints, anti-cheat
> mechanisms, scoring, modifiable surface, and harness contract.

---

## 1. Challenge premise

DeepSeek V4 Flash is a MoE language model with 256 routed experts (6 activated per token)
and compressed multi-head latent attention (MLA). At 4-bit quantization the full model
occupies ~30 GB of unified memory — more than most Apple Silicon configurations have
available. The baseline solves this with **mandatory SSD streaming**: expert weights live
on disk and only the 6 activated experts per token are loaded into Metal memory on demand.
This brings peak RAM to a level accessible on 24 GB machines.

The challenge is to run this streaming model *faster and cheaper*, not to implement
streaming from scratch. The baseline ships with a working SSD streaming implementation in
`mlx_models/mlx_lm_shims/switch_layers.py`. Every participant starts from a runnable
model. The score measures how much better than baseline you can get.

**What the baseline streaming gives you for free:** low peak RAM (~4–6 GB for non-expert
weights only). **What it costs:** per-token SSD reads for 6 expert trios, GPU-CPU sync
overhead, and no reuse between tokens. These are the bottlenecks to attack.

The entire model execution pipeline is in scope. Every subsystem — expert streaming
strategy, attention projections, compressed KV, shared experts, hyper connections,
embeddings, LM head — can be modified. A change that only affects expert loading will only
show partial savings. The scoring formula rewards whole-model thinking.

**Offline transform is optional.** Participants who find a purely inference-side improvement
do not need to transform the weights at all. The harness accepts both paths. Only if a
`transform.py` is present does the harness verify weight provenance.

---

## 2. Architecture reference

Model: `mlx-community/DeepSeek-V4-Flash-4bit`
Framework: `mlx-vlm==0.6.3`, `mlx>=0.31.2`, `mlx-lm>=0.31.3`
Minimum hardware: 24 GB unified memory (expert streaming required; baseline ships with it)
Reference machine: Apple M5 Max, 128 GB unified memory

Key parameters (from `ModelConfig`):

| Parameter | Value |
|---|---|
| `num_hidden_layers` | 43 |
| `hidden_size` | 4096 |
| `n_routed_experts` | 256 |
| `num_experts_per_tok` | 6 |
| `n_shared_experts` | 1 |
| `moe_intermediate_size` | 2048 |
| `intermediate_size` | 18432 |
| `num_attention_heads` | 64 |
| `num_key_value_heads` | 1 (MLA) |
| `q_lora_rank` | 1024 |
| `head_dim` | 512 |
| `qk_rope_head_dim` | 64 |
| `sliding_window` | 128 |
| `compress_ratios` | per-layer, values in {0, 4, 128} |
| `num_nextn_predict_layers` | 1 (MTP drafter) |
| `scoring_func` | `sqrtsoftplus` |

### Subsystems participants can optimize

1. **Routed experts** (`switch_mlp` — `SwitchGLU`): 256 experts × 3 projections.
   Only 6 are read per token. Already sparse by routing but dominant in absolute bytes.
2. **Shared experts** (`shared_experts` — `DeepseekV4MLP`): Read on every token.
   No routing sparsity — high leverage for transforms.
3. **MLA attention** (`Compressor`, `LocalAttention`, `CompressedAttention`): Q/KV
   projections, compressed KV cache. The `wkv` projection in `Compressor` is substantial.
4. **Hyper connections** (`HyperConnection`): Per-layer cross-layer mixing, read every
   token, commonly overlooked.
5. **Embeddings** (`embed_tokens`): One row lookup per token.
6. **LM head**: Full projection read on every decode step.
7. **MoE gate** (`MoEGate`): Small but read every layer every token.
8. **MTP speculative drafter** (`deepseek_v4_mtp`): Optional — participants may exploit
   the built-in MTP module for throughput. Bandwidth from drafter weights is included.

---

## 3. Baseline streaming implementation

The challenge ships a working baseline in `mlx_models/mlx_lm_shims/switch_layers.py`
that replaces mlx-lm's `SwitchGLU` / `QuantizedSwitchLinear` with an SSD-streaming
equivalent. Participants start here — the model runs out of the box before any
modification.

The baseline design is informed by [ds4-ssd](https://github.com/Anemll/ds4-ssd), a
C + Metal streaming engine for DeepSeek V4 Flash. Key architectural ideas are borrowed
and translated to clean Python/MLX. The goal is code that is maximally readable and
correct — not a port of ds4-ssd, and not a reference implementation that has already
solved the interesting problems.

### File layout — expert-major per-layer records

`transform.py` splits the reference checkpoint into per-layer expert files:

```
weights/
    config.json
    *.safetensors          # all non-expert weights (attention, shared experts,
                           #   embeddings, LM head, hyper connections)
    experts/
        manifest.json      # {layer_idx: {expert_idx: byte_offset}} for direct seeks
        layer_00.bin       # expert-major: 256 × 3-proj records, fixed record size
        layer_01.bin
        ...
        layer_42.bin
```

Each `layer_N.bin` stores experts as **contiguous fixed-size records** in expert-major
order. Record `i` contains the quantized weights, scales, and biases for expert `i`
across all three projections (`gate_proj`, `up_proj`, `down_proj`) packed back-to-back.
The `manifest.json` stores the byte offset of each expert record so any expert can be
addressed with a single `O(1)` seek — no header parsing on every access.

This is the key insight borrowed from ds4-ssd: the file format is designed for direct
random access, not sequential load. A participant loading 6 experts reads exactly 6
records — nothing else.

### Slot-bank — configurable resident Metal slots

Rather than an unbounded dict-based LRU (which grows until it fills RAM), the baseline
maintains a fixed pool of **N pre-allocated Metal arrays** as slots:

```
SLOT_BANK_SIZE = 32    # configurable; default fits ~5 decode steps of typical routing
```

When an expert is needed:
1. Check slot index — O(1) lookup by `(layer, expert_idx)`
2. **Hit**: use the slot's Metal array directly
3. **Miss**: pick the LRU slot, read the expert record from disk into that slot's
   pre-allocated Metal buffer, update the index

Pre-allocation matters: the Metal buffers are allocated once at startup, not on every
load. This is borrowed from ds4-ssd's "slot bank" concept. The wired memory footprint
is fixed and predictable: `SLOT_BANK_SIZE × record_size_bytes`.

### Page-cache cliff avoidance

A large wired slot bank evicts the OS unified-memory file cache, which the kernel uses
to absorb SSD reads. If the slot bank is larger than what fits comfortably alongside
the OS cache, decode slows dramatically (ds4-ssd observed a 13× slowdown in this
regime). The baseline uses a conservative default (32 slots ≈ 200 MB wired) that leaves
ample room for OS file cache to warm up across repeated expert accesses.

The slot bank size is a named constant in `switch_layers.py` — it is an explicit,
documented optimization target.

### What the baseline does NOT do

These are **intentional gaps** — known optimizations left for participants to discover
and implement:

- **No async I/O**: expert reads are synchronous and block the forward pass. An async
  worker pool that decouples disk reads from GPU compute is a significant win.
- **No prefetching**: the next layer's experts are not pre-loaded while the current
  layer computes. Even a one-step lookahead overlaps I/O with computation entirely.
- **No prefill-to-decode cache seeding**: the routing pattern during prefill predicts
  which experts will be hot during decode. Pre-warming the slot bank during prefill is
  a measurable decode win.
- **No cross-layer slot sharing**: each layer has its own slot bank. A global bank
  shared across layers could exploit the fact that expert routing correlates across
  adjacent layers.
- **No weight transform**: experts are stored as-is from the reference checkpoint
  (4-bit quantized). Abel summation, sorted layouts, or any other offline transform
  could reduce the bytes read per expert.
- **No alternative quantization**: the reference 4-bit is kept verbatim.

### Baseline expected metrics (reference M5 Max, 128 GB)

To be measured and published before challenge launch.

| Metric | Approximate baseline |
|---|---|
| Peak RAM | ~5–7 GB (non-expert weights only) |
| Bandwidth | TBD via mactop |
| Decode | TBD tok/s |
| Prefill | TBD tok/s |
| Score | TBD |

---

## 4. Score formula

```
score = peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token
```

- **`peak_ram_GB`** — peak unified memory during a 512-token decode run, from
  `mx.get_peak_memory()`, reset immediately before the decode loop.
- **`bandwidth_GB_per_token`** — actual DRAM bandwidth consumed per decoded token,
  measured via mactop hardware counters (§5).
- **`decode_sec_per_token`** — wall-clock decode latency averaged over 512 tokens of
  single-token autoregressive generation.
- **`prefill_sec_per_token`** — wall-clock latency per token for a 512-token full-sequence
  forward pass (2 timed runs after 1 warmup).

Correctness is a **hard gate** — failing submissions are not scored.

---

## 4. Correctness gate

The correctness gate has three independent layers, all of which must pass. A submission
that passes one but not another is a failing submission.

**Both the reference model and the participant model run with SSD streaming enabled.**
The reference model uses the baseline streaming implementation from the template. This
means the reference hidden states are themselves products of streaming inference — not a
fully-materialized reference. Any participant transform must match this baseline-streaming
reference, not a hypothetical fully-loaded reference. The correctness epsilon accounts for
any numerical variance introduced by the baseline streaming path itself.

### 4.1 Greedy token sequence match (primary gate)

Run greedy decode (temperature=0, top-p=1) for 256 steps on each eval prompt. Every
generated token must exactly match the reference model's greedy output at the same position:

```
for each position t in [0, 256):
    assert submission_token[t] == reference_token[t]
```

This is the strongest practical test. It catches any transformation that subtly shifts
the probability mass enough to change which token ranks first, even if hidden states look
fine under a floating-point epsilon. Floating-point non-determinism in GPU reductions can
in theory cause rare token flips on identical hardware — if this becomes an issue in
practice, a fallback to top-3 rank inclusion is available, but exact match is the default.

### 4.2 Layer-wise hidden state check (intermediate check)

At each of the 43 layer boundaries, record the hidden state tensor after the full block
(post-residual, pre-next-block). The submission must satisfy:

```
max( abs( submission_hidden[l] - reference_hidden[l] ) ) < ε_hidden
```

where `ε_hidden = 5e-3`. This is **tighter** than the Gemma 4 challenge (which used 1e-2)
and explicitly chosen to be below the range where small hidden-state errors compound into
token flips over a 256-step generation. A transform that keeps errors just under 1e-2 but
above 5e-3 fails this gate even if the token sequence happened to match on the eval set.

The hidden state check runs on a separate, shorter batch of prompts (32 tokens each)
where both models run in parallel and intermediate activations are captured.

### 4.3 Logit distribution check (output sanity)

For every position in the hidden-state check batch, the top-10 tokens by logit value must
be an identical set (order within the set is not required to match):

```
set(top_10_tokens(submission_logits[t])) == set(top_10_tokens(reference_logits[t]))
```

This catches cases where hidden states match within ε but the LM head amplifies small
differences into a different candidate set.

### Eval prompt design

Correctness is evaluated on prompts generated at runtime from a seed unknown to
participants before the harness runs:

```
seed = server_secret XOR SHA256(submission_commit_hash)
```

The seed generates:
- 5 diverse natural-language prompts (varied domain, length, vocabulary)
- 3 code-completion prompts
- 2 adversarial prompts (repetitive tokens, max-length context, rare Unicode)

All three checks (§4.1, §4.2, §4.3) run on the same prompt set.

---

## 5. Bandwidth measurement via mactop

DRAM bandwidth is measured using hardware performance counters via
[mactop](https://github.com/metaspartan/mactop), which reads the Apple Silicon IOReport
API. This provides ground-truth DRAM traffic — including activations, KV cache, weight
reads, and all intermediate tensors — not a model-derived estimate.

### Measurement protocol

```
baseline_bw = measure_idle_dram_bw(duration=3s)     # before model loads

# ... load both models ...

mx.reset_peak_memory()
start_mactop_sampling(interval=100ms)               # background process
t0 = time.perf_counter()

for _ in range(512):
    token = model.decode_one_step(...)

t1 = time.perf_counter()
samples = stop_mactop_sampling()

peak_ram = mx.get_peak_memory()
```

### Bandwidth calculation

```python
# samples: list of dram_bw_combined_gbs readings at 100ms intervals
# Filter out zero readings (IOReport calibration artifacts)
valid = [s for s in samples if s > 0]

# Subtract baseline idle bandwidth
net_bw_gbs = [max(s - baseline_bw, 0) for s in valid]

# Mean bandwidth × decode duration = total GB transferred
total_gb = mean(net_bw_gbs) × (t1 - t0)

bandwidth_per_token = total_gb / 512
```

### Why mactop over a software model

The Gemma 4 challenge used a software model (sum of `array.nbytes` × activation
fractions). That model:
- Misses activation tensor traffic (attention scores, gate logits, residuals)
- Cannot measure actual SSD streaming reads vs Metal cache hits
- Cannot distinguish a transform that reads fewer bytes from one that reads the same
  bytes but lies about it

mactop reads actual DRAM counters. If a participant finds a way to read fewer DRAM bytes —
whether by SSD streaming, sparse access, recomputation, or anything else — the hardware
counter reflects it. There is no way to game a hardware counter from Python.

### Mactop availability requirement

mactop must be installed on the machine running the harness:

```bash
brew install mactop
```

The harness checks for mactop at startup and aborts if not found. The harness runs mactop
as a subprocess with no elevated privileges (mactop supports unprivileged bandwidth
reading on M-series chips via IOReport).

### Known limitations

- mactop on M5+ uses "auto-calibrated power-based estimation" rather than direct counter
  reads; accuracy is validated to within ~5% by the mactop authors.
- The 100ms sampling interval means very short decode runs (< 10 tokens) have high
  variance. The 512-token decode window provides sufficient sample count.
- Idle baseline must be measured on the same machine with no other heavy processes
  running. The harness warns if system DRAM load during baseline exceeds a threshold.

---

## 6. Anti-cheat mechanisms

### 6.1 Weight provenance verification (transform path only)

If `transform.py` is present, the harness verifies that `weights/` was produced by it:

```
expected_hash = SHA256(transform.py_content || reference_manifest_hash)
actual_hash   = SHA256(weights/ file tree, sorted by path, content-hashed)
```

If hashes do not match, the run aborts.

If `transform.py` is absent, the harness expects `weights/` to be equivalent to the
reference weights (verified by comparing the content hash of `weights/` against the
reference manifest). In this case, the participant's improvement is purely in inference
code.

### 6.2 Harness self-integrity check

At startup, the harness hashes every file under `harness/` against a manifest embedded
at build time. Any modification aborts the run. This prevents patching the bandwidth
model, correctness checker, or score formula.

### 6.3 Architecture invariant check

Before loading participant weights, the harness parses `weights/config.json` and asserts
all frozen fields match the reference config exactly:

```python
FROZEN_CONFIG_FIELDS = [
    "model_type", "num_hidden_layers", "hidden_size", "n_routed_experts",
    "num_experts_per_tok", "n_shared_experts", "num_attention_heads",
    "num_key_value_heads", "vocab_size", "moe_intermediate_size",
    "head_dim", "q_lora_rank",
]
# Note: `intermediate_size` is NOT included — the reference DS4 Flash
# config.json does not contain this field (it uses `moe_intermediate_size`).
# The ModelConfig dataclass defaults intermediate_size=18432.
```

### 6.4 Peak RAM: harness-controlled measurement

`mx.get_peak_memory()` and `mx.reset_peak_memory()` are called only by the harness,
framing the decode loop. Participant code cannot manipulate the reported peak.

### 6.5 Bandwidth: hardware counters, not self-reported

mactop reads DRAM counters at the hardware level. There is no Python API that lets
participant code influence what the IOReport registers report.

### 6.6 Correctness: three independent gates

The three-layer correctness gate (§4) cannot be gamed on a single axis. Passing 4.1
(token match) while failing 4.2 (hidden states) is a failing submission. This prevents:
- Compensating errors: corrupt layer 5, fix at layer 20 → 4.2 catches it
- Calibrated lossy approximation: stay under 1e-2 but over 5e-3 → 4.2 catches it
- Output manipulation: hidden states match but logit set changes → 4.3 catches it
- Lucky token match on a small eval set: adversarial prompts in the eval set make
  it unlikely that a bad transform happens to produce the right tokens

### 6.7 Decode measurement: strictly single-token autoregressive

Decode latency is measured as wall-clock time for 512 individual one-token forward passes.
Batched decode or speculative decoding that produces multiple tokens per step must
normalize by accepted tokens per step (measured empirically during the run). Any
submission using MTP speculative decoding reports:
- `draft_tokens_per_step` (measured average)
- Raw `wall_clock_time / 512_accepted_tokens` as the score input

---

## 7. Modifiable surface

The harness uses a shadow-import mechanism: before loading any model code, it prepends
`mlx_models/` to `sys.path` and monkey-patches the relevant package namespaces so that
`mlx_vlm.models.deepseek_v4.*`, `mlx_vlm.models.base`, `mlx_vlm.models.cache`,
`mlx_vlm.turboquant`, `mlx_vlm.speculative.drafters.deepseek_v4_mtp.*`, and the relevant
`mlx_lm.models.*` modules all resolve to the participant's local copies first. This means
participants get a complete, editable copy of every file that participates in a forward
pass. There is no part of the execution pipeline that is inaccessible.

### 7.1 Core model — `mlx_models/deepseek_v4/`

Sourced from `mlx_vlm.models.deepseek_v4`. Every file in this package is modifiable.

```
mlx_models/deepseek_v4/
    __init__.py                  # package exports
    config.py                    # ModelConfig — frozen fields enforced by harness (§6.3),
                                 #   non-frozen fields (e.g. rope_scaling) may be changed
    deepseek_v4.py               # top-level Model: forward dispatch, sanitize, quant_predicate
    language.py                  # everything: LanguageModel, DeepseekV4Block,
                                 #   DeepseekV4MoE, DeepseekV4MLP, MoEGate,
                                 #   Compressor, Indexer, LimitedSwiGLU,
                                 #   LocalAttention, CompressedAttention,
                                 #   SparseCompressedAttention, DeepseekV4RoPE
    hyper_connection.py          # HyperConnection, HyperHead, hc_expand
                                 #   includes a custom fused Metal kernel —
                                 #   participants may rewrite or replace it
    processing_deepseek_v4.py    # FROZEN — tokenizer/processor; harness relies on it
```

### 7.2 Shared model infrastructure — `mlx_models/`

Sourced from `mlx_vlm.models` and `mlx_vlm.turboquant`. These files are shared across
all mlx-vlm models but are fully modifiable here — changes only affect the DeepSeek V4
execution path.

```
mlx_models/
    base.py              # BaseModelConfig, InputEmbeddingsFeatures,
                         #   LanguageModelOutput, create_attention_mask,
                         #   scaled_dot_product_attention
    cache.py             # KVCache, RotatingKVCache, BufferedRotatingKVCache,
                         #   QuantizedKVCache, BatchKVCache, CacheList —
                         #   primary target for KV memory reduction
    turboquant.py        # TurboQuantKVCache, BatchTurboQuantKVCache
                         #   Metal kernels for quantized KV cache —
                         #   modifiable; contains MSE-scoring and polar-coding kernels
```

### 7.3 MTP speculative drafter — `mlx_models/speculative/drafters/deepseek_v4_mtp/`

Sourced from `mlx_vlm.speculative.drafters.deepseek_v4_mtp`. The drafter shares
`language.py` blocks and hyper connections with the main model.

```
mlx_models/speculative/drafters/deepseek_v4_mtp/
    __init__.py
    config.py               # DeepseekV4MTPConfig — block_size, runtime_block_size
    deepseek_v4_mtp.py      # DeepseekV4MTPDraftModel — full drafter forward pass,
                            #   quantization config, shared KV state handling
    split.py                # offline utility: split MTP weights from main checkpoint
```

### 7.4 mlx-lm layer primitives — `mlx_models/mlx_lm_shims/`

Sourced from `mlx_lm.models`. These are the low-level compute primitives that
`language.py` delegates to. Modifying these changes how every MoE layer and every
attention projection computes.

```
mlx_models/mlx_lm_shims/
    switch_layers.py     # SwitchGLU, QuantizedSwitchLinear,
                         #   _gather_sort, _scatter_unsort —
                         #   the expert dispatch and compute kernel;
                         #   primary target for expert-level transforms
    mla.py               # MultiLinear, QuantizedMultiLinear —
                         #   used for Q/KV projections in MLA attention
    pipeline.py          # PipelineMixin — layer pipeline sharding
```

### 7.5 Offline transform and weights

```
transform.py    # OPTIONAL offline conversion script; must be deterministic if present
weights/        # transformed weights directory; must equal reference if no transform.py
```

### 7.6 What participants may also do

- Add new `.py` files anywhere under `mlx_models/` (imported by the above)
- Add helper scripts alongside `transform.py`
- Write custom Metal kernels (`.metal` source inline via `mx.metal.compile` or as
  separate files loaded at runtime)

### 7.7 What is frozen

```
harness/                   # measurement and validation — self-hashed (§6.2)
reference_weights/         # original 4-bit checkpoint — content-hashed (§6.1)
mlx_models/deepseek_v4/
    processing_deepseek_v4.py   # tokenizer — frozen
constants.py               # scoring constants, epsilon values
pyproject.toml             # dependency manifest — frozen; harness environment is fixed
```

---

## 8. Bandwidth breakdown (informational)

After each run the harness prints a per-phase bandwidth attribution derived from the
mactop timeline. The timeline is segmented by layer using Python-level timestamps inserted
around each layer's forward call:

```
Bandwidth breakdown (mactop-measured, GB/tok):
  prefill phase           X.XX
  decode — attention      X.XX   (avg over 512 steps)
  decode — experts        X.XX
  decode — shared mlp     X.XX
  decode — other          X.XX
  ─────────────────────────────
  decode total            X.XX   ← enters score
```

The per-phase breakdown is estimated by aligning mactop sample timestamps with layer
call timestamps. It is informational only — the score uses the total decode bandwidth.
The breakdown helps participants identify which subsystem is still the bottleneck.

---

## 9. Handrails — what the harness actively prevents

| Attempted shortcut | Prevention |
|---|---|
| Load a different checkpoint | Weight provenance hash §6.1 |
| Patch the bandwidth model | Harness self-hash §6.2 |
| Change model architecture | Frozen config fields §6.3 |
| Fake low peak RAM | Harness-controlled mx.get_peak_memory §6.4 |
| Claim bandwidth reduction without it | Hardware DRAM counters §6.5 |
| Compensating errors across layers | Per-layer hidden state check §4.2 |
| Calibrated lossy approximation | Tighter ε_hidden = 5e-3 §4.2 |
| Lucky token match on small eval set | Adversarial prompts + top-10 logit check §4.3 |
| Skip layers | Greedy token sequence must match exactly §4.1 |
| Inflate tok/s via batching | Strictly single-token autoregressive timing §6.7 |
| Out-of-scope model substitution | Architecture invariant check §6.3 |
| No transform but modified weights | No-transform path requires weights == reference §6.1 |

---

## 10. Permitted optimizations (non-exhaustive)

- Abel summation / suffix-sum basis on any weight matrix
- Sorted / permuted weight layouts that concentrate Δx energy
- Hadamard / DCT / learned orthogonal transforms baked into weights
- Block-sparse representations of any weight matrix
- Streaming weights from SSD (reduces peak RAM at cost of latency — fully scored)
- Per-expert, per-layer, per-subsystem transform schemas
- Custom quantization on any weight beyond the reference 4-bit
- MTP speculative decoding via the built-in `deepseek_v4_mtp` drafter
- Shared-expert fusion or pre-caching
- Custom KV cache compression
- Hyper connection folding (subject to correctness gate)
- Recomputation strategies that trade bandwidth for compute

---

## 11. Harness CLI contract

```bash
mlxfast run --note "description"
```

Steps in order:
1. Check mactop is installed and accessible (abort if not)
2. Harness self-hash check
3. Determine transform path: if `transform.py` exists → provenance check; else → verify
   `weights/` matches reference content hash
4. Architecture invariant check on `weights/config.json`
5. Measure idle DRAM baseline via mactop (3 seconds)
6. Load reference model from `reference_weights/`
7. Load participant model from `weights/`
8. Correctness: hidden-state + logit check on short prompts (§4.2, §4.3)
9. Correctness: greedy token sequence match on full eval prompts (§4.1)
10. `mx.reset_peak_memory()`; start mactop background sampler at 100ms
11. Decode benchmark: 512 tokens, single-token autoregressive, wall-clock timing
12. Stop mactop sampler; `peak_ram = mx.get_peak_memory()`
13. Compute bandwidth from mactop samples per §5
14. Prefill benchmark: 512-token forward pass, 2 timed runs after 1 warmup
15. Compute score; print per-phase breakdown
16. Append row to `results.tsv`; write `score.json`

```bash
mlxfast submit
```

Packages and uploads: `transform.py` (if present), `weights/` manifest, `mlx_models/`
source, `results.tsv`, git commit hash.

---

## 12. Open questions

- [ ] **mactop zero-sample filtering**: mactop occasionally returns `dram_bw_combined_gbs=0`
  as a calibration artifact (observed in testing). The harness must filter these before
  computing the mean. Need to determine: filter zeros only, or also filter outliers > 2σ
  from the mean? Risk of filtering legitimate low-bandwidth windows.

- [ ] **Baseline idle subtraction**: Subtracting idle baseline assumes system background
  DRAM activity is constant. If the OS or another process spikes during decode, the
  measurement is polluted. The harness should warn if baseline variance is high (> 0.5
  GB/s std dev) and optionally re-measure.

- [ ] **Per-phase breakdown precision**: Aligning mactop 100ms samples with per-layer
  Python timestamps will have quantization error (~100ms per layer transition). For a
  43-layer model decoding one token in ~200ms, each token's decode spans ~2 samples.
  The breakdown will be coarse. Consider whether it is worth implementing or just showing
  total bandwidth.

- [ ] **MTP normalization**: If a participant uses speculative decoding, how is
  `bandwidth_per_token` normalized? The decode loop runs K draft steps per accepted
  token. Proposal: measure total DRAM GB over the decode window, divide by number of
  accepted tokens (not draft steps). This means a bad drafter (low acceptance rate) pays
  in bandwidth AND latency simultaneously, which is correct.

- [ ] **No-transform path provenance**: If transform.py is absent, the harness verifies
  `weights/` == reference. But what if a participant wants to ship a lightly modified
  weights directory without a formal transform.py? Need to decide: require transform.py
  for any weights change (simplest), or allow no-transform path only when weights are
  byte-identical to reference.

- [ ] **Baseline publication**: Run the baseline streaming implementation through the
  harness on the reference M5 Max to establish the frontier baseline before launch.

- [ ] **Reference model streaming**: The correctness gate runs both models. The reference
  model uses the baseline streaming implementation. This means the harness must ship the
  baseline `switch_layers.py` as a frozen reference copy used only for correctness — not
  the participant's modifiable copy. Need to decide: (a) frozen copy in `harness/`, or
  (b) load the reference model from a snapshot of the baseline `mlx_models/` at a pinned
  commit, or (c) require the participant's implementation to match a non-streaming
  reference within a wider epsilon. Option (a) is simplest.

- [ ] **LRU cache size for baseline**: The default LRU of 64 entries needs to be chosen
  such that it fits in available RAM headroom on a 24 GB machine without evicting useful
  entries too aggressively. Needs profiling on target hardware.

- [ ] **Expert file split format**: Per-layer safetensors is the simplest split. But with
  256 experts × 3 projections per layer × 43 layers, alternative splits (per-expert
  files, sharded by expert group) may be worth specifying in the baseline to give
  participants flexibility in how they read subsets.
