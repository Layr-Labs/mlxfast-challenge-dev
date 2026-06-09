# quantizationfail — can you run Gemma 4 26B MoE without materializing it?

> A benchmark arena for memory-bandwidth-optimal LLM inference.  
> Find an offline transform of the weights such that inference never reads the full model — and beat the baseline score.

---

## The problem

Memory bandwidth is the dominant bottleneck for LLM inference on modern hardware. During each forward pass, the activated expert weights of a MoE model must be streamed through memory for every token routed to them. No matter how fast your compute is, you are waiting for memory.

The standard response is quantization: store weights at lower precision, read fewer bytes. But quantization still **materializes** the weight tensor — it just materializes a smaller one. Every token, every expert, every layer, every time.

This challenge asks a different question: **can you transform the weights offline such that the forward pass never needs to reconstruct them at all?**

The seed insight is Abel summation. For any vectors `x` and `y`:

```
dot(x, y) = dot(Δx, suffix_sum(y))
```

where `Δx[i] = x[i] - x[i+1]` and `suffix_sum(y)[i] = Σ_{j≥i} y[j]`.

This identity means you can store `suffix_sum(y)` instead of `y` and compute dot products against `Δx` — without ever recovering `y`. If you can find a basis in which `Δx` is sparse, you read only the nonzero entries of `suffix_sum(y)`. The weights are never materialized in their original form.

Note on floating point: Abel summation is exact in real arithmetic, but computing `Δx` and `suffix_sum(y)` in bfloat16 introduces rounding errors beyond simple reassociation. In practice these errors fall well within the `ε_float = 1e-2` correctness threshold, but this should be verified for your specific weight distribution and transform. Computing the prefix/suffix sums in float32 and then casting down is one way to keep accumulated error minimal.

This applies directly to the MoE expert weight matrices. Gemma 4 26B-A4B activates 3.8B parameters per token across its expert layers — those activated expert weight matrices are dense, and they are streamed from memory on every single forward pass. The challenge is finding a schema that makes that cheaper.

This is one point in a large space. The challenge is finding the best schema in that space.

---

## Current frontier

| Submission | Peak RAM (GB) | Bandwidth (GB/tok) | Decode tok/s | Prefill tok/s | Score | Correctness |
|---|---|---|---|---|---|---|
| **Baseline** — MLX 4-bit Gemma 4 26B MoE | ~27 | ~13.5 | ~69 | TBD | **TBD** | ✓ reference |

Score = `peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token`. Lower is better. Correctness is a hard gate — submissions that fail it do not appear on the leaderboard.

The baseline uses [`mlx-community/gemma-4-26b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-it-4bit) — the standard 4-bit MLX checkpoint, **not** the QAT variant — on an Apple M5 Max 128 GB, pinned at `mlx==0.31.1`, `mlx-lm==0.21.0`. Baseline bandwidth and score will be published once the reference harness measurement is complete.

---

## How it works

### What you submit

A fork of this repo with modifications to the files listed under [Modifiable surface](#modifiable-surface). Your submission must include:

1. **A modified weight directory** — the result of your offline transform applied to the original 4-bit MLX weights. This must be a deterministic function of the original weights, verified by the harness via content hashing of `transform.py` + output.
2. **Modified inference code** — changes to the MLX Gemma 4 implementation that operate natively on your transformed representation.
3. **A `transform.py`** — the offline conversion script that produces your weight directory from the originals. Must be reproducible.
4. **A `results.tsv` entry** — appended automatically by `quantizationfail run`.

### What the harness measures

```
quantizationfail run --note "what I tried"
```

This single command:

1. Verifies weight provenance — checks that your weight files are a deterministic function of the originals via content hashing of `transform.py` + output
2. Runs correctness validation — layer-wise exact match against the reference model at every hidden layer, on prompts seeded from a runtime-generated random seed (unknown until eval time)
3. Measures peak unified memory — via `mx.get_peak_memory()` (MLX's caching allocator high-water mark), isolated to the decode phase
4. Measures memory bandwidth — via an MoE-aware software model: `(shared_weight_bytes + activated_expert_bytes + kv_cache_bytes) / token_count`, where expert bytes are scaled by `experts_per_tok / num_experts`. This has been validated to within ~5% of hardware counters on Apple Silicon. MLX does not expose `MTLCounterSampleBuffer` values through its Python API.
5. Measures decode latency — wall-clock seconds per token, averaged over a 512-token autoregressive decode run
6. Measures prefill latency — wall-clock seconds per token for a 512-token full-sequence forward pass (2 timed runs after 1 warmup)
7. Computes score — `peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token`
8. Appends one row to `results.tsv` with timestamp, git commit, peak RAM, bandwidth, decode tok/s, prefill tok/s, score, correctness status, and your note

To submit to the leaderboard:

```
quantizationfail submit
```

---

## Correctness gate

The transform must be **mathematically lossless**. A submission passes correctness if, for every layer `l` in the model, hidden states match the reference exactly up to floating point associativity:

```
max( abs( submission_hidden[l] - reference_hidden[l] ) ) < ε_float
```

where `ε_float = 1e-2` (absolute tolerance on bfloat16 hidden states). This is the value used by the harness (`CORRECTNESS_EPSILON` in `constants.py`). It is generous enough to permit non-deterministic GPU reduction order but tight enough to reject any lossy approximation. Reordering of floating point operations is permitted; lossy approximation is not.

This is evaluated on tokens sampled from prompts generated by a seed produced by the harness at runtime. The seed is unknown to participants before `quantizationfail run` executes — it is derived from a server-side secret XORed with your submission commit hash, preventing hardcoding against a fixed eval set.

Layer-wise checking (rather than output-only) prevents compensating errors. A schema that corrupts an intermediate representation, even if the output logits happen to match, does not pass. This applies inside every activated expert — the hidden states after each expert MLP must match the reference, not just the final layer output.

---

## Modifiable surface

You may modify any of the following files. Everything else is frozen.

```
mlx_models/gemma4/              # MLX Gemma 4 model implementation
    model.py                    # Attention, MoE routing, layer definitions
    linear.py                   # Linear layer — the primary compute target
    weights.py                  # Weight loading and layout
    experts.py                  # Expert MLP implementation
transform.py                    # Your offline conversion script (you create this)
weights/                        # Your transformed weight directory (you create this)
```

You may **not** modify:

```
harness/                        # Measurement and validation code
    run.py
    correctness.py
    bandwidth.py
    score.py
    constants.py
tokenizer/                      # Tokenizer and embeddings
reference_weights/              # Original MLX 4-bit Gemma 4 26B weights (read-only)
```

There is no restriction on what transformations you apply. The score and correctness gate are the only judges. A submission that materializes the full expert weights during inference will pay for it in bandwidth and latency — you do not need to prove you avoided it.

---

## Getting started

### Requirements

- Apple Silicon Mac with at least 24 GB unified memory (M2/M3/M4 Mac Mini, MacBook Pro, Mac Studio, or Mac Pro)
- macOS Sequoia or later
- Python 3.11+

### Install the CLI

```bash
curl -fsSL https://api.quantizationfail.com/install.sh | sh
```

### Get the weights

Gemma 4 is Apache 2.0 licensed — no license gate required. Download the reference checkpoint via:

```bash
quantizationfail weights
```

This downloads `mlx-community/gemma-4-26b-it-4bit` to `reference_weights/` and computes the reference content hash. Do not substitute the QAT variant — it has different quantization structure and will not match the reference hidden states.

### Clone the benchmark

```bash
quantizationfail login <api-key>
quantizationfail clone
```

### Run the baseline

```bash
quantizationfail run --note "baseline, no changes"
```

You should see a score matching the published baseline. If your hardware differs from the reference machine, your absolute numbers will differ — the leaderboard normalizes by hardware tier.

### Improve and iterate

Modify the files under [Modifiable surface](#modifiable-surface). Run `quantizationfail run` after each change. The harness appends to `results.tsv` so your local experiment history is always intact.

```bash
# Example workflow
vim mlx_models/gemma4/linear.py   # implement your schema
python transform.py                # generate transformed weights
quantizationfail run --note "suffix-sum basis, experts sorted by L2 norm"
```

---

## Approach space

The seed insight — Abel summation — is one member of a large family of equivalent reformulations. Some directions worth exploring:

**Sorting / permutation.** Permute expert weight dimensions offline so that adjacent entries are similar. `Δx` concentrates energy in few large terms; most entries are near-zero and can be skipped. The permutation is a one-time offline cost and adds no runtime overhead beyond the modified compute path.

**Learned or fixed rotations.** Apply a Hadamard, DCT, or learned orthogonal transform to both weights and activations jointly. In the rotated basis, activation differences may be sparse by construction. A fixed transform such as Hadamard adds no per-token overhead since it can be baked into the weight representation.

**Hierarchical representations.** Store expert weights as a coarse base plus a sparse residual. At inference time, read only the base for most tokens and the residual selectively.

**Block-sparse suffix structures.** Tile each expert weight matrix into blocks, apply the identity within each block. Block sparsity in `Δx` translates directly to skipped memory reads with standard sparse kernels.

**Expert-aware transforms.** Different experts have different weight distributions. Per-expert transforms that exploit the specific structure of each expert's weights may outperform a single global schema.

**Quantization-aware transforms.** The existing 4-bit representation has algebraic structure — block scaling, fixed codebook. Transforms that respect this structure may compose with the existing quantization rather than replacing it, avoiding a full dequant/requant cycle.

These are starting points, not constraints. The challenge is open — any schema that passes the correctness gate and reduces the score is valid.

---

## Hardware

The reference machine for official scoring is:

- **Apple M5 Max**, 14-core CPU, 32-core GPU
- **128 GB** unified memory
- **macOS** Sequoia 15.x
- **MLX** 0.31.1 · **mlx-lm** 0.21.0

Self-reported scores from other Apple Silicon hardware are accepted for the community leaderboard with hardware tier noted. Only scores run on the reference machine via `quantizationfail submit` appear on the official frontier.

Community leaderboard entries from non-reference hardware display the raw measured score alongside the chip model (e.g., M3 Pro, M4 Max) so readers can make an informed comparison. No cross-hardware normalization formula is applied — hardware differences are visible in the `tok/s` axis, which reflects both compute and memory bandwidth differences between chips.

The 4-bit Gemma 4 26B MoE requires approximately 18 GB peak unified memory at baseline, making it accessible on any Mac with 24 GB or more — including M2/M3 Mac Mini, M3/M4 MacBook Pro, and Mac Studio configurations.

---

## Leaderboard and frontier

The leaderboard tracks the Pareto frontier on three axes: **peak RAM**, **bandwidth per token**, and **tok/s**. A single score (`peak_ram × bandwidth × seconds_per_token`) determines rank, but all three axes are displayed so submissions that trade one for another are visible.

The current frontier has one point: the baseline. We believe it is beatable. The Abel summation identity proves that mathematically equivalent reformulations exist — the challenge is finding ones that are computationally efficient on real hardware with real weight distributions.

---

## Scoring formula

```
score = peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token
```

- `peak_ram_GB` — peak unified memory during a 512-token decode run, measured by `mx.get_peak_memory()`
- `bandwidth_GB_per_token` — estimated GB read per decoded token via an MoE-aware software model (shared weights + activated expert weights + KV cache), validated to ~5% accuracy on Apple Silicon
- `decode_sec_per_token` — wall-clock decode latency per token, averaged over a 512-token autoregressive decode run
- `prefill_sec_per_token` — wall-clock latency per token for a 512-token full-sequence forward pass (2 timed runs after 1 warmup)
- Correctness is a **hard gate** — failing submissions are not scored

Lower score is better. This formula cannot be gamed on a single axis:

- Reducing bandwidth by reading a compressed representation but requiring expensive compute to do so pays for it in `decode_sec_per_token`
- Shifting computation from decode to prefill (e.g. computing sparse index structures once per prompt) pays for it in `prefill_sec_per_token`
- Speeding up decode by skipping correctness-critical computation fails the hard gate before scoring
- Reducing peak RAM by streaming large intermediate tensors through the GPU pays for it in `bandwidth_GB_per_token`

All four dimensions must improve together for the score to meaningfully drop.

---

## FAQ

**Can I change the model architecture?**  
You may change how the existing layers compute their outputs. You may not add new trained parameters or fine-tune the weights.

**Can I partially materialize weights — e.g. one expert block at a time?**  
Yes. If your schema reads blocks sequentially and the Metal counters reflect the actual bytes read, your score reflects the real cost. There is no rule against streaming — only the measured bandwidth matters.

**Can my transform be lossy?**  
No. The transform must be mathematically lossless. Hidden states at every layer must match the reference exactly within IEEE 754 bfloat16 numerical tolerance. Approximations that trade correctness for bandwidth do not qualify — the bandwidth win must come entirely from the representation and compute schema, not from degrading the model.

**Can I use the QAT checkpoint as my baseline?**  
No. The reference checkpoint is `mlx-community/gemma-4-26b-it-4bit` — the standard 4-bit MLX checkpoint. QAT variants have different quantization structure and will fail the correctness gate against the reference hidden states.

**Does my offline transform have to be fast?**  
No. Transform time is not scored. It runs once. It can take hours or days.

**Does the transform have to apply to all experts equally?**  
No. Per-expert transforms are permitted. The correctness gate applies uniformly — every expert's hidden states must match — but the schema for each expert can differ.

**Does this only apply to the expert weights?**  
No. You may transform any weights in the modifiable surface — attention projections, shared layers, expert weights. The expert FFN matrices are the dominant bandwidth target given MoE routing, but nothing is off-limits.

**What if I find a schema that beats the score but I can't open-source the transform?**  
Contact us. We will work something out.

---

## Community

- Slack: [join here](https://join.slack.com/t/quantizationfail/shared_invite)
- GitHub Discussions: open an issue or discussion on this repo
- Results are public — every submitted `results.tsv` is visible on the leaderboard

This challenge is led by [your name / org here]. Contributions, issues, and forks welcome.

---

## License

The harness and benchmark code are MIT licensed. The Gemma 4 model weights are released under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) — use, modification, and redistribution are freely permitted. Your submissions belong to you.
