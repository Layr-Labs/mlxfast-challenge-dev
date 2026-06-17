# Model Architecture — DeepSeek V4 Flash

## Reference

Model: `mlx-community/DeepSeek-V4-Flash-4bit`
Local path: `mlxfast/reference_weights/DeepSeek-V4-Flash-4bit/`
Quantization: mxfp4 (4-bit, MLX group quantization, uint32 packed weights + uint8 scales)

## Top-Level Parameters

| Parameter | Value | Notes |
|---|---|---|
| `num_hidden_layers` | 43 | All layers are MoE (no dense-only layers) |
| `hidden_size` | 4096 | |
| `vocab_size` | 129280 | |
| `n_routed_experts` | 256 | Per-layer expert pool |
| `num_experts_per_tok` (K) | 6 | Experts activated per token |
| `n_shared_experts` | 1 | Shared MLP always runs (not streamed) |
| `moe_intermediate_size` | 2048 | Expert hidden dim |
| `intermediate_size` | 18432 | Shared expert hidden dim |
| `num_attention_heads` | 64 | |
| `num_key_value_heads` | 1 | MLA: single KV head |
| `q_lora_rank` | 1024 | Query low-rank compression |
| `head_dim` | 512 | |
| `qk_rope_head_dim` | 64 | RoPE dimension per head |
| `sliding_window` | 128 | LocalAttention window size |
| `hc_mult` | 4 | HyperConnection parallel streams |
| `num_hash_layers` | 3 | First 3 layers use hash routing |
| `scoring_func` | `sqrtsoftplus` | Gate activation |
| `num_nextn_predict_layers` | 1 | MTP speculative drafter attached |

## Per-Token Forward Pass (Decode, N=1)

For each of the 43 transformer layers:

```
HyperConnection.expand(h)              # (B, L, hc_mult=4, H) → single stream x
  ├─ RMSNorm
  ├─ Attention (MLA — see §Attention)
  └─ FFN: DeepseekV4MoE
       ├─ MoEGate → top-6 expert indices + scores    [GPU sync: .tolist()]
       ├─ StreamingSwitchGLU(x, indices)              [SSD I/O per miss]
       │    └─ 3× gather_qmm (gate, up, down projections)
       ├─ shared_experts(x)                           [always runs, in Metal]
       └─ output = sum(scores[k] * expert_k(x)) + shared(x)
HyperConnection.collapse(...)          # → (B, L, hc_mult=4, H)
```

The 43 `MoEGate` calls each force a GPU→CPU sync via `.tolist()` on the routing
indices. This is the hard floor on decode latency — expert selection requires
knowing which records to load before any disk I/O can begin.

## Attention Layers

Three modes, assigned per layer based on `compress_ratios[layer_idx]`:

| `compress_ratio` | Class | Layers | Description |
|---|---|---|---|
| 0 | `LocalAttention` | 0, 42 (first + last) | Pure sliding-window, no compression |
| 128 | `CompressedAttention` | Even middle layers (2, 4, 6, ...) | KV compressed by factor 128 |
| 4 | `SparseCompressedAttention` | Odd middle layers (1, 3, 5, ...) | KV compressed by factor 4 + sparse top-512 index retrieval |

`compress_ratios` generation (from `config.py.__post_init__`):
```python
[0]                                                    # layer 0
+ [4 if i % 2 else 128 for i in range(max(n-2, 0))]   # middle: alternating 128/4
+ ([0] if n >= 2 else [])                              # layer 42
```

KV cache types:
- `LocalAttention` → `RotatingKVCache(max_size=128)` (circular buffer)
- `CompressedAttention` → `PoolingCache` (compressed pooled KV)
- `SparseCompressedAttention` → `PoolingCache` + `Indexer` (top-512 retrieval)

The `Compressor` module (`wkv` projection) shapes:
- `compress_ratio=128`: `wkv` shape `(4096, 512)` = 4 MB at bfloat16
- `compress_ratio=4`: `wkv` shape `(4096, 1024)` = 8 MB at bfloat16

Both are dense (in Metal, not streamed), read every token every layer.

## MoE Routing

### Layers 0–2: Hash Routing (`num_hash_layers=3`)

```python
expert_idx = tid2eid[input_token_id % len(tid2eid)]
```

Deterministic. No learned gate. Same token always activates same expert.

### Layers 3–42: Learned Routing (`_expert_select`, `@mx.compile`)

```python
biased = logits + e_score_correction_bias
inds = mx.argpartition(-biased, kth=K-1, axis=-1)[..., :K]
weights = mx.softmax(biased[..., inds], axis=-1)
```

`mx.argpartition` is non-deterministic for ties on GPU. For bfloat16 arithmetic
at boundary values, different hardware runs may select different experts.
This is documented variance — the spec allows bfloat16 associativity.

`scoring_func = sqrtsoftplus`: applies `sqrt(softplus(x))` to weights after softmax.

## HyperConnection

Each of the 43 layers has two `HyperConnection` modules (`attn_hc`, `ffn_hc`).

```
HyperConnection params:
  fn:    (mix=24, hc_mult×hidden) = (24, 16384) ≈ 1.6M float32 = 6.3 MB
  base:  (hc_mult, hidden) = (4, 4096)
  scale: (hc_mult, hidden) = (4, 4096)
```

2 HyperConnections × 43 layers × 6.3 MB ≈ 541 MB/token bandwidth for HC matrices alone.
This is in Metal (not streamed), read every token.

The Metal kernel `hc_sinkhorn_collapse` runs 20-iteration Sinkhorn normalization.
Falls back to `@mx.compile`d Python ops if Metal is unavailable.

## HyperConnection Hidden State Shape

The hidden state is 4D throughout the model:
```
(B, L, hc_mult=4, hidden_size=4096)
```

`embed_tokens` produces `(B, L, H)`, then broadcast to `(B, L, 4, H)` with
`mx.contiguous()` — 4× the memory of a single hidden state.

## Expert Sizes

Each routed expert has three projections:

| Projection | Shape | Operation |
|---|---|---|
| `gate_proj` | `(moe_intermediate_size=2048, hidden_size=512)` | ×½ hidden (packed) |
| `up_proj` | `(moe_intermediate_size=2048, hidden_size=512)` | ×½ hidden (packed) |
| `down_proj` | `(hidden_size=4096, moe_intermediate_size=256)` | ×½ intermediate (packed) |

All stored as `uint32` (packed mxfp4, 2 values per uint32) + `uint8` scales.
Record size per expert: **13,369,344 bytes** (~12.75 MB). See [04-weights.md](04-weights.md).

## Speculative Decoding Drafter

An MTP (Multi-Token Prediction) drafter is attached at `num_nextn_predict_layers=1`.
It shares weights with the main model. `BufferedRotatingKVCache` in `cache.py`
supports the rollback needed for speculative decoding.

The drafter is not currently wired into the measurement loop. Integrating it
would reduce expert loads per accepted final token and lower `decode_sec_per_token`.
