# Modifiable Surface

## What Participants Can Change

Everything NOT under `mlxfast/` is modifiable. The harness self-hash does not
cover `mlx_models/` or `transform.py` — only `mlxfast/*.py`.

### Primary Target: `mlx_models/mlx_lm_shims/switch_layers.py`

The expert streaming engine. Every optimization that reduces expert I/O cost
flows through here.

Key constants and targets:

| Target | Current Value | Optimization Lever |
|---|---|---|
| `SLOT_BANK_SIZE` | 128 | Larger = fewer misses, more Metal RAM |
| `os.pread` per expert | 13.37 MB read | Smaller `record_size` = faster reads |
| LRU data structure | `OrderedDict` | Alternative eviction policies |
| I/O model | synchronous `os.pread` | Async/thread-pool for overlap with GPU |
| Prefetching | none | Predict next layer's experts and pre-load |
| Prefill-to-decode seeding | none | Collect routing during prefill, pre-warm LRU |

**`ExpertSlotBank` contract** (required by harness via `_configure_streaming`):
- Must be instantiated as a global singleton accessible to `StreamingSwitchGLU`
- Must expose a `get(layer_idx, expert_idx)` method returning a dict:
  ```python
  {
    "gate_proj": {"weight": mx.array, "scales": mx.array},
    "up_proj":   {"weight": mx.array, "scales": mx.array},
    "down_proj":  {"weight": mx.array, "scales": mx.array},
  }
  ```
- The `mx.array` values may be lazy (Metal not yet allocated). `gather_qmm`
  will materialize them.
- Must accept `_layer_idx` attribute set on `StreamingSwitchGLU` instances

**`StreamingSwitchGLU.__call__` contract**:
- Signature: `__call__(self, x: mx.array, indices: mx.array) -> mx.array`
- `x` shape: `(B, L, hidden_size)` — sequence of token hidden states
- `indices` shape: `(B, L, K)` — top-K expert indices, values in [0, 255]
- Returns shape: `(B, L, K, hidden_size)` — one output per activated expert
  (the MoE wrapper in `language.py` applies routing scores and sums)

### `transform.py`

Offline weight conversion. No scoring cost — runs once before benchmarking.
Output must be deterministic (same input → same bytes).

**What can change**:
- Expert storage format (dtype, packing, compression)
- Record layout within `layer_NN.bin`
- Number of shards for dense safetensors
- Any offline preprocessing (e.g., quantization, delta compression, reordering)

**What must remain**:
- `weights/config.json` with the 4 checked architecture fields
- `weights/experts/manifest.json` parseable by `ExpertSlotBank._load()`
- `weights/model.safetensors.index.json` for shard loading
- Output must be deterministic

### `mlx_models/deepseek_v4/deepseek_v4.py`

Top-level `Model` class. Interface the harness requires:

| Method/Attribute | Required Behavior |
|---|---|
| `__call__(inputs, cache=None, **kwargs)` | Returns `LanguageModelOutput(logits=..., hidden_states=...)` |
| `make_cache()` | Returns list of cache objects, one per layer |
| `load_weights(weights, strict=False)` | Loads dense safetensors weights; filters expert keys |
| `sanitize(weights)` | Key remapping for mlx_vlm compatibility |
| `model.parameters()` | Returns all dense (non-expert) tensors for `mx.eval` |
| `_configure_streaming(experts_dir)` | Wires `ExpertSlotBank` to all `StreamingSwitchGLU` instances |

### `mlx_models/deepseek_v4/language.py`

All layer computation. Modifiable targets:

- `MoEGate` — routing function; currently uses `@mx.compile` + `argpartition`
- `DeepseekV4MoE.__call__` — expert dispatch and weight application
- `LocalAttention`, `CompressedAttention`, `SparseCompressedAttention` — attention modes
- `Compressor` — KV projection (read every token; quantizable)
- `DeepseekV4Block.__call__` — layer structure (attention + FFN sequence)
- `DeepseekV4Model.__call__` — full model forward pass

### `mlx_models/deepseek_v4/hyper_connection.py`

HyperConnection residual mixing. Modifiable targets:

- `HyperConnection.__call__` — the expand/collapse ops
- `_hc_sinkhorn_collapse_kernel` — custom Metal kernel (20 iterations Sinkhorn)
- `_hc_split_sinkhorn_ops` — Python fallback with `@mx.compile`

Reducing HC compute or bandwidth is a non-obvious optimization target
(~541 MB/token in HC matrices alone).

### `mlx_models/deepseek_v4/config.py`

`ModelConfig` dataclass. Most fields read by the model code during construction.
The harness does not check most fields — only the 4 architecture invariants.

### `mlx_models/cache.py`

KV cache implementations. The key optimization opportunity:

- `QuantizedKVCache` — quantize KV to 4-bit or 8-bit
- `TurboQuantKVCache` (in `turboquant.py`) — Metal-kernel-backed quantization
- Reducing KV size lowers both `peak_ram_GB` (~1.7 GB) and `bandwidth_GB_per_token`

### `mlx_models/speculative/drafters/deepseek_v4_mtp/`

MTP speculative decoding drafter. Shares weights with the main model.
Accepted draft tokens reduce the number of full expert loads per accepted token.
Bandwidth is normalized by accepted tokens.

Currently not wired into the measurement loop. Integrating speculative decoding
requires modifications to the decode loop in the participant's model code.

---

## What Participants Cannot Change

### Frozen Harness (`mlxfast/`)

| File | What It Does |
|---|---|
| `harness/run.py` | Measurement pipeline (correctness, prefill, decode, bandwidth) |
| `harness/bandwidth.py` | mactop integration and bandwidth formula |
| `harness/correctness.py` | Correctness gate implementation |
| `harness/score.py` | Score formula |
| `harness/constants.py` | All measurement constants |
| `cli.py` | `mlxfast run/submit/weights/login/clone` CLI |
| `_harness_runner.py` | Subprocess spawner |
| `_sandbox.py` | `transform.py` verification sandbox |
| `_self_hash.py` | Harness integrity check |

Modifying any `.py` file under `mlxfast/` changes the harness hash and causes
`mlxfast run` to refuse (once the server sets `MLXFAST_EXPECTED_HARNESS_HASH`).

### Frozen Tokenizer

`mlx_models/deepseek_v4/processing_deepseek_v4.py` — the tokenizer processor.
This is not hashed by the harness but should not be modified (server uses
the reference tokenizer for the correctness prompt).

### Reference Weights

`mlxfast/reference_weights/DeepSeek-V4-Flash-4bit/` — read-only input to
`transform.py`. Must not be modified.
