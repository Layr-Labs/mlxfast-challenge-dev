# Weight Formats

## Reference Checkpoint (`mlxfast/reference_weights/DeepSeek-V4-Flash-4bit/`)

Original mlx-community checkpoint. Managed by `mlxfast weights` (downloads via
`huggingface_hub.snapshot_download`). Read-only — participants must not modify it.

### Key Format

Expert weights use per-expert keys:
```
model.layers.{L}.ffn.experts.{E}.w1.weight   → gate_proj weight (uint32, packed mxfp4)
model.layers.{L}.ffn.experts.{E}.w1.scales   → gate_proj scales (uint8)
model.layers.{L}.ffn.experts.{E}.w2.weight   → down_proj weight
model.layers.{L}.ffn.experts.{E}.w2.scales   → down_proj scales
model.layers.{L}.ffn.experts.{E}.w3.weight   → up_proj weight
model.layers.{L}.ffn.experts.{E}.w3.scales   → up_proj scales
```

Where L = 0..42, E = 0..255. Total expert weight: ~147 GB.

Dense weights (attention, embeddings, LM head, shared experts, HC, gates) use:
```
model.layers.{L}.self_attn.*
model.layers.{L}.ffn.shared_experts.*
model.layers.{L}.ffn.gate.*
model.layers.{L}.ffn_hc.*
model.layers.{L}.attn_hc.*
model.embed_tokens.weight
model.norm.weight
lm_head.weight
```

### Loading Quirk — `_force_sanitize_load`

The checkpoint's safetensors files carry `"format": "mlx"` metadata.
`mlx_vlm.load` sees this and skips `sanitize()`. But DeepSeek V4 Flash still needs
sanitize to remap `model.*` → `language_model.model.*` key paths.

The harness patches `safetensors.safe_open` to strip the format metadata flag,
forcing the sanitize path. Additionally, `config.json` quantization override keys
use `model.layers.N.*` prefix but post-sanitize paths are `language_model.model.layers.N.*`;
the harness patches `mlx_vlm.utils.load_config` to remap them.

These patches are applied inside a `try/finally` block and always restored.

---

## Transformed Weights (`weights/`)

Output of `transform.py`. Must be re-creatable byte-for-byte by re-running
`transform.py` (verified by `_sandbox.verify_transform`).

### Dense Safetensors

All non-expert weights: attention, embeddings, LM head, shared experts,
HyperConnection matrices, MoE gates. These are loaded into Metal and stay
resident for the entire run (~4–6 GB).

Must include `config.json` with the architecture parameters. The harness
reads the following fields from it:
```json
{
  "num_hidden_layers": 43,
  "n_routed_experts": 256,
  "num_experts_per_tok": 6,
  "vocab_size": 129280
}
```
*(Only these 4 are checked. See [11-integrity.md §invariants](11-integrity.md).)*

Also requires `model.safetensors.index.json` for multi-shard loading.

### Expert Binary Files (`weights/experts/`)

One `.bin` file per layer: `layer_00.bin` through `layer_42.bin`.

Each file stores 256 fixed-size records packed contiguously:
```
file[j * record_size : (j+1) * record_size] = expert j's record
```

#### Record Layout

Fields within each record, in this order:

| Field | dtype | Shape | Bytes |
|---|---|---|---|
| `down_proj.weight` | uint32 | [4096, 256] | 4,194,304 |
| `down_proj.scales` | uint8 | [4096, 64] | 262,144 |
| `gate_proj.weight` | uint32 | [2048, 512] | 4,194,304 |
| `gate_proj.scales` | uint8 | [2048, 128] | 262,144 |
| `up_proj.weight` | uint32 | [2048, 512] | 4,194,304 |
| `up_proj.scales` | uint8 | [2048, 128] | 262,144 |
| **Total** | | | **13,369,344 B (~12.75 MB)** |

#### `manifest.json`

Located at `weights/experts/manifest.json`. Describes the byte layout so the
runtime can parse records without hardcoded offsets.

Schema:
```json
{
  "record_size": 13369344,
  "tensors": {
    "down_proj": {
      "weight": {"dtype": "uint32", "shape": [4096, 256], "offset_in_record": 0,       "nbytes": 4194304},
      "scales":  {"dtype": "uint8",  "shape": [4096, 64],  "offset_in_record": 4194304, "nbytes": 262144}
    },
    "gate_proj": {
      "weight": {"dtype": "uint32", "shape": [2048, 512], "offset_in_record": 4456448,  "nbytes": 4194304},
      "scales":  {"dtype": "uint8",  "shape": [2048, 128], "offset_in_record": 8650752,  "nbytes": 262144}
    },
    "up_proj": {
      "weight": {"dtype": "uint32", "shape": [2048, 512], "offset_in_record": 8912896,  "nbytes": 4194304},
      "scales":  {"dtype": "uint8",  "shape": [2048, 128], "offset_in_record": 13107200, "nbytes": 262144}
    }
  }
}
```

**The runtime uses `manifest.json` for all record parsing.** Participants can
change `record_size`, tensor order, dtypes, and shapes in `transform.py` as long
as `manifest.json` is updated to match. `ExpertSlotBank._load()` reads
`manifest.json` on startup and uses it for all `os.pread` calls.

### `transform.py` Contract

- **Input**: `mlxfast/reference_weights/DeepSeek-V4-Flash-4bit/` (read-only)
- **Output**: `weights/` (written fresh; existing contents overwritten)
- **Must be deterministic**: same input → byte-for-byte identical output
- **No network access**: verified by `_sandbox.py` audit hooks
- **No subprocess spawns**: verified by `_sandbox.py` audit hooks
- **No environment mutation**: verified by `_sandbox.py` audit hooks
- **No fork**: verified by `_sandbox.py` audit hooks
- **No new ctypes/CDLL loads**: verified by `_sandbox.py` audit hooks

Participants may use any installed Python packages. Pre-loaded native extensions
are not blocked (residual risk — see [13-issues.md §10](13-issues.md)).
