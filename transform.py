"""
transform.py — Baseline weight transform for the DeepSeek V4 Flash challenge.

Splits routed-expert weights out of the reference checkpoint into per-layer
binary files for SSD streaming. Non-expert weights are copied as-is.

Output layout
─────────────
  weights/
    config.json
    tokenizer.json          (and other tokenizer files)
    *.safetensors           non-expert weights (attention, shared experts,
                            embeddings, LM head, hyper connections, gates)
    experts/
      manifest.json         quantisation metadata + per-tensor byte offsets
      layer_00.bin          256 expert records, expert-major, fixed record size
      layer_01.bin
      ...
      layer_42.bin

Binary record format
────────────────────
  layer_NN.bin = [record_0][record_1]...[record_255]

  record_j contains the packed arrays for expert j across all 3 projections:
    [gate_proj.weight][gate_proj.scales][gate_proj.biases?]
    [up_proj.weight  ][up_proj.scales  ][up_proj.biases?  ]
    [down_proj.weight][down_proj.scales][down_proj.biases?]

  All tensors are written as raw bytes (numpy .tobytes(), row-major).
  Shapes, dtypes, and byte offsets within the record are stored in manifest.json.
  Records are fixed-size, so expert j starts at byte offset j * record_size.

Expert key format in the reference checkpoint
─────────────────────────────────────────────
The mlx-community/DeepSeek-V4-Flash-4bit checkpoint uses per-expert keys:
  layers.{layer}.ffn.experts.{expert}.w1.{weight|scale}   ← gate_proj
  layers.{layer}.ffn.experts.{expert}.w2.{weight|scale}   ← down_proj
  layers.{layer}.ffn.experts.{expert}.w3.{weight|scale}   ← up_proj

This transform reads those per-expert keys directly, avoiding the need to
load and then un-stack the full (256, out, in) stacked tensors.

If the checkpoint already has stacked keys (model.layers.*.ffn.switch_mlp.*),
the fallback path handles that format too.

Participants replacing this file may use any layout they choose, provided:
  1. weights/config.json is valid and contains the frozen fields unchanged.
  2. weights/experts/manifest.json exists and is readable by ExpertSlotBank.
  3. The layout is fully deterministic given the reference checkpoint.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# ─── paths ────────────────────────────────────────────────────────────────────

REFERENCE_DIR = Path("mlxfast/reference_weights")
OUTPUT_DIR    = Path("weights")
EXPERTS_DIR   = OUTPUT_DIR / "experts"

# Projection name mapping: checkpoint key → canonical name used in manifest
_W_REMAP = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

# ─── helpers ──────────────────────────────────────────────────────────────────

def _find_reference_dir(base: Path) -> Path:
    """Return the directory that contains config.json.

    Follows symlinks when traversing subdirectories (rglob does not follow
    directory symlinks in Python ≤3.12, so we use os.walk with followlinks).
    """
    if (base / "config.json").exists():
        return base
    # os.walk follows symlinked subdirectories when followlinks=True.
    for root, _dirs, files in os.walk(base, followlinks=True):
        if "config.json" in files:
            return Path(root)
    raise FileNotFoundError(
        f"No config.json found under {base}. "
        "Run: mlxfast weights   to download the reference checkpoint."
    )


def _all_keys_by_shard(model_dir: Path) -> dict[str, str]:
    """Return {tensor_key: shard_filename} for the whole checkpoint."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    result: dict[str, str] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="numpy") as f:
            for k in f.keys():
                result[k] = shard.name
    return result


def _is_expert_key(key: str) -> bool:
    """True if key belongs to a routed (non-shared) expert."""
    return (
        ".ffn.experts." in key or ".ffn.switch_mlp." in key
    ) and ".shared_experts." not in key


def _layer_from_key(key: str) -> Optional[int]:
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p in ("layers", "layer") and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def _expert_from_key(key: str) -> Optional[int]:
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "experts" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def _proj_and_type(key: str) -> Optional[tuple[str, str]]:
    """Return (canonical_proj_name, tensor_type) or None.

    Handles w1/w2/w3 (per-expert format) and gate_proj/up_proj/down_proj
    (stacked format). tensor_type is 'weight', 'scales', or 'biases'.
    """
    key_lower = key.lower()
    for wN, proj in _W_REMAP.items():
        for suffix in ("weight", "scales", "scale", "biases", "bias"):
            if f".{wN}.{suffix}" in key_lower:
                ttype = "scales" if suffix in ("scale", "scales") else suffix
                ttype = "biases" if ttype == "bias" else ttype
                return proj, ttype
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("weight", "scales", "biases"):
            if f".{proj}.{suffix}" in key_lower:
                return proj, suffix
    return None


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    model_dir = _find_reference_dir(REFERENCE_DIR)
    print(f"Reference checkpoint: {model_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── pass 1: categorise every key ──────────────────────────────────────
    key_to_shard = _all_keys_by_shard(model_dir)

    # (layer, expert, proj, ttype) → (shard_filename, key)
    expert_keys: dict[tuple[int, int, str, str], tuple[str, str]] = {}
    dense_keys:  dict[str, str] = {}   # key → shard_filename

    # (layer, proj, ttype) → (shard_filename, key) for stacked format
    stacked_keys: dict[tuple[int, str, str], tuple[str, str]] = {}

    for key, shard in key_to_shard.items():
        if _is_expert_key(key):
            layer  = _layer_from_key(key)
            expert = _expert_from_key(key)
            parsed = _proj_and_type(key)
            if layer is None or parsed is None:
                print(f"  [warn] unparseable expert key, treating as dense: {key}")
                dense_keys[key] = shard
                continue
            proj, ttype = parsed
            if expert is None:
                # Stacked format: all experts in one tensor (axis 0 = expert index).
                stacked_keys[(layer, proj, ttype)] = (shard, key)
            else:
                expert_keys[(layer, expert, proj, ttype)] = (shard, key)
        else:
            dense_keys[key] = shard

    # Expand stacked keys into per-expert entries.
    if stacked_keys and not expert_keys:
        print("  Detected stacked expert format — expanding per-expert records...")
        # Determine n_experts from the first stacked weight tensor.
        first_layer = min(l for l, _, t in stacked_keys if t == "weight")
        first_proj  = next(p for (l, p, t) in stacked_keys if l == first_layer and t == "weight")
        shard_name, key = stacked_keys[(first_layer, first_proj, "weight")]
        with safe_open(str(model_dir / shard_name), framework="numpy") as f:
            n_experts = f.get_tensor(key).shape[0]
        print(f"  Experts per layer: {n_experts}")
        # Build per-expert key map by recording which stacked key to slice for each expert.
        # We use a sentinel expert index of -1 to mean "read from stacked".
        for (layer, proj, ttype), (shard, key) in stacked_keys.items():
            for e in range(n_experts):
                expert_keys[(layer, e, proj, ttype)] = (shard, key)
        # Mark stacked keys so the loader knows to slice by axis 0.
        _STACKED_KEYS: set[str] = {key for (_, key) in stacked_keys.values()}
    else:
        _STACKED_KEYS = set()
        n_experts = max(e for _, e, *_ in expert_keys) + 1 if expert_keys else 0

    if not expert_keys:
        raise RuntimeError(
            "No expert keys found in checkpoint. "
            "Expected keys containing '.ffn.experts.' or '.ffn.switch_mlp.'."
        )

    n_layers  = max(l for l, *_ in expert_keys) + 1
    projs     = sorted({p for _, _, p, _ in expert_keys})

    print(f"MoE layers: {n_layers}  experts/layer: {n_experts}  "
          f"projections: {projs}")
    print(f"Dense tensors: {len(dense_keys)}  Expert tensors: {len(expert_keys)}")

    # ── pass 2: compute manifest from layer-0 expert-0 shapes ─────────────
    manifest_projs: dict[str, dict] = {}
    record_size = 0
    running_offset = 0

    for proj in projs:
        manifest_projs[proj] = {}
        for ttype in ("weight", "scales", "biases"):
            entry = expert_keys.get((0, 0, proj, ttype))
            if entry is None:
                manifest_projs[proj][ttype] = None
                continue
            shard_name, key = entry
            with safe_open(str(model_dir / shard_name), framework="numpy") as f:
                arr = f.get_tensor(key)
            # Stacked format: arr shape is (n_experts, ...) — slice expert 0.
            if key in _STACKED_KEYS:
                arr = arr[0]
            # mxfp4 weights are stored as uint8 in the checkpoint but must be
            # viewed as uint32 for gather_qmm (matches language.py sanitize).
            if ttype == "weight" and arr.dtype == np.uint8:
                arr = arr.view(np.uint32)
            nbytes = int(arr.nbytes)
            manifest_projs[proj][ttype] = {
                "dtype":            arr.dtype.str,
                "shape":            list(arr.shape),
                "nbytes":           nbytes,
                "offset_in_record": running_offset,
            }
            running_offset += nbytes
            record_size    += nbytes

    manifest = {
        "num_layers":  n_layers,
        "num_experts": n_experts,
        "record_size": record_size,
        "projections": manifest_projs,
        "quant": {"group_size": 32, "bits": 4, "mode": "mxfp4"},
    }
    with open(EXPERTS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written — record_size = {record_size:,} B "
          f"({record_size / 1e6:.1f} MB per expert)")

    # ── pass 3: write per-layer binary files ──────────────────────────────
    # Keep shard file descriptors open for the duration; close in finally.
    needed_shards = {s for s, _ in expert_keys.values()}
    handles: dict[str, safe_open] = {}
    try:
        for shard_name in needed_shards:
            handles[shard_name] = safe_open(
                str(model_dir / shard_name), framework="numpy"
            )

        for layer_idx in range(n_layers):
            buf = bytearray(n_experts * record_size)

            for expert_idx in range(n_experts):
                rec_start = expert_idx * record_size
                for proj in projs:
                    for ttype in ("weight", "scales", "biases"):
                        meta = manifest_projs[proj].get(ttype)
                        if meta is None:
                            continue
                        entry = expert_keys.get(
                            (layer_idx, expert_idx, proj, ttype)
                        )
                        if entry is None:
                            raise KeyError(
                                f"Missing: layer={layer_idx} expert={expert_idx} "
                                f"proj={proj} type={ttype}"
                            )
                        shard_name, key = entry
                        arr = handles[shard_name].get_tensor(key)
                        # Stacked format: slice expert axis.
                        if key in _STACKED_KEYS:
                            arr = arr[expert_idx]
                        if ttype == "weight" and arr.dtype == np.uint8:
                            arr = arr.view(np.uint32)
                        raw = arr.tobytes()
                        start = rec_start + meta["offset_in_record"]
                        buf[start : start + len(raw)] = raw

            bin_path = EXPERTS_DIR / f"layer_{layer_idx:02d}.bin"
            bin_path.write_bytes(buf)
            print(f"  layer_{layer_idx:02d}.bin  "
                  f"{len(buf) / 1e6:.0f} MB")
    finally:
        for h in handles.values():
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass

    # ── pass 4: write dense-only shard files ──────────────────────────────
    # Extract only non-expert keys from each source shard and save to new
    # (much smaller) safetensors files.  We use mlx to handle bfloat16
    # (numpy doesn't support it).  mx.load returns lazy arrays so only the
    # keys we actually access are read from disk — stacked expert tensors
    # in the same shard file are never loaded.
    import mlx.core as mx

    print("Writing dense-only shard files...")

    # Group dense keys by source shard.
    shard_to_dense_keys: dict[str, list[str]] = {}
    for key, shard in dense_keys.items():
        if shard.endswith(".safetensors"):
            shard_to_dense_keys.setdefault(shard, []).append(key)

    for shard_name, keys in sorted(shard_to_dense_keys.items()):
        src = model_dir / shard_name
        dst = OUTPUT_DIR / shard_name
        if dst.exists():
            print(f"  {shard_name}  (exists, skip)")
            continue
        # Lazy load — only keys we access are read from disk.
        all_tensors = mx.load(str(src))
        dense_tensors = {k: all_tensors[k] for k in keys if k in all_tensors}
        mx.eval(dense_tensors)
        mx.save_safetensors(str(dst), dense_tensors)
        size_mb = dst.stat().st_size / 1e6
        print(f"  {shard_name}  {size_mb:.0f} MB  ({len(dense_tensors)} tensors)")

    # ── pass 5: copy tokenizer + config files ─────────────────────────────
    skip_names = {"model.safetensors.index.json"}
    for src in sorted(model_dir.iterdir()):
        if src.name in skip_names:
            continue
        if src.suffix in (".json", ".model", ".tiktoken") and not src.name.endswith(
            ".safetensors"
        ):
            shutil.copy2(src, OUTPUT_DIR / src.name)

    # Rewrite the index JSON with expert keys removed (they live in experts/).
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        index["weight_map"] = {
            k: v for k, v in index["weight_map"].items() if k in dense_keys
        }
        with open(OUTPUT_DIR / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    # ── summary ───────────────────────────────────────────────────────────
    total_gb = n_layers * n_experts * record_size / 1e9
    print(f"\nDone. Output: {OUTPUT_DIR.resolve()}")
    print(f"Expert files: {n_layers} layers × {n_experts} experts × "
          f"{record_size / 1e6:.1f} MB/expert = {total_gb:.1f} GB total")


if __name__ == "__main__":
    main()
