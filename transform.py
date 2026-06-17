"""
transform.py — Baseline weight transform for the DeepSeek V4 Flash challenge.

Copies non-expert (dense) weights from the reference checkpoint to weights/,
along with config, tokenizer, and a stripped model.safetensors.index.json.
Expert weights are NOT copied; they are read on-demand via pread() directly
from the reference safetensors shards at inference time (ExpertSlotBank).

Output layout
─────────────
  weights/
    config.json
    tokenizer.json          (and other tokenizer files)
    *.safetensors           non-expert weights only (attention, shared experts,
                            embeddings, LM head, hyper connections, gates)
    model.safetensors.index.json   (expert keys stripped out)

Participants may replace this file with any transform they choose, provided:
  1. weights/config.json is valid and contains the frozen fields unchanged.
  2. The layout is fully deterministic given the reference checkpoint.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import mlx.core as mx

# ─── paths ────────────────────────────────────────────────────────────────────

REFERENCE_DIR = Path("mlxfast/reference_weights")
OUTPUT_DIR    = Path("weights")
EXPERTS_DIR   = OUTPUT_DIR / "experts"

# ─── helpers ──────────────────────────────────────────────────────────────────

def _find_reference_dir(base: Path) -> Path:
    """Return the directory that contains config.json."""
    if (base / "config.json").exists():
        return base
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
    from safetensors import safe_open
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


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    model_dir = _find_reference_dir(REFERENCE_DIR)
    print(f"Reference checkpoint: {model_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERTS_DIR.mkdir(parents=True, exist_ok=True)

    key_to_shard = _all_keys_by_shard(model_dir)
    dense_keys = {k: v for k, v in key_to_shard.items() if not _is_expert_key(k)}

    print(f"Total tensors: {len(key_to_shard)}  "
          f"Dense: {len(dense_keys)}  "
          f"Expert (skipped): {len(key_to_shard) - len(dense_keys)}")

    # ── pass 1: write dense-only shard files ──────────────────────────────
    # Extract only non-expert keys from each source shard and save to new
    # (much smaller) safetensors files.  mx.load returns lazy arrays so only
    # the keys we access are read from disk — stacked expert tensors in the
    # same shard are never loaded.
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

    # ── pass 2: copy tokenizer + config files ─────────────────────────────
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

    print(f"\nDone. Output: {OUTPUT_DIR.resolve()}")
    print("Expert weights will be pread on-demand from reference shards at inference time.")


if __name__ == "__main__":
    main()
