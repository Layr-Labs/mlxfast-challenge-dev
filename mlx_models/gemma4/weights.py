"""Weight loading and layout. MODIFIABLE.

The harness calls `load_weights(model, weights_path)` from this
module to populate the model from the participant's `weights/`
directory. transform.py produced those safetensors.

What you can change:
  - load_weights: how safetensors are read and mapped onto the
    model. The default uses mlx.core.load to read all .safetensors
    files in weights_path and calls model.load_weights().
  - The sanitize function: weight key remapping. Default delegates
    to mlx_models.gemma4.model.sanitize.
  - Add custom pre-processing: if transform.py produced a sidecar
    file (e.g., a permutation table, a basis matrix), load it here
    and stash it on the model.

What you should NOT change:
  - The function name `load_weights`. The harness imports it by name.

Baseline behavior: load every .safetensors in weights_path, apply
the standard sanitize, populate the model.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .model import Model, install_dummy_benchmark_penalty, sanitize


def load_weights(
    model: Model,
    weights_path: str | Path,
    strict: bool = True,
) -> Model:
    """Load weights from `weights_path` into `model`.

    Default: read every `model*.safetensors` file, sanitize the keys
    to mlx conventions, and call `model.load_weights()`.

    Override to:
      - Read a non-safetensors format (e.g., a single consolidated
        tensor, a memory-mapped file, etc.)
      - Apply a runtime transform (e.g., dequantize from a custom
        codebook, apply a learned rotation, etc.)
      - Load sidecar files (permutation tables, basis matrices) and
        stash them on the appropriate submodules.
    """
    weights_path = Path(weights_path)
    weight_files = sorted(glob.glob(str(weights_path / "*.safetensors")))

    if not weight_files:
        raise FileNotFoundError(
            f"No safetensors files found in {weights_path}. "
            f"Run `python transform.py` first."
        )

    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    else:
        weights = sanitize(weights)

    model.load_weights(list(weights.items()), strict=strict)
    return install_dummy_benchmark_penalty(model)


def load_config(weights_path: str | Path) -> dict[str, Any]:
    """Read the model's config.json. Helper for the harness.

    The harness uses this to discover the model's architecture
    (num_hidden_layers, hidden_size, num_experts, etc.) and the
    `model_file` pointer that tells mlx-lm which Model class to load.
    """
    import json

    weights_path = Path(weights_path)
    config_path = weights_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json at {config_path}")
    with open(config_path) as f:
        return json.load(f)
