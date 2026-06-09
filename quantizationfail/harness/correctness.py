"""Correctness gate. FROZEN.

The challenge spec says: hidden states at every layer must match the
reference exactly up to floating point associativity. We implement
this as layer-wise comparison of the activations flowing out of
every DecoderLayer.

Implementation strategy:
  1. Load the reference model (mlx_lm's standard load, no modifiable
     surface involvement).
  2. Load the submission model (using the participant's modifiable
     surface).
  3. Run both on the same input tokens (seeded at runtime by the
     harness from a server-side secret + commit hash).
  4. At every DecoderLayer, capture the output hidden state.
  5. Compare with allclose using CORRECTNESS_EPSILON.

mlx has no `register_forward_hook` (torch idiom) and method-binding
tricks on `nn.Module` subclasses don't work because the model code
calls `layer(x, ...)` which uses a metaclass-bound `__call__`, not
a per-instance `__call__`.

We work around this by manually iterating the model's layers and
calling each one. We use the model's own helper methods
(`_make_masks`, `_get_per_layer_inputs`, `_project_per_layer_inputs`)
for the per-layer setup, then capture the output of each layer call
in our own loop.

This is model-specific (it knows the Gemma 4 architecture) but
cleaner than fighting mlx's module system.

A pass means every layer is within tolerance. A fail reports the
first diverging layer and the magnitude of the difference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx

from .constants import CORRECTNESS_EPSILON


@dataclass
class CorrectnessResult:
    passed: bool
    num_layers: int
    first_failing_layer: Optional[int] = None
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    failing_layer_diffs: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "num_layers": self.num_layers,
            "first_failing_layer": self.first_failing_layer,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
        }


def _run_layers_capturing(inner_model, tokens: mx.array, cache=None):
    """Manually run inner_model's layers and capture h after each.

    Returns a list of mx.array, one per layer, holding the hidden
    state at the output of that layer (before the final norm).

    Works for Gemma 4 TextModel (gemma4_text.Gemma4TextModel). The
    inner_model must have:
      - embed_tokens
      - embed_scale
      - hidden_size_per_layer_input (int, may be 0)
      - previous_kvs (list[int])
      - layers (list[DecoderLayer])
      - _make_masks, _get_per_layer_inputs, _project_per_layer_inputs
    """
    # 1. Embed.
    input_embeddings = inner_model.embed_tokens(tokens)
    h = input_embeddings * inner_model.embed_scale

    # 2. Per-layer inputs (if applicable).
    if inner_model.hidden_size_per_layer_input:
        per_layer_inputs = inner_model._get_per_layer_inputs(tokens, input_embeddings)
        per_layer_inputs = inner_model._project_per_layer_inputs(h, per_layer_inputs)
        per_layer_inputs = [per_layer_inputs[:, :, i, :] for i in range(len(inner_model.layers))]
    else:
        per_layer_inputs = [None] * len(inner_model.layers)

    # 3. Cache.
    if cache is None:
        cache = [None] * len(inner_model.layers)
    else:
        cache = list(cache) + [None] * (len(inner_model.layers) - len(cache))

    # 4. Build masks.
    masks = inner_model._make_masks(h, cache)

    # 5. Iterate layers, capture h.
    intermediates: List[mx.array] = []
    kvs_offsets = [(None, None)] * len(inner_model.layers)
    for idx, (layer, c, mask, prev_idx, pli) in enumerate(zip(
        inner_model.layers, cache, masks, inner_model.previous_kvs, per_layer_inputs
    )):
        kvs, offset = kvs_offsets[prev_idx]
        h, kvs, offset = layer(
            h, mask, c,
            per_layer_input=pli,
            shared_kv=kvs,
            offset=offset,
        )
        kvs_offsets[idx] = (kvs, offset)
        intermediates.append(h)

    return intermediates


def _capture_intermediates(model, tokens: mx.array) -> List[mx.array]:
    """Run the model and capture the hidden state at the output of
    every DecoderLayer. Returns a list of length num_hidden_layers.
    """
    # The model may be nested up to two levels deep:
    #   gemma4.Model
    #     .language_model = gemma4_text.Model
    #       .model = Gemma4TextModel    (has embed_tokens, layers, etc.)
    # Drill down to the innermost one that has embed_tokens, walking
    # through any sub-module that itself contains a sub-module with
    # embed_tokens.
    def _find_with_embed_tokens(mod, depth=0):
        if depth > 3:
            return None
        if hasattr(mod, "embed_tokens"):
            return mod
        for attr in ("model", "language_model", "text_model"):
            if hasattr(mod, attr):
                found = _find_with_embed_tokens(getattr(mod, attr), depth + 1)
                if found is not None:
                    return found
        return None

    inner = _find_with_embed_tokens(model)
    if inner is None:
        raise RuntimeError(
            f"Cannot find embed_tokens on model of type {type(model).__name__}"
        )

    cache = inner.make_cache() if hasattr(inner, "make_cache") else None
    intermediates = _run_layers_capturing(inner, tokens, cache=cache)
    mx.eval(intermediates)
    return intermediates


def check(
    reference_model,
    submission_model,
    tokens: mx.array,
    epsilon: float = CORRECTNESS_EPSILON,
) -> CorrectnessResult:
    """Compare layer-wise hidden states of two models on the same
    input tokens. Returns a CorrectnessResult.

    Both models must have the same num_hidden_layers and accept
    tokens of the same shape.
    """
    ref_intermediates = _capture_intermediates(reference_model, tokens)
    sub_intermediates = _capture_intermediates(submission_model, tokens)

    if len(ref_intermediates) != len(sub_intermediates):
        return CorrectnessResult(
            passed=False,
            num_layers=len(ref_intermediates),
            first_failing_layer=0,
            max_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
        )

    num_layers = len(ref_intermediates)
    max_abs = 0.0
    max_rel = 0.0
    first_failing: Optional[int] = None
    failing_diffs: List[float] = []

    for i, (ref_h, sub_h) in enumerate(zip(ref_intermediates, sub_intermediates)):
        diff = mx.abs(ref_h - sub_h)
        abs_diff = float(mx.max(diff))
        ref_abs = mx.abs(ref_h)
        rel = mx.where(ref_abs > 1e-6, diff / ref_abs, mx.zeros_like(diff))
        rel_diff = float(mx.max(rel))

        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, rel_diff)

        if abs_diff > epsilon:
            if first_failing is None:
                first_failing = i
            failing_diffs.append(abs_diff)

    passed = first_failing is None
    return CorrectnessResult(
        passed=passed,
        num_layers=num_layers,
        first_failing_layer=first_failing,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        failing_layer_diffs=failing_diffs,
    )
