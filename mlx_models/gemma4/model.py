"""Model class: attention, MoE routing, layer definitions. MODIFIABLE.

This is the top-level Model class. The harness loads it via mlx-lm's
`model_file` config escape hatch: config.json in weights/ has
`"model_file": "../mlx_models/gemma4/model.py"`, and mlx-lm imports
this file directly when constructing the model.

What you can change:
  - The Model class itself: layer count, attention pattern, MoE
    activation, etc. The harness reads num_hidden_layers,
    hidden_size, num_experts, moe_intermediate_size, etc. from
    config.json and passes them to ModelArgs. ModelArgs is defined
    here and should match what config.json provides.
  - The Attention/DecoderLayer classes: if you want a custom
    attention pattern (e.g., per-layer Hadamard) or a different
    decoder structure, redefine them here.
  - The sanitize function: how raw safetensors are mapped onto
    model parameters. Default: pass-through to upstream.

What you should NOT change:
  - The class names. config.json expects "Model" and "ModelArgs"
    attributes on this module (the standard mlx-lm model_file
    contract).
"""
from __future__ import annotations

from mlx_lm.models.gemma4 import Model as _UpstreamModel
from mlx_lm.models.gemma4 import ModelArgs as _UpstreamModelArgs


class ModelArgs(_UpstreamModelArgs):
    """Inherit all upstream fields. Add custom fields if needed.

    The harness reads these from config.json and instantiates the
    model. To add a new field (e.g., a custom routing hyperparam),
    add it here and ensure config.json provides a value.
    """

    pass


class Model(_UpstreamModel):
    """Inherit the upstream Gemma 4 text model verbatim.

    Default: pass-through. Override __init__ to add custom buffers,
    override __call__ to add pre/post hooks, or override sanitize
    to remap weights from a transformed representation.

    The key thing: this class is constructed by mlx-lm's standard
    load_model() flow, so the __init__ signature must accept a
    ModelArgs.
    """

    pass


def sanitize(weights: dict) -> dict:
    """Default: delegate to upstream. Override to remap weight keys.

    The harness calls this after loading safetensors and before
    calling model.load_weights(). The default upstream sanitize
    handles the standard HF-to-mlx conversion (dropping KV-shared
    projections, splitting experts.gate_up_proj into gate_proj/up_proj,
    etc.).
    """
    return _UpstreamModel.sanitize(weights)
