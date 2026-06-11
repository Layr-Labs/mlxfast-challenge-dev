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

import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gemma4 import Model as _UpstreamModel
from mlx_lm.models.gemma4 import ModelArgs as _UpstreamModelArgs

# Local test knobs. Set these down toward zero to simulate improvements.
DUMMY_PEAK_RAM_GB = 1.0
DUMMY_BANDWIDTH_GB_PER_TOKEN = 1.0
DUMMY_SECONDS_PER_TOKEN = 0.005


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

    def load_weights(self, file_or_weights, strict: bool = True):
        result = super().load_weights(file_or_weights, strict=strict)
        install_dummy_benchmark_penalty(self)
        return result


class DummyBenchmarkPenalty(nn.Module):
    """Wrapper that adds local-only benchmark penalties without changing outputs."""

    def __init__(self, wrapped: nn.Module):
        super().__init__()
        self.wrapped = wrapped
        bandwidth_bytes = max(0, int(DUMMY_BANDWIDTH_GB_PER_TOKEN * 1024**3))
        ram_bytes = max(0, int(DUMMY_PEAK_RAM_GB * 1024**3))
        if bandwidth_bytes:
            self.dummy_bandwidth = mx.zeros((bandwidth_bytes,), dtype=mx.uint8)
        if ram_bytes:
            self._dummy_ram = mx.zeros((ram_bytes,), dtype=mx.uint8)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            wrapped = dict.get(self, "wrapped")
            if wrapped is None:
                raise
            return getattr(wrapped, name)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings=None,
        per_layer_inputs=None,
    ):
        if DUMMY_SECONDS_PER_TOKEN > 0:
            tokens = int(inputs.shape[-1]) if hasattr(inputs, "shape") and inputs.shape else 1
            time.sleep(DUMMY_SECONDS_PER_TOKEN * max(1, tokens))
        return self.wrapped(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=per_layer_inputs,
        )


def install_dummy_benchmark_penalty(model: Model) -> Model:
    """Attach local dummy benchmark penalties after strict weight loading."""
    if hasattr(model, "language_model") and not isinstance(model.language_model, DummyBenchmarkPenalty):
        model.language_model = DummyBenchmarkPenalty(model.language_model)
        arrays = []
        if hasattr(model.language_model, "dummy_bandwidth"):
            arrays.append(model.language_model.dummy_bandwidth)
        if hasattr(model.language_model, "_dummy_ram"):
            arrays.append(model.language_model._dummy_ram)
        if arrays:
            mx.eval(*arrays)
    return model


def sanitize(weights: dict) -> dict:
    """Default: delegate to upstream. Override to remap weight keys.

    The harness calls this after loading safetensors and before
    calling model.load_weights(). The default upstream sanitize
    handles the standard HF-to-mlx conversion (dropping KV-shared
    projections, splitting experts.gate_up_proj into gate_proj/up_proj,
    etc.).
    """
    return _UpstreamModel.sanitize(weights)
