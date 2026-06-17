from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel as _LanguageModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = _LanguageModel(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        # Correctness harness passes ref_tok[None] which can produce a 3-D
        # (batch, seq, 1) tensor.  Squeeze the trailing size-1 dimension so
        # embed_tokens always receives (batch, seq) integer indices.
        if input_ids.ndim == 3 and input_ids.shape[-1] == 1:
            input_ids = input_ids.squeeze(-1)
        return self.language_model(input_ids, cache=cache, **kwargs)

    def sanitize(self, weights):
        weights = self.language_model.sanitize(weights)

        def transform_key(key):
            if key.startswith("language_model."):
                return key
            if key.startswith("model.") or key.startswith("lm_head."):
                return f"language_model.{key}"
            return key

        return {transform_key(k): v for k, v in weights.items()}

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def load_weights(self, weights, strict: bool = True):
        # mlx_vlm skips sanitize when is_mlx_format=True (format: mlx metadata).
        # The DS4 Flash checkpoint needs sanitize to remap 'model.*' keys to
        # 'language_model.model.*'. Call it here so load works in both cases.
        weight_dict = dict(weights)
        weight_dict = self.sanitize(weight_dict)
        weights = list(weight_dict.items())

        # Expert weights are loaded on-demand by StreamingSwitchGLU from
        # weights/experts/*.bin. Filter them from the safetensors payload so
        # strict-mode load_weights doesn't reject them as unexpected.
        def _is_expert_weight(key: str) -> bool:
            return (
                ".ffn.switch_mlp." in key
                and any(x in key for x in (".gate_proj.", ".up_proj.", ".down_proj."))
                and ".shared_experts." not in key
            )
        filtered = [(k, v) for k, v in weights if not _is_expert_weight(k)]
        result = super().load_weights(filtered, strict=strict)
        experts_dir = Path("weights") / "experts"
        ref_dir = self._find_reference_dir()
        if ref_dir is not None:
            self._configure_safetensors_streaming(ref_dir, str(experts_dir))
        return result

    @staticmethod
    def _find_reference_dir() -> Optional[str]:
        """Return the path to the HF reference weights directory, or None."""
        candidates = [
            Path("mlxfast") / "reference_weights" / "DeepSeek-V4-Flash-4bit",
            Path("reference_weights") / "DeepSeek-V4-Flash-4bit",
            Path("weights") / "reference",
        ]
        for p in candidates:
            if (p / "model.safetensors.index.json").exists():
                return str(p)
        return None

    def _configure_safetensors_streaming(
        self, reference_dir: str, experts_dir: str
    ) -> None:
        """Wire the global slot bank (safetensors mode) and set layer indices."""
        from mlx_models.mlx_lm_shims.switch_layers import configure_safetensors_streaming
        configure_safetensors_streaming(reference_dir, experts_dir)
        self._assign_layer_indices()

    def _assign_layer_indices(self) -> None:
        """Set _layer_idx on every StreamingSwitchGLU in the model."""

        # Collect all MoE switch layers in order.
        moe_layers: list[tuple[int, object]] = []
        for layer_idx, layer in enumerate(self.language_model.model.layers):
            mlp = getattr(layer, "ffn", None)
            switch = getattr(mlp, "switch_mlp", None)
            if switch is not None:
                switch._layer_idx = layer_idx
                moe_layers.append((layer_idx, switch))

