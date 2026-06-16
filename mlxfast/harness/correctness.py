"""Correctness gate. FROZEN.

Three independent layers per the challenge spec:

  Layer 1 — Greedy token sequence: both models must produce the
    exact same greedy token for every step in a CORRECTNESS_STEPS-
    step autoregressive decode.

  Layer 2 — Hidden state tolerance: at each decode step the final
    hidden state (post-norm, pre-lm_head) must agree within
    CORRECTNESS_EPSILON (absolute).

  Layer 3 — Top-K logit set: the set of the K highest-probability
    token IDs must match exactly at every step.

All three are evaluated on the same seeded prompt.  A submission
passes only if all three pass on every step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx

from .constants import CORRECTNESS_EPSILON

CORRECTNESS_STEPS = 256     # decode steps to check (spec §4.2 specifies 256)
CORRECTNESS_TOP_K = 10      # top-K logit set size for Layer 3


@dataclass
class CorrectnessResult:
    passed: bool
    num_layers: int           # model layer count (from config)
    first_failing_layer: Optional[int] = None   # decode step index of first failure
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


def _get_num_layers(model) -> int:
    """Return the transformer layer count from model config/args."""
    for attr in ("config", "args"):
        cfg = getattr(model, attr, None)
        if cfg is None:
            continue
        for key in ("num_hidden_layers", "n_layers", "num_layers"):
            v = getattr(cfg, key, None)
            if v is not None:
                return int(v)
    # Drill into language_model
    lm = getattr(model, "language_model", None)
    if lm is not None:
        return _get_num_layers(lm)
    layers = getattr(model, "layers", None)
    if layers is not None:
        return len(layers)
    return 0


def _make_cache(model):
    if hasattr(model, "make_cache"):
        return model.make_cache()
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "make_cache"):
        return lm.make_cache()
    return None


def _forward(model, token: mx.array, cache) -> tuple[mx.array, mx.array]:
    """Single decode step. Returns (logits_1d, hidden_1d).

    logits_1d: shape (vocab_size,) — the logit vector for the last position.
    hidden_1d: shape (hidden_size,) — post-norm hidden state for the last position.
    """
    out = model(token, cache=cache, return_hidden=True)
    logits = out.logits[0, -1]          # (vocab_size,)

    # hidden_states is a list; last entry is the final hidden state tensor.
    hidden_list = out.hidden_states
    if hidden_list:
        h = hidden_list[-1]             # (B, L, hc_mult, H) or (B, L, H)
        # Collapse hc_mult if present, take last position, remove batch dim.
        if h.ndim == 4:
            h = h.mean(axis=2)          # (B, L, H)
        hidden = h[0, -1]               # (H,)
    else:
        # Fallback: use logits as proxy (lm_head is linear so this is sufficient).
        hidden = logits

    return logits, hidden


def check(
    reference_model,
    submission_model,
    tokens: mx.array,
    epsilon: float = CORRECTNESS_EPSILON,
) -> CorrectnessResult:
    """Run CORRECTNESS_STEPS greedy decode steps on both models and compare.

    tokens: shape (1, prompt_length) — the seeded prompt.
    """
    num_layers = _get_num_layers(reference_model)

    ref_cache = _make_cache(reference_model)
    sub_cache = _make_cache(submission_model)

    # Prefill: run the full prompt through both models to warm caches.
    ref_out = reference_model(tokens, cache=ref_cache)
    sub_out = submission_model(tokens, cache=sub_cache)
    mx.eval(ref_out.logits, sub_out.logits)

    # Seed the decode from the last token of the prompt.
    ref_tok = mx.argmax(ref_out.logits[0, -1:], axis=-1, keepdims=True)
    sub_tok = mx.argmax(sub_out.logits[0, -1:], axis=-1, keepdims=True)
    mx.eval(ref_tok, sub_tok)

    max_abs = 0.0
    max_rel = 0.0
    first_failing: Optional[int] = None
    failing_diffs: List[float] = []

    for step in range(CORRECTNESS_STEPS):
        ref_logits, ref_hidden = _forward(reference_model, ref_tok[None], ref_cache)
        sub_logits, sub_hidden = _forward(submission_model, sub_tok[None], sub_cache)
        mx.eval(ref_logits, sub_logits, ref_hidden, sub_hidden)

        # Layer 1: greedy token match.
        ref_next = int(mx.argmax(ref_logits))
        sub_next = int(mx.argmax(sub_logits))

        # Layer 2: hidden state tolerance.
        h_diff = float(mx.max(mx.abs(ref_hidden - sub_hidden)))
        ref_abs = mx.abs(ref_hidden)
        rel = mx.where(ref_abs > 1e-6,
                       mx.abs(ref_hidden - sub_hidden) / ref_abs,
                       mx.zeros_like(ref_hidden))
        h_rel = float(mx.max(rel))

        max_abs = max(max_abs, h_diff)
        max_rel = max(max_rel, h_rel)

        # Layer 3: top-K logit set match.
        ref_topk = set(mx.argpartition(-ref_logits, kth=CORRECTNESS_TOP_K - 1)
                       [:CORRECTNESS_TOP_K].tolist())
        sub_topk = set(mx.argpartition(-sub_logits, kth=CORRECTNESS_TOP_K - 1)
                       [:CORRECTNESS_TOP_K].tolist())

        step_failed = (
            ref_next != sub_next          # Layer 1
            or h_diff > epsilon           # Layer 2
            or ref_topk != sub_topk       # Layer 3
        )
        if step_failed and first_failing is None:
            first_failing = step
            failing_diffs.append(h_diff)

        ref_tok = mx.array([[ref_next]])
        sub_tok = mx.array([[sub_next]])

    passed = first_failing is None
    return CorrectnessResult(
        passed=passed,
        num_layers=num_layers,
        first_failing_layer=first_failing,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        failing_layer_diffs=failing_diffs,
    )
