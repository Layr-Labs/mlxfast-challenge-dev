"""Memory bandwidth measurement. FROZEN.

Bandwidth is measured via mactop, which reads Apple's hardware IOReport
DRAM counters directly:

  mactop --headless --count N --interval 100 --format json

Each JSON line contains soc_metrics.dram_bw_combined_gbs — the
instantaneous DRAM read+write bandwidth at that sample in GB/s.

The harness starts mactop before the decode loop, collects samples
concurrently, and computes:

  gb_per_token = (mean_non_zero_gbps × decode_duration_s) / num_tokens

If mactop is not installed or fails, the harness falls back to an
MoE-aware software model with DeepSeek V4 Flash defaults.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional

from .constants import DECODE_LENGTH, N_ROUTED_EXPERTS, NUM_EXPERTS_PER_TOK, NUM_HIDDEN_LAYERS

MACTOP_BINARY = "/opt/homebrew/bin/mactop"
MACTOP_INTERVAL_MS = 100          # sample every 100 ms
MACTOP_MAX_SAMPLES = 600          # 60 s — enough to outlast any decode run


@dataclass
class BandwidthResult:
    bytes_read: int
    tokens_decoded: int
    gb_per_token: float
    source: str   # "mactop_hardware" | "moe_software_model" | "unavailable"

    def to_dict(self) -> dict:
        return {
            "bytes_read": self.bytes_read,
            "tokens_decoded": self.tokens_decoded,
            "gb_per_token": self.gb_per_token,
            "source": self.source,
        }


# ── mactop hardware measurement ───────────────────────────────────────────────

class MactopSession:
    """Context manager that runs mactop in the background and collects samples."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._samples: List[float] = []
        self._start: float = 0.0

    def start(self) -> bool:
        """Start mactop. Returns False if binary not found."""
        if not os.path.exists(MACTOP_BINARY):
            return False
        try:
            self._proc = subprocess.Popen(
                [
                    MACTOP_BINARY,
                    "--headless",
                    "--count", str(MACTOP_MAX_SAMPLES),
                    "--interval", str(MACTOP_INTERVAL_MS),
                    "--format", "json",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._start = time.perf_counter()
            return True
        except OSError:
            return False

    def stop(self) -> List[float]:
        """Terminate mactop and return non-zero DRAM BW samples (GB/s)."""
        if self._proc is None:
            return []
        try:
            os.kill(self._proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, _ = self._proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            stdout, _ = self._proc.communicate()

        samples: List[float] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                bw = obj.get("soc_metrics", {}).get("dram_bw_combined_gbs", 0.0)
                if bw > 0.0:
                    samples.append(float(bw))
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue
        return samples

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self._samples = self.stop()


def _mactop_result(session: MactopSession, num_tokens: int, decode_duration: float) -> Optional[BandwidthResult]:
    samples = session._samples
    if not samples:
        return None
    mean_gbps = sum(samples) / len(samples)
    total_gb = mean_gbps * decode_duration
    gb_per_token = total_gb / num_tokens if num_tokens > 0 else 0.0
    total_bytes = int(total_gb * (1024 ** 3))
    return BandwidthResult(
        bytes_read=total_bytes,
        tokens_decoded=num_tokens,
        gb_per_token=gb_per_token,
        source="mactop_hardware",
    )


# ── software model fallback ───────────────────────────────────────────────────

def _software_model(model, num_tokens: int, prompt_length: int) -> BandwidthResult:
    """MoE-aware software bandwidth estimate with DS4 defaults."""
    from mlx.utils import tree_flatten

    leaves = tree_flatten(model.parameters())
    shared_bytes = 0
    expert_bytes = 0

    for name, arr in leaves:
        path = "/".join(str(p) for p in name) if isinstance(name, tuple) else str(name)
        path_lower = path.lower()
        is_expert = (
            "switch_mlp" in path_lower
            and any(x in path_lower for x in ("gate_proj", "up_proj", "down_proj"))
        )
        if is_expert:
            expert_bytes += arr.nbytes
        else:
            shared_bytes += arr.nbytes

    # DS4 Flash defaults
    n_experts = N_ROUTED_EXPERTS
    experts_per_tok = NUM_EXPERTS_PER_TOK
    n_layers = NUM_HIDDEN_LAYERS

    try:
        cfg = getattr(model, "config", None) or getattr(model, "args", None)
        if cfg is not None:
            n_experts = getattr(cfg, "n_routed_experts", n_experts)
            experts_per_tok = getattr(cfg, "num_experts_per_tok", experts_per_tok)
            n_layers = getattr(cfg, "num_hidden_layers", n_layers)
    except Exception:
        pass

    expert_frac = experts_per_tok / n_experts if n_experts > 0 else 1.0
    param_bytes_per_token = shared_bytes + expert_bytes * expert_frac
    total_bytes = int(param_bytes_per_token * num_tokens)
    gb_per_token = (total_bytes / num_tokens) / (1024 ** 3) if num_tokens > 0 else 0.0

    return BandwidthResult(
        bytes_read=total_bytes,
        tokens_decoded=num_tokens,
        gb_per_token=gb_per_token,
        source="moe_software_model",
    )


# ── public API ────────────────────────────────────────────────────────────────

def measure(
    model,
    prompt: "mx.array",
    num_tokens: int = DECODE_LENGTH,
    mactop_session: Optional[MactopSession] = None,
    decode_duration: float = 0.0,
) -> BandwidthResult:
    """Return bandwidth estimate for `num_tokens` decode steps.

    If a MactopSession is provided (started before the decode loop and
    stopped after), use hardware DRAM counters.  Otherwise fall back to
    the MoE software model.
    """
    if mactop_session is not None and mactop_session._samples:
        result = _mactop_result(mactop_session, num_tokens, decode_duration)
        if result is not None:
            return result

    # Software model fallback.
    prompt_length = (
        prompt.shape[1] if (hasattr(prompt, "shape") and prompt.ndim > 1)
        else (prompt.shape[0] if hasattr(prompt, "shape") else len(prompt))
    )
    return _software_model(model, num_tokens, prompt_length)
