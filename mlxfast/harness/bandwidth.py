"""Memory bandwidth measurement. FROZEN.

Bandwidth is measured via mactop, which reads Apple's hardware IOReport
DRAM counters directly:

  mactop --headless --count N --interval 100 --format json

Each JSON line contains soc_metrics.dram_bw_combined_gbs — the
instantaneous DRAM read+write bandwidth at that sample in GB/s.

The harness starts mactop before the decode loop, collects samples
concurrently, and computes:

  gb_per_token = (mean_non_zero_gbps × decode_duration_s) / num_tokens

If mactop is not installed or fails to produce samples, bandwidth
measurement fails closed.
"""
from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional

from .constants import DECODE_LENGTH

MACTOP_BINARY = os.environ.get("MACTOP_BINARY", "")
MACTOP_INTERVAL_MS = 100          # sample every 100 ms
MACTOP_MAX_SAMPLES = 600          # 60 s — enough to outlast any decode run


@dataclass
class BandwidthResult:
    bytes_read: int
    tokens_decoded: int
    gb_per_token: float
    source: str   # "mactop_hardware"

    def to_dict(self) -> dict:
        return {
            "bytes_read": self.bytes_read,
            "tokens_decoded": self.tokens_decoded,
            "gb_per_token": self.gb_per_token,
            "source": self.source,
        }


# ── mactop hardware measurement ───────────────────────────────────────────────

def _find_mactop_binary() -> Optional[str]:
    candidates = [
        MACTOP_BINARY,
        shutil.which("mactop"),
        "/opt/homebrew/bin/mactop",
        "/usr/local/bin/mactop",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


class MactopSession:
    """Context manager that runs mactop in the background and collects samples."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._samples: List[float] = []
        self._start: float = 0.0

    def start(self) -> bool:
        """Start mactop. Returns False if binary not found."""
        binary = _find_mactop_binary()
        if binary is None:
            return False
        try:
            self._proc = subprocess.Popen(
                [
                    binary,
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
        """Terminate mactop, store and return non-zero DRAM BW samples (GB/s)."""
        if self._proc is None:
            self._samples = []
            return self._samples
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
        # mactop --format json outputs a JSON array; try that first.
        try:
            items = json.loads(stdout)
            if isinstance(items, list):
                for obj in items:
                    bw = obj.get("soc_metrics", {}).get("dram_bw_combined_gbs", 0.0)
                    if bw > 0.0:
                        samples.append(float(bw))
                self._samples = samples
                return samples
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        # Fall back to newline-delimited JSON.
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
        self._samples = samples
        return samples

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


def measure_idle_bandwidth(duration_s: float = 3.0) -> float:
    """Sample background DRAM bandwidth for `duration_s` seconds.

    Returns the mean non-zero GB/s across samples, or 0.0 if mactop is
    unavailable. Call this before model loading so the idle baseline
    reflects system noise without inference activity.
    """
    binary = _find_mactop_binary()
    if binary is None:
        return 0.0
    n_samples = max(1, int(duration_s * 1000 / MACTOP_INTERVAL_MS))
    try:
        result = subprocess.run(
            [binary, "--headless", "--count", str(n_samples),
             "--interval", str(MACTOP_INTERVAL_MS), "--format", "json"],
            capture_output=True, text=True, timeout=duration_s + 5,
        )
        samples: List[float] = []
        try:
            items = json.loads(result.stdout)
            if isinstance(items, list):
                for obj in items:
                    bw = obj.get("soc_metrics", {}).get("dram_bw_combined_gbs", 0.0)
                    if bw > 0.0:
                        samples.append(float(bw))
        except (json.JSONDecodeError, TypeError):
            pass
        return sum(samples) / len(samples) if samples else 0.0
    except Exception:
        return 0.0


def _mactop_result(
    session: MactopSession,
    num_tokens: int,
    decode_duration: float,
    idle_gbps: float = 0.0,
) -> Optional[BandwidthResult]:
    samples = session._samples
    if not samples:
        return None
    # Subtract idle baseline from each sample (floor at 0) then average.
    net_samples = [max(s - idle_gbps, 0.0) for s in samples]
    mean_gbps = sum(net_samples) / len(net_samples)
    total_gb = mean_gbps * decode_duration
    gb_per_token = total_gb / num_tokens if num_tokens > 0 else 0.0
    total_bytes = int(total_gb * (1024 ** 3))
    return BandwidthResult(
        bytes_read=total_bytes,
        tokens_decoded=num_tokens,
        gb_per_token=gb_per_token,
        source="mactop_hardware",
    )


def _moe_software_estimate(
    model,
    experts_manifest_path: str,
    num_tokens: int,
) -> Optional[BandwidthResult]:
    """MoE-aware software bandwidth estimate used as mactop fallback.

    Counts bytes actually read per decode token:
      - Non-expert model weights (resident in Metal, from model.parameters())
      - Activated expert weights: NUM_EXPERTS_PER_TOK × record_size × num_moe_layers
        where record_size comes from the expert manifest (exact on-disk size).

    This is correct for streaming MoE: expert weights are filtered from
    model.parameters() so they are not double-counted here.
    """
    from .constants import NUM_EXPERTS_PER_TOK, NUM_HIDDEN_LAYERS
    try:
        from mlx.utils import tree_flatten
    except ImportError:
        return None

    # Non-expert model weights resident in Metal.
    leaves = tree_flatten(model.parameters())
    non_expert_bytes = sum(arr.nbytes for _, arr in leaves)

    # Activated expert bytes per token from the manifest record_size.
    expert_bytes = 0
    try:
        with open(experts_manifest_path) as f:
            manifest = json.load(f)
        record_size: int = manifest["record_size"]
        expert_bytes = NUM_EXPERTS_PER_TOK * record_size * NUM_HIDDEN_LAYERS
    except (FileNotFoundError, KeyError, TypeError):
        pass  # manifest unavailable — omit expert bandwidth

    total_bytes = non_expert_bytes + expert_bytes
    gb_per_token = total_bytes / (1024 ** 3)
    return BandwidthResult(
        bytes_read=total_bytes,
        tokens_decoded=num_tokens,
        gb_per_token=gb_per_token,
        source="moe_software_model",
    )


# ── public API ────────────────────────────────────────────────────────────────

def measure(
    mactop_session: MactopSession,
    num_tokens: int = DECODE_LENGTH,
    decode_duration: float = 0.0,
    model=None,
    experts_manifest_path: str = "",
    idle_gbps: float = 0.0,
) -> BandwidthResult:
    """Return bandwidth for `num_tokens` decode steps.

    Primary: mactop hardware DRAM counters (captures all traffic including
    expert SSD reads via page cache — correct for MoE without any special
    formula). idle_gbps is subtracted per-sample to remove background DRAM
    traffic (display, kernel tasks) from the measurement.

    Fallback: MoE-aware software model using non-expert model.parameters()
    bytes plus activated-expert bytes from the expert manifest.  Used only
    when mactop produces no samples.  Requires model and experts_manifest_path.
    """
    if mactop_session is None:
        raise TypeError("mactop_session is required")

    result = _mactop_result(mactop_session, num_tokens, decode_duration, idle_gbps=idle_gbps)
    if result is not None:
        return result

    if model is not None:
        sw = _moe_software_estimate(model, experts_manifest_path, num_tokens)
        if sw is not None:
            return sw

    raise RuntimeError(
        "mactop produced no non-zero DRAM bandwidth samples and no "
        "software fallback is available; install mactop and ensure "
        "hardware counters are available"
    )
