"""
switch_layers.py — SSD-streaming replacement for mlx-lm's SwitchGLU.

Replaces the fully-resident QuantizedSwitchLinear/SwitchGLU stack with a
slot-bank that keeps only SLOT_BANK_SIZE expert records in Metal memory at
a time, loading the rest on-demand from per-layer binary files on SSD.

Design (see SPEC.md §3):
  - Per-layer binary files:  weights/experts/layer_NN.bin
  - Expert-major layout: record[j] = all projections for expert j, fixed size
  - Slot bank: fixed-capacity LRU of pre-loaded expert dicts keyed by (layer, expert)
  - Direct pread: os.pread into a bytes buffer, zero-copy numpy view, mx.array

Optimization targets left deliberately open:
  - No async I/O       — reads block the forward pass
  - No prefetching     — next layer's experts are not pre-loaded
  - No prefill seeding — routing pattern during prefill not used to warm decode bank
  - No cross-layer sharing — each (layer, expert) is a separate slot

SLOT_BANK_SIZE is the primary knob. Raising it keeps more experts resident
(fewer disk reads) at the cost of wired Metal memory. The default (32) keeps
~200–400 MB wired and leaves ample room for the OS file cache to stay warm.
See SPEC.md §3 "Page-cache cliff avoidance" before setting this above ~128.
"""
from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Tuneable constant — first thing participants see and adjust
# ---------------------------------------------------------------------------

SLOT_BANK_SIZE: int = 128
"""Number of expert weight records kept resident in Metal memory.

Each record holds gate_proj + up_proj + down_proj for one expert.
At mxfp4 4-bit, each record is roughly 13 MB for the default DS4-Flash dims.
128 slots ≈ 1.66 GB wired; 50% cache-hit rate with 256 experts halves SSD
reads vs the default 32-slot bank.  Safe on 24 GB machines.

Raise for fewer SSD reads; lower if you observe memory pressure.
"""


# ---------------------------------------------------------------------------
# Slot bank
# ---------------------------------------------------------------------------

class ExpertSlotBank:
    """Fixed-capacity LRU cache of expert weight records in Metal memory.

    Records are loaded from per-layer binary files via os.pread on cache miss.
    File descriptors are kept open for the lifetime of the bank.

    Args:
        capacity: Maximum number of (layer, expert) records to keep resident.
    """

    def __init__(self, capacity: int = SLOT_BANK_SIZE) -> None:
        self.capacity = capacity
        self._lru: OrderedDict[tuple[int, int], dict[str, dict[str, mx.array]]] = (
            OrderedDict()
        )
        self._fds: dict[str, int] = {}
        self._experts_dir: Optional[str] = None
        self._manifest: Optional[dict] = None

    def configure(self, experts_dir: str) -> None:
        """Point the bank at the experts directory and load the manifest."""
        self._experts_dir = experts_dir
        manifest_path = os.path.join(experts_dir, "manifest.json")
        with open(manifest_path) as f:
            self._manifest = json.load(f)

    def get(self, layer_idx: int, expert_idx: int) -> dict[str, dict[str, mx.array]]:
        """Return weight dict for (layer_idx, expert_idx).

        Returns:
            {"gate_proj": {"weight": mx.array, "scales": mx.array, ...},
             "up_proj":   {...},
             "down_proj": {...}}

        Loads from disk on cache miss; evicts LRU entry if at capacity.
        """
        key = (layer_idx, expert_idx)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]

        record = self._load(layer_idx, expert_idx)
        if len(self._lru) >= self.capacity:
            self._lru.popitem(last=False)
        self._lru[key] = record
        return record

    def _open_fd(self, path: str) -> int:
        if path not in self._fds:
            self._fds[path] = os.open(path, os.O_RDONLY)
        return self._fds[path]

    def _load(self, layer_idx: int, expert_idx: int) -> dict[str, dict[str, mx.array]]:
        """Load one expert record from disk with a single pread."""
        manifest = self._manifest
        record_size: int = manifest["record_size"]
        offset: int = expert_idx * record_size

        bin_path = os.path.join(
            self._experts_dir, f"layer_{layer_idx:02d}.bin"
        )
        fd = self._open_fd(bin_path)
        data: bytes = os.pread(fd, record_size, offset)

        result: dict[str, dict[str, mx.array]] = {}
        for proj_name, proj_info in manifest["projections"].items():
            arrays: dict[str, mx.array] = {}
            for tensor_name in ("weight", "scales", "biases"):
                t = proj_info.get(tensor_name)
                if t is None:
                    continue
                raw = data[t["offset_in_record"] : t["offset_in_record"] + t["nbytes"]]
                np_arr = np.frombuffer(raw, dtype=t["dtype"]).reshape(t["shape"])
                arrays[tensor_name] = mx.array(np_arr)
            result[proj_name] = arrays
        return result

    def close(self) -> None:
        """Close all open file descriptors."""
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    @property
    def stats(self) -> dict:
        return {
            "capacity": self.capacity,
            "resident": len(self._lru),
            "open_files": len(self._fds),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_SLOT_BANK: Optional[ExpertSlotBank] = None


def configure_streaming(experts_dir: str, capacity: int = SLOT_BANK_SIZE) -> ExpertSlotBank:
    """Create and configure the global slot bank. Call once after model load.

    Args:
        experts_dir: Path to the directory containing manifest.json and layer_NN.bin.
        capacity: Slot bank size (default: SLOT_BANK_SIZE).
    """
    global _SLOT_BANK
    _SLOT_BANK = ExpertSlotBank(capacity)
    _SLOT_BANK.configure(experts_dir)
    return _SLOT_BANK


def get_slot_bank() -> ExpertSlotBank:
    if _SLOT_BANK is None:
        raise RuntimeError(
            "Expert slot bank not configured. "
            "configure_streaming(experts_dir) must be called before inference."
        )
    return _SLOT_BANK


# ---------------------------------------------------------------------------
# Streaming SwitchGLU
# ---------------------------------------------------------------------------

class StreamingSwitchGLU(nn.Module):
    """SSD-streaming drop-in replacement for SwitchGLU.

    Instead of keeping all num_experts weight matrices in Metal memory, this
    module loads only the activated experts from the slot bank on each forward
    pass, stacks them into a small dense tensor, and calls mx.gather_qmm on
    that subset — reusing the same optimised kernel as the original.

    Args:
        input_dims:   Hidden dimension of incoming tokens.
        hidden_dims:  Expert intermediate dimension.
        num_experts:  Total number of routed experts (256 for DS4-Flash).
        activation:   Gate activation (LimitedSwiGLU from language.py).
        group_size:   Quantisation group size (matches transform.py).
        bits:         Quantisation bit width (matches transform.py).
        mode:         Quantisation mode (matches transform.py).
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation: Any = None,
        bias: bool = False,
        group_size: int = 32,
        bits: int = 4,
        mode: str = "mxfp4",
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_experts = num_experts
        self.activation = activation
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Set by Model._configure_streaming after weights are loaded.
        self._layer_idx: Optional[int] = None

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        """Forward pass with on-demand expert loading.

        Args:
            x:       (*batch, hidden)  — token hidden states.
            indices: (*batch, K)       — routing indices in [0, num_experts).

        Returns:
            (*batch, K, hidden)  — weighted expert outputs (weights applied
                                   by the caller, DeepseekV4MoE).
        """
        if self._layer_idx is None:
            raise RuntimeError(
                "StreamingSwitchGLU._layer_idx not set. "
                "Call Model._configure_streaming(weights_dir) after loading."
            )

        bank = get_slot_bank()
        batch = indices.shape[:-1]
        K = indices.shape[-1]
        N = math.prod(batch)

        # Flatten batch dims for uniform processing.
        x_flat = x.reshape(N, x.shape[-1])         # (N, hidden)
        idx_flat = indices.reshape(N, K)            # (N, K)

        # Sort tokens by expert index — mirrors SwitchGLU's _gather_sort so
        # gather_qmm sees contiguous expert accesses for better Metal performance.
        flat = idx_flat.flatten()                   # (N*K,)
        order = mx.argsort(flat)
        inv_order = mx.argsort(order)
        sorted_idx = flat[order]                    # (N*K,) ascending expert ids

        # Each position in sorted_idx came from token (order[i] // K).
        x_sorted = x_flat[order // K]              # (N*K, hidden)

        # Unique experts activated this forward pass (at most N*K, usually ≤6 per token).
        unique: list[int] = sorted(set(sorted_idx.tolist()))

        # Load records from slot bank and stack into small dense tensors.
        # This is the only disk I/O in the forward pass.
        records = [bank.get(self._layer_idx, e) for e in unique]

        gate_w = mx.stack([r["gate_proj"]["weight"] for r in records])
        gate_s = mx.stack([r["gate_proj"]["scales"] for r in records])
        up_w   = mx.stack([r["up_proj"]["weight"]   for r in records])
        up_s   = mx.stack([r["up_proj"]["scales"]   for r in records])
        down_w = mx.stack([r["down_proj"]["weight"] for r in records])
        down_s = mx.stack([r["down_proj"]["scales"] for r in records])

        # Force Metal allocation of the stacked tensors before gather_qmm.
        mx.eval(gate_w, gate_s, up_w, up_s, down_w, down_s)

        # Remap sorted expert indices → dense [0, len(unique)).
        remap = {e: i for i, e in enumerate(unique)}
        dense_idx = mx.array(
            [remap[int(i)] for i in sorted_idx.tolist()], dtype=mx.int32
        )   # (N*K,)

        # gather_qmm expects x of shape (batch, 1, in_dim).
        x_qmm = x_sorted[:, None, :]              # (N*K, 1, hidden)

        x_gate = mx.gather_qmm(
            x_qmm, gate_w, gate_s, None,
            rhs_indices=dense_idx, transpose=True,
            group_size=self.group_size, bits=self.bits, mode=self.mode,
        )   # (N*K, 1, hidden_dims)

        x_up = mx.gather_qmm(
            x_qmm, up_w, up_s, None,
            rhs_indices=dense_idx, transpose=True,
            group_size=self.group_size, bits=self.bits, mode=self.mode,
        )   # (N*K, 1, hidden_dims)

        # Activation from DeepseekV4MoE (LimitedSwiGLU).
        x_act = self.activation(
            x_up.squeeze(-2), x_gate.squeeze(-2)
        )[:, None, :]                              # (N*K, 1, hidden_dims)

        x_out = mx.gather_qmm(
            x_act, down_w, down_s, None,
            rhs_indices=dense_idx, transpose=True,
            group_size=self.group_size, bits=self.bits, mode=self.mode,
        )   # (N*K, 1, hidden)

        # Squeeze expert dim, unsort to original token order.
        x_out = x_out.squeeze(-2)                 # (N*K, hidden)
        x_out = x_out[inv_order]                  # restore original order

        # Reshape to (*batch, K, hidden).
        result = x_out.reshape(*batch, K, x.shape[-1])

        # Force evaluation and release Metal buffers immediately.
        # Without this, the stacked expert tensors (gate_w, up_w, down_w ...)
        # freed from each layer stay in MLX's wired buffer cache.  With 256
        # unique experts per layer during a 512-token prefill, 43 layers ×
        # ~3 GB of stacked tensors × 2 (individual records + stacks) would
        # accumulate ~250 GB of wired Metal memory before the OS could reclaim
        # it — causing OOM on even a 64 GB machine.
        mx.eval(result)
        # Explicitly drop the large local arrays before clearing so their
        # Metal buffers are in the cache when we flush it.
        del gate_w, gate_s, up_w, up_s, down_w, down_s
        del x_gate, x_up, x_act, x_out, records
        mx.clear_cache()
        return result


# ---------------------------------------------------------------------------
# Keep original classes available for non-streaming use / harness reference
# ---------------------------------------------------------------------------

# Re-export the originals so anything that imports from this shim still works.
from mlx_lm.models.switch_layers import (  # noqa: E402, F401
    QuantizedSwitchLinear,
    SwitchLinear,
    SwitchMLP,
    _gather_sort,
    _scatter_unsort,
)

# SwitchGLU points to our streaming version; the original is available as
# _OriginalSwitchGLU if needed for debugging.
from mlx_lm.models.switch_layers import SwitchGLU as _OriginalSwitchGLU  # noqa: F401
SwitchGLU = StreamingSwitchGLU
