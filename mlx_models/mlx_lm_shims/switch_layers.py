"""
switch_layers.py — SSD-streaming replacement for mlx-lm's SwitchGLU.

Replaces the fully-resident QuantizedSwitchLinear/SwitchGLU stack with a
slot-bank that keeps only SLOT_BANK_SIZE expert records in Metal memory at
a time, loading the rest on-demand from SSD via pread() directly from the
original reference safetensors shards.

ExpertSlotBank.configure_from_safetensors() parses shard headers once at load
time to compute per-expert byte offsets, then issues 6 pread calls per expert
on each cache miss (one per proj×ttype: gate/up/down × weight/scales).

transform.py is always run but only copies the dense (non-expert) weights to
weights/. Expert weights live in the reference shards and are never repacked.
Participants may replace transform.py to repack or permute experts as they
choose — ExpertSlotBank will use any layout as long as the shard files and
safetensors headers are valid.

SLOT_BANK_SIZE is the primary knob. Raising it keeps more experts resident
(fewer disk reads) at the cost of wired Metal memory. The default (128) keeps
~1.7 GB wired; good hit rate for autoregressive decode.
"""
from __future__ import annotations

import json
import math
import os
import struct
from collections import OrderedDict
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Safetensors header parsing helpers (used by safetensors streaming mode)
# ---------------------------------------------------------------------------

# Map safetensors dtype strings to (numpy_dtype, bytes_per_element).
# Only the dtypes that appear in mxfp4 checkpoints are needed.
_ST_DTYPE = {
    "U8":   (np.dtype("uint8"),   1),
    "U16":  (np.dtype("uint16"),  2),
    "U32":  (np.dtype("uint32"),  4),
    "F16":  (np.dtype("float16"), 2),
    "F32":  (np.dtype("float32"), 4),
    "BF16": (np.dtype("uint16"),  2),  # stored as raw uint16 bytes; MLX handles BF16
    "I32":  (np.dtype("int32"),   4),
}


def _parse_safetensors_header(shard_path: str) -> dict:
    """Return {tensor_key: (abs_file_data_offset, total_bytes, st_dtype_str, shape)}
    for every tensor in the shard.

    Safetensors binary layout:
      [8 bytes: little-endian uint64 header_length]
      [header_length bytes: UTF-8 JSON]
      [tensor data at offsets given by header["data_offsets"]]
    data_offsets are relative to the first byte after the header JSON.
    """
    with open(shard_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = json.loads(f.read(header_len).decode("utf-8"))
    abs_data_start = 8 + header_len  # first byte of tensor data region
    result: dict = {}
    for key, meta in header_json.items():
        if key == "__metadata__":
            continue
        ds, de = meta["data_offsets"]
        result[key] = (
            abs_data_start + ds,  # absolute file offset where this tensor's bytes begin
            de - ds,              # total bytes for the whole tensor
            meta["dtype"],        # safetensors dtype string e.g. "U8"
            meta["shape"],        # list of ints
        )
    return result


# ---------------------------------------------------------------------------
# Tuneable constant — first thing participants see and adjust
# ---------------------------------------------------------------------------

SLOT_BANK_SIZE: int = 128
"""Number of expert weight records kept resident in Metal memory.

Each record holds gate_proj + up_proj + down_proj for one expert.
At mxfp4 4-bit, each record is roughly 13 MB for the default DS4-Flash dims.
128 slots ≈ 1.66 GB wired; good hit rate for autoregressive decode.

Raise for fewer SSD reads; lower if you observe memory pressure.
"""


# ---------------------------------------------------------------------------
# Slot bank
# ---------------------------------------------------------------------------

class ExpertSlotBank:
    """Fixed-capacity LRU cache of expert weight records.

    Reads expert weights directly from the original HF safetensors shards via
    pread().  configure_from_safetensors() parses shard headers once to build
    a map of byte offsets; each cache miss issues 6 pread calls (one per
    proj×ttype: gate/up/down × weight/scales).

    Weights are stored as lazy mx.arrays; Metal is allocated only when they
    are consumed by gather_qmm, so cached-but-idle records hold no Metal memory.
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
        # (layer_idx, proj_name, ttype) -> (shard_path, abs_file_offset, bpe, st_dtype_str, raw_shape)
        self._st_info: Optional[dict] = None

    def configure_from_safetensors(
        self, reference_dir: str, experts_dir: str
    ) -> None:
        """Configure the bank to pread directly from original safetensors shards.

        Parses the safetensors index.json and shard headers to build a map of
        byte offsets for every (layer, proj, ttype) stacked tensor.  Each
        expert's slice is located at:

            file_offset = tensor_data_start + expert_idx × bytes_per_expert

        because safetensors stores stacked [N, H, W] tensors in row-major
        order, making each expert a contiguous slab of bytes.

        A synthetic manifest.json is written to experts_dir so that any
        downstream code that reads it (e.g. harness validation) still works.

        Args:
            reference_dir: Directory containing the original safetensors shards
                and model.safetensors.index.json.
            experts_dir: Destination for the generated manifest.json.
        """
        self._experts_dir = experts_dir
        self._mode = "safetensors"

        # ── 1. Load weight_map: tensor_key → shard_filename ──────────────
        index_path = os.path.join(reference_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]

        # ── 2. Identify expert keys and group by shard ────────────────────
        # Expected format: model.layers.{L}.ffn.switch_mlp.{proj}.{weight|scales}
        # (stacked: shape [n_experts, ...])
        expert_by_shard: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            if self._is_st_expert_key(key):
                expert_by_shard.setdefault(shard, []).append(key)

        # ── 3. Parse shard headers once per unique shard ──────────────────
        key_to_info: dict[str, tuple] = {}   # key → (shard_path, abs_offset, total_bytes, dtype_str, shape)
        for shard_name, keys in expert_by_shard.items():
            shard_path = os.path.join(reference_dir, shard_name)
            header = _parse_safetensors_header(shard_path)
            for key in keys:
                if key in header:
                    abs_off, total_bytes, dtype_str, shape = header[key]
                    key_to_info[key] = (shard_path, abs_off, total_bytes, dtype_str, shape)

        # ── 4. Build st_info keyed by (layer_idx, proj, ttype) ───────────
        st_info: dict = {}
        n_experts: int = 0
        layers: set[int] = set()

        for key, shard in weight_map.items():
            if not self._is_st_expert_key(key):
                continue
            layer_idx, proj, ttype = self._parse_st_expert_key(key)
            if layer_idx is None:
                continue
            if key not in key_to_info:
                continue
            shard_path, abs_off, total_bytes, dtype_str, shape = key_to_info[key]
            n_exp = shape[0]
            n_experts = max(n_experts, n_exp)
            layers.add(layer_idx)
            bpe = total_bytes // n_exp          # bytes per expert slice
            raw_shape = list(shape[1:])         # per-expert shape in raw (uint8) dtype
            st_info[(layer_idx, proj, ttype)] = (
                shard_path, abs_off, bpe, dtype_str, raw_shape
            )

        if not st_info:
            raise RuntimeError(
                f"No stacked expert keys found in {index_path}. "
                "Expected keys matching model.layers.*.ffn.switch_mlp.*"
            )

        self._st_info = st_info

        # ── 5. Build synthetic manifest matching transform.py's schema ────
        projs = sorted({p for _, p, _ in st_info})
        manifest_projs: dict = {}
        running_offset = 0
        record_size = 0

        for proj in projs:
            manifest_projs[proj] = {}
            for ttype in ("weight", "scales", "biases"):
                info = st_info.get((0, proj, ttype))
                if info is None:
                    manifest_projs[proj][ttype] = None
                    continue
                _, _, bpe, dtype_str, raw_shape = info
                np_dtype_obj, _ = _ST_DTYPE.get(dtype_str, (np.dtype("uint8"), 1))
                if ttype == "weight" and dtype_str == "U8":
                    # U8-packed mxfp4: viewed as uint32, last dim shrinks by 4×
                    np_dtype = np.dtype("<u4")
                    h = raw_shape[0]
                    w_u8 = raw_shape[1] if len(raw_shape) > 1 else 1
                    post_shape = [h, w_u8 // 4]
                else:
                    np_dtype = np_dtype_obj
                    post_shape = raw_shape
                manifest_projs[proj][ttype] = {
                    "dtype":            np_dtype.str,
                    "shape":            post_shape,
                    "nbytes":           bpe,
                    "offset_in_record": running_offset,
                }
                running_offset += bpe
                record_size    += bpe

        n_layers = max(layers) + 1
        manifest = {
            "num_layers":  n_layers,
            "num_experts": n_experts,
            "record_size": record_size,
            "projections": manifest_projs,
            "quant":       {"group_size": 32, "bits": 4, "mode": "mxfp4"},
            "source":      "safetensors",
        }
        self._manifest = manifest

        # Write manifest.json so the harness can still locate quant metadata.
        os.makedirs(experts_dir, exist_ok=True)
        manifest_path = os.path.join(experts_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    @staticmethod
    def _is_st_expert_key(key: str) -> bool:
        """True if key is a stacked expert tensor in safetensors format."""
        return (
            ".ffn.switch_mlp." in key
            and ".shared_experts." not in key
            and (key.endswith(".weight") or key.endswith(".scales"))
            and any(f".{p}." in key for p in ("gate_proj", "up_proj", "down_proj"))
        )

    @staticmethod
    def _parse_st_expert_key(key: str) -> tuple:
        """Return (layer_idx, proj, ttype) from a stacked expert key, or (None, None, None)."""
        parts = key.split(".")
        try:
            li = parts.index("layers")
            layer_idx = int(parts[li + 1])
            si = parts.index("switch_mlp")
            proj  = parts[si + 1]
            ttype = parts[si + 2]
            return layer_idx, proj, ttype
        except (ValueError, IndexError):
            return None, None, None

    # ------------------------------------------------------------------
    # Public API (same for both modes)
    # ------------------------------------------------------------------

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
        return self._load_from_safetensors(layer_idx, expert_idx)

    def _load_from_safetensors(
        self, layer_idx: int, expert_idx: int
    ) -> dict[str, dict[str, mx.array]]:
        """Load one expert by preading each projection directly from shard files.

        Six pread calls per expert:
          gate_proj.weight, gate_proj.scales,
          up_proj.weight,   up_proj.scales,
          down_proj.weight, down_proj.scales
        Each pread reads bytes_per_expert bytes at:
          tensor_data_start + expert_idx × bytes_per_expert
        """
        result: dict[str, dict[str, mx.array]] = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            arrays: dict[str, mx.array] = {}
            for ttype in ("weight", "scales"):
                info = self._st_info.get((layer_idx, proj, ttype))
                if info is None:
                    continue
                shard_path, abs_off, bpe, dtype_str, raw_shape = info
                file_offset = abs_off + expert_idx * bpe
                fd = self._open_fd(shard_path)
                raw: bytes = os.pread(fd, bpe, file_offset)
                np_dtype, _ = _ST_DTYPE.get(dtype_str, (np.dtype("uint8"), 1))
                np_arr = np.frombuffer(raw, dtype=np_dtype).reshape(raw_shape)
                # U8-packed mxfp4 weights: reinterpret 4 uint8 bytes as uint32
                # to match what gather_qmm expects.
                if ttype == "weight" and dtype_str == "U8":
                    np_arr = np_arr.view(np.uint32)
                arrays[ttype] = mx.array(np_arr)
            result[proj] = arrays
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


def configure_safetensors_streaming(
    reference_dir: str,
    experts_dir: str,
    capacity: int = SLOT_BANK_SIZE,
) -> ExpertSlotBank:
    """Create and configure the global slot bank for safetensors direct-pread mode.

    Use this when transform.py has NOT been run (no layer_NN.bin files exist).
    The bank reads expert weights directly from the original HF safetensors shards
    using per-expert byte offsets derived from the safetensors headers.

    A synthetic manifest.json is written to experts_dir so downstream harness
    code still works.  Idempotent: a bank already configured for the same
    experts_dir is reused.

    Args:
        reference_dir: Directory containing the HF safetensors shards and
            model.safetensors.index.json.
        experts_dir: Destination for the generated manifest.json (created if
            it does not exist).
        capacity: Slot bank size (default: SLOT_BANK_SIZE).
    """
    global _SLOT_BANK
    if _SLOT_BANK is not None and _SLOT_BANK._experts_dir == experts_dir:
        return _SLOT_BANK
    _SLOT_BANK = ExpertSlotBank(capacity)
    _SLOT_BANK.configure_from_safetensors(reference_dir, experts_dir)
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

        if N > 1:
            # Prefill: force eval + clear cache to prevent ~3 GB per-layer
            # stacked-tensor accumulation (43 layers × ~3 GB = ~129 GB without
            # this).  mx.clear_cache() releases Metal buffers to the OS so the
            # next layer can allocate fresh ones without OOM.
            mx.eval(result)
            del gate_w, gate_s, up_w, up_s, down_w, down_s
            del x_gate, x_up, x_act, x_out, records
            mx.clear_cache()
        # During decode (N == 1) return the lazy tensor.  The computation
        # graph keeps stacked tensors alive (~78 MB × 43 layers = 3.4 GB max)
        # which fits comfortably in Metal.  The next layer's tolist() sync
        # will batch this layer's GPU ops with its own, reducing roundtrips.
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
