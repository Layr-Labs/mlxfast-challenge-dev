"""Main harness entry point. FROZEN.

This is the function the CLI's `mlxfast run` calls. It:

  1. Verifies the modifiable surface exists and is loadable.
  2. Loads the submission model (using the modifiable surface).
  3. Generates a runtime-random prompt and seeds the decode.
  4. Measures peak RAM, bandwidth, latency.
  5. Computes the score.
  6. Returns a RunReport that the CLI appends to results.tsv.

Correctness checking is disabled for now — this harness measures
performance only (see the `correctness` module, currently unused).

The harness runs in a subprocess (started by _harness_runner.py)
so that peak RAM is measured via the subprocess's resident set size
rather than the parent CLI's.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from . import bandwidth, constants, score


@dataclass
class RunReport:
    timestamp: str
    git_commit: str
    note: str
    peak_ram_gb: float
    bandwidth_gb_per_token: float
    decode_seconds_per_token: float
    prefill_seconds_per_token: float
    score: float
    passed_correctness: bool
    num_layers: int
    first_failing_layer: Optional[int]
    max_abs_diff: float
    harness_hash: str
    mlx_version: str
    mlx_lm_version: str
    error: Optional[str] = None
    bandwidth_source: str = ""

    def to_tsv_row(self) -> str:
        return "\t".join(
            [
                self.timestamp,
                self.git_commit,
                self.note,
                f"{self.peak_ram_gb:.4f}",
                f"{self.bandwidth_gb_per_token:.6f}",
                f"{self.decode_seconds_per_token:.6f}",
                f"{self.prefill_seconds_per_token:.6f}",
                f"{self.score:.6f}" if self.score != float("inf") else "inf",
                "1" if self.passed_correctness else "0",
                str(self.num_layers),
                str(self.first_failing_layer) if self.first_failing_layer is not None else "-",
                f"{self.max_abs_diff:.6e}",
                self.bandwidth_source,
                self.harness_hash[:12],
            ]
        )

    @staticmethod
    def tsv_header() -> str:
        return "\t".join(
            [
                "timestamp",
                "commit",
                "note",
                "peak_ram_gb",
                "bandwidth_gb_per_tok",
                "decode_sec_per_tok",
                "prefill_sec_per_tok",
                "score",
                "passed",
                "num_layers",
                "first_failing_layer",
                "max_abs_diff",
                "bandwidth_source",
                "harness_hash",
            ]
        )


def _get_git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def _versions() -> tuple[str, str]:
    try:
        import mlx
        mlx_v = mlx.__version__
    except Exception:
        mlx_v = "unknown"
    try:
        import mlx_lm
        mlx_lm_v = mlx_lm.__version__
    except Exception:
        mlx_lm_v = "unknown"
    return mlx_v, mlx_lm_v


def _seed_prompt(vocab_size: int, length: int, seed: int) -> mx.array:
    """Generate a random prompt of `length` tokens, seeded by `seed`.

    Returns shape (1, length) so embed_tokens gives (1, length, hidden).
    """
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab_size, size=(1, length)).astype(np.int32)
    return mx.array(tokens)


def _peak_gpu_memory_gb() -> float:
    """Return peak GPU memory usage in GB.

    Uses mx.get_peak_memory() which tracks GPU memory allocations
    through MLX's caching allocator. This is the standard approach
    used by mlx_lm.stream_generate() and mlx_lm.benchmark.py.

    The caller must call mx.reset_peak_memory() before the measured
    section and mx.get_peak_memory() after eval completes.
    """
    return mx.get_peak_memory() / (1024**3)


def _force_sanitize_load(load_fn):
    """Call load_fn() with mlx_vlm forced to run sanitize.

    mlx_vlm skips sanitize when checkpoint metadata contains 'format: mlx'.
    The DS4 Flash checkpoint has this metadata but still needs sanitize to
    remap 'model.*' keys to 'language_model.model.*'.  We:

    1. Strip 'format: mlx' from safetensors metadata → is_mlx_format=False →
       mlx_vlm calls sanitize which remaps keys to 'language_model.model.*'.

    2. Remap per-layer quantization config keys to match the new parameter
       paths.  The checkpoint's config.json contains per-layer quant overrides
       like 'model.layers.0.ffn.switch_mlp.gate_proj' → mxfp4, but after
       sanitize the model parameter paths are
       'language_model.model.layers.0.ffn.switch_mlp.gate_proj'.  Without the
       remap, mlx_vlm's class_predicate falls back to the default affine mode,
       producing affine-biases that the mxfp4 checkpoint doesn't contain.
    """
    import mlx_vlm.utils as _vu
    import safetensors as _st

    _orig_safe_open = _st.safe_open
    _orig_load_config = _vu.load_config

    class _SafeOpenNoFormatMeta:
        def __init__(self, path, *args, **kwargs):
            self._f = _orig_safe_open(path, *args, **kwargs)

        def metadata(self):
            m = self._f.metadata()
            if not m:
                return m
            return {k: v for k, v in m.items() if not (k == "format" and v == "mlx")}

        def keys(self):
            return self._f.keys()

        def get_tensor(self, key):
            return self._f.get_tensor(key)

        def __enter__(self):
            self._f.__enter__()
            return self

        def __exit__(self, *args):
            return self._f.__exit__(*args)

    def _patched_load_config(model_path, **kwargs):
        config = _orig_load_config(model_path, **kwargs)
        quant = config.get("quantization")
        if isinstance(quant, dict):
            remapped = {}
            for k, v in quant.items():
                if isinstance(v, dict) and not k.startswith("language_model."):
                    remapped[f"language_model.{k}"] = v
                else:
                    remapped[k] = v
            config["quantization"] = remapped
        return config

    _vu.safetensors.safe_open = _SafeOpenNoFormatMeta
    _vu.load_config = _patched_load_config
    try:
        return load_fn()
    finally:
        _vu.safetensors.safe_open = _orig_safe_open
        _vu.load_config = _orig_load_config


def _load_participant_model(weights_path: str):
    """Load a model via the participant's mlx_models.deepseek_v4 module.

    The DS4 Flash checkpoint uses stacked expert tensors (~130 GB total)
    which cannot be loaded into GPU memory.  The participant's streaming
    SwitchGLU loads experts on-demand from binary files, so only the
    ~4 GB of non-expert weights reside in Metal.

    Force sanitize is applied so that checkpoint keys 'model.*' are
    remapped to 'language_model.model.*' by the model's sanitize() method.
    The per-layer mxfp4 quantisation config entries are also remapped to
    match the 'language_model.*' parameter namespace.
    """
    import sys
    import importlib
    import mlx_vlm

    participant_mod = importlib.import_module("mlx_models.deepseek_v4.deepseek_v4")

    _orig = sys.modules.get("mlx_vlm.models.deepseek_v4")
    sys.modules["mlx_vlm.models.deepseek_v4"] = participant_mod
    try:
        model, tokenizer = _force_sanitize_load(
            lambda: mlx_vlm.load(weights_path, trust_remote_code=False)
        )
    finally:
        if _orig is None:
            sys.modules.pop("mlx_vlm.models.deepseek_v4", None)
        else:
            sys.modules["mlx_vlm.models.deepseek_v4"] = _orig

    return model, tokenizer


def _load_models(weights_path: Path):
    """Load the submission model.

    The reference model is only used during transform (to extract expert
    weights into the streaming layout).  During inference measurement only
    the submission model is needed; the local correctness gate runs it as
    both ref and sub (self-consistency check).  The competition server
    performs the true comparison against the ground-truth reference.
    """
    sub_model, sub_tokenizer = _load_participant_model(str(weights_path))
    sub_model.eval()
    mx.eval(sub_model.parameters())

    return sub_model, sub_tokenizer, sub_model


def _load_config_dict(weights_path: Path) -> dict:
    config_path = weights_path / "config.json"
    with open(config_path) as f:
        return json.load(f)


def _measure_latency_and_memory(model, prompt: mx.array, num_tokens: int) -> tuple[float, float, "bandwidth.MactopSession"]:
    """Decode `num_tokens` tokens, return (seconds_per_token, peak_ram_gb).

    Resets the MLX peak memory counter before the decode loop, then
    reads peak memory after eval. This ensures peak RAM captures only
    the decode phase, not prefill or warmup.
    """
    # Warmup
    cache = model.make_cache() if hasattr(model, "make_cache") else None
    _ = model(prompt, cache=cache)
    mx.eval(model.parameters())

    # Reset peak memory counter before measuring decode.
    mx.reset_peak_memory()

    # Start mactop hardware bandwidth measurement.
    mactop = bandwidth.MactopSession()
    if not mactop.start():
        raise RuntimeError(
            "mactop is required for bandwidth measurement but was not found; "
            "run ./setup.sh or install it with Homebrew"
        )

    # Decode loop
    cache = model.make_cache() if hasattr(model, "make_cache") else None
    t0 = time.perf_counter()
    next_tok = mx.argmax(model(prompt, cache=cache).logits[0, -1:], axis=-1, keepdims=True)
    mx.eval(next_tok)
    for _ in range(num_tokens - 1):
        out = model(next_tok, cache=cache)
        next_tok = mx.argmax(out.logits[0, -1:], axis=-1, keepdims=True)
    mx.eval(next_tok)
    elapsed = time.perf_counter() - t0

    mactop._samples = mactop.stop()

    seconds_per_token = elapsed / num_tokens
    peak_ram_gb = _peak_gpu_memory_gb()

    return seconds_per_token, peak_ram_gb, mactop


def _measure_prefill_latency(model, prompt: mx.array) -> float:
    """Measure prefill seconds-per-token for a full prompt forward pass.

    Prefill processes the entire prompt in parallel (one forward call
    with the full sequence), which is the dominant cost for long-context
    use cases and reflects the per-token cost of any schema that
    requires setup work proportional to sequence length.

    A transform that shifts computation from the decode phase to the
    prefill phase — e.g. computing sparse representations per prompt —
    will show up here rather than in decode_seconds_per_token.

    Returns:
      Wall-clock seconds per prompt token, averaged over two timed
      runs (after one warmup). A fresh KV cache is created for each
      run so results are comparable across submissions with different
      cache layouts.
    """
    prompt_len = prompt.shape[-1]

    def _prefill_once():
        cache = model.make_cache() if hasattr(model, "make_cache") else None
        out = model(prompt, cache=cache)
        mx.eval(out.logits)

    # Warmup — load weights, fill caches, JIT compile.
    _prefill_once()

    # Two timed runs, take the mean.
    times = []
    for _ in range(2):
        t0 = time.perf_counter()
        _prefill_once()
        times.append(time.perf_counter() - t0)

    return (sum(times) / len(times)) / prompt_len


def run(weights_path: Path, note: str, secret: str = "") -> RunReport:
    """The main harness entry point.

    `secret` is the server-side secret used to seed the prompt
    generation. If empty (local dev), a deterministic seed derived
    from the git commit is used. The server passes a real secret
    so the prompt is unpredictable to the participant.
    """
    import datetime

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    commit = _get_git_commit()
    mlx_v, mlx_lm_v = _versions()
    harness_hash = constants.compute_harness_hash()

    # Seed the prompt.
    seed = int(hashlib_sha256(f"{secret}|{commit}")) % (2**31)

    try:
        sub_model, _, _ = _load_models(weights_path)

        # Correctness checking is disabled for now: this harness measures
        # performance only. num_layers is still reported as run metadata.
        cfg = _load_config_dict(weights_path)
        num_layers = int(
            cfg.get("num_hidden_layers")
            or cfg.get("text_config", {}).get("num_hidden_layers", 0)
        )

        # Build the decode prompt of typical length.
        vocab_size = constants.VOCAB_SIZE  # DeepSeek V4
        prompt = _seed_prompt(
            vocab_size,
            constants.PROMPT_SEED_PREFIX_LENGTH,
            seed,
        )

        # Build the prefill prompt (longer than the decode prompt).
        prefill_prompt = _seed_prompt(
            vocab_size,
            constants.PREFILL_PROMPT_LENGTH,
            seed ^ 0xDEADBEEF,  # distinct seed from the decode prompt
        )

        # Measure decode latency, peak RAM, and bandwidth together.
        decode_spt, peak_gb, mactop_session = _measure_latency_and_memory(
            sub_model, prompt, constants.DECODE_LENGTH
        )
        peak_bytes = int(peak_gb * (1024**3))
        bw = bandwidth.measure(
            mactop_session,
            constants.DECODE_LENGTH,
            decode_duration=decode_spt * constants.DECODE_LENGTH,
            model=sub_model,
            experts_manifest_path=str(weights_path / "experts" / "manifest.json"),
        )

        # Measure prefill latency.
        prefill_spt = _measure_prefill_latency(sub_model, prefill_prompt)

        sr = score.compute(
            peak_ram_bytes=peak_bytes,
            bandwidth_gb_per_token=bw.gb_per_token,
            decode_seconds_per_token=decode_spt,
            prefill_seconds_per_token=prefill_spt,
            passed_correctness=True,
            note=note,
        )

        return RunReport(
            timestamp=timestamp,
            git_commit=commit,
            note=note,
            peak_ram_gb=sr.peak_ram_gb,
            bandwidth_gb_per_token=sr.bandwidth_gb_per_token,
            decode_seconds_per_token=sr.decode_seconds_per_token,
            prefill_seconds_per_token=sr.prefill_seconds_per_token,
            score=sr.score,
            passed_correctness=True,
            num_layers=num_layers,
            first_failing_layer=None,
            max_abs_diff=0.0,
            harness_hash=harness_hash,
            mlx_version=mlx_v,
            mlx_lm_version=mlx_lm_v,
            bandwidth_source=bw.source,
        )
    except Exception as e:
        return RunReport(
            timestamp=timestamp,
            git_commit=commit,
            note=note,
            peak_ram_gb=0.0,
            bandwidth_gb_per_token=0.0,
            decode_seconds_per_token=0.0,
            prefill_seconds_per_token=0.0,
            score=float("inf"),
            passed_correctness=False,
            num_layers=0,
            first_failing_layer=None,
            max_abs_diff=0.0,
            harness_hash=harness_hash,
            mlx_version=mlx_v,
            mlx_lm_version=mlx_lm_v,
            error=str(e),
        )


def hashlib_sha256(s: str) -> int:
    """SHA-256 of a string, returned as an int (truncated to 31 bits)."""
    import hashlib

    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big")


def main():
    parser = argparse.ArgumentParser(description="mlxfast harness")
    parser.add_argument("--weights", type=Path, default=constants.PARTICIPANT_WEIGHTS_DIR)
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--secret", type=str, default="")
    parser.add_argument(
        "--output-tsv", type=Path, default=constants.RESULTS_FILE
    )
    args = parser.parse_args()

    report = run(args.weights, args.note, args.secret)

    # Append to results.tsv.
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    if not args.output_tsv.exists():
        args.output_tsv.write_text(RunReport.tsv_header() + "\n")
    with open(args.output_tsv, "a") as f:
        f.write(report.to_tsv_row() + "\n")

    # Print a one-line summary.
    status = "PASS" if report.passed_correctness else "FAIL"
    print(
        f"[{status}] score={report.score:.4f} "
        f"ram={report.peak_ram_gb:.2f}GB "
        f"bw={report.bandwidth_gb_per_token:.4f}GB/tok "
        f"decode={report.decode_seconds_per_token:.4f}s/tok "
        f"prefill={report.prefill_seconds_per_token:.4f}s/tok"
    )
    if not report.passed_correctness:
        if report.first_failing_layer is not None:
            print(
                f"  first failing layer: {report.first_failing_layer} "
                f"(max_abs_diff={report.max_abs_diff:.3e})"
            )
        if report.error:
            print(f"  error: {report.error}")
    sys.exit(0 if report.passed_correctness else 1)


if __name__ == "__main__":
    main()
