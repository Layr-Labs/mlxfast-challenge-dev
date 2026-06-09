"""Main harness entry point. FROZEN.

This is the function the CLI's `quantizationfail run` calls. It:

  1. Verifies the modifiable surface exists and is loadable.
  2. Loads the reference model (no modifiable surface involvement).
  3. Loads the submission model (using the modifiable surface).
  4. Generates a runtime-random prompt and seeds the decode.
  5. Runs the correctness gate.
  6. Measures peak RAM, bandwidth, latency.
  7. Computes the score.
  8. Returns a RunReport that the CLI appends to results.tsv.

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

from . import bandwidth, constants, correctness, score


@dataclass
class RunReport:
    timestamp: str
    git_commit: str
    note: str
    peak_ram_gb: float
    bandwidth_gb_per_token: float
    seconds_per_token: float
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
                f"{self.seconds_per_token:.6f}",
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
                "sec_per_tok",
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
            ["git", "rev-parse", "--short", "HEAD"],
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


def _load_models(weights_path: Path):
    """Load the reference and submission models.

    The reference model is loaded with mlx_lm's standard flow (no
    modifiable surface). The submission model is loaded by manually
    constructing the Model class from the modifiable surface and
    calling the participant's load_weights function.
    """
    import mlx_lm
    from mlx_lm.models import gemma4 as _upstream_gemma4

    # Reference: standard mlx_lm.load with no model_file override.
    # We force the upstream module's Model to be the pristine one
    # (in case the modifiable surface patched it before the harness
    # imported it; we re-bind here for safety).
    ref_model, ref_tokenizer = mlx_lm.load(
        str(constants.REFERENCE_WEIGHTS_DIR / constants.REFERENCE_MODEL_DIRNAME)
    )

    # Submission: use mlx_lm.load, which honors config.json's
    # `model_file` pointer to import the participant's modifiable
    # model class. This is the canonical load path and handles all
    # the sanitize / quantize edge cases.
    sub_model, sub_tokenizer = mlx_lm.load(str(weights_path))

    sub_model.eval()
    mx.eval(sub_model.parameters())
    ref_model.eval()
    mx.eval(ref_model.parameters())

    return ref_model, ref_tokenizer, sub_model


def _load_config_dict(weights_path: Path) -> dict:
    config_path = weights_path / "config.json"
    with open(config_path) as f:
        return json.load(f)


def _measure_latency_and_memory(model, prompt: mx.array, num_tokens: int) -> tuple[float, float]:
    """Decode `num_tokens` tokens, return (seconds_per_token, peak_ram_gb).

    Resets the MLX peak memory counter before the decode loop, then
    reads peak memory after eval. This ensures peak RAM captures only
    the decode phase, not prefill or warmup.
    """
    cache = model.make_cache() if hasattr(model, "make_cache") else None
    inner = model.language_model if hasattr(model, "language_model") else model

    # Warmup
    if cache is not None:
        _ = inner(prompt, cache=cache)
    else:
        _ = inner(prompt)
    mx.eval(inner.parameters())

    # Reset peak memory counter before measuring decode.
    mx.reset_peak_memory()

    # Decode loop
    t0 = time.perf_counter()
    next_tok = prompt[-1:]  # seed
    for _ in range(num_tokens):
        if cache is not None:
            logits = inner(next_tok, cache=cache)
        else:
            logits = inner(next_tok)
        # Greedy argmax
        next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(next_tok)
    elapsed = time.perf_counter() - t0

    seconds_per_token = elapsed / num_tokens
    peak_ram_gb = _peak_gpu_memory_gb()

    return seconds_per_token, peak_ram_gb


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
        ref_model, ref_tokenizer, sub_model = _load_models(weights_path)

        # Build a prompt of typical length.
        vocab_size = 262144  # Gemma 4 vocab
        prompt = _seed_prompt(
            vocab_size,
            constants.PROMPT_SEED_PREFIX_LENGTH,
            seed,
        )

        # Correctness check first. A model that fails correctness
        # gets scored as inf and we skip the expensive measurement.
        correctness_result = correctness.check(ref_model, sub_model, prompt)

        if not correctness_result.passed:
            return RunReport(
                timestamp=timestamp,
                git_commit=commit,
                note=note,
                peak_ram_gb=0.0,
                bandwidth_gb_per_token=0.0,
                seconds_per_token=0.0,
                score=float("inf"),
                passed_correctness=False,
                num_layers=correctness_result.num_layers,
                first_failing_layer=correctness_result.first_failing_layer,
                max_abs_diff=correctness_result.max_abs_diff,
                harness_hash=harness_hash,
                mlx_version=mlx_v,
                mlx_lm_version=mlx_lm_v,
            )

        # Measure.
        bw = bandwidth.measure(sub_model, prompt, constants.DECODE_LENGTH)
        spt, peak_gb = _measure_latency_and_memory(
            sub_model, prompt, constants.DECODE_LENGTH
        )
        peak_bytes = int(peak_gb * (1024**3))

        sr = score.compute(
            peak_ram_bytes=peak_bytes,
            bandwidth_gb_per_token=bw.gb_per_token,
            seconds_per_token=spt,
            passed_correctness=True,
            note=note,
        )

        return RunReport(
            timestamp=timestamp,
            git_commit=commit,
            note=note,
            peak_ram_gb=sr.peak_ram_gb,
            bandwidth_gb_per_token=sr.bandwidth_gb_per_token,
            seconds_per_token=sr.seconds_per_token,
            score=sr.score,
            passed_correctness=True,
            num_layers=correctness_result.num_layers,
            first_failing_layer=None,
            max_abs_diff=correctness_result.max_abs_diff,
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
            seconds_per_token=0.0,
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
    parser = argparse.ArgumentParser(description="quantizationfail harness")
    parser.add_argument("--weights", type=Path, default=constants.PARTICIPANT_WEIGHTS_DIR)
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--secret", type=str, default="")
    parser.add_argument(
        "--output-tsv", type=Path, default=constants.RESULTS_FILE
    )
    args = parser.parse_args()

    report = run(args.weights, args.note, args.secret)

    # Append to results.tsv.
    args.output_tsva.parent.mkdir(parents=True, exist_ok=True)
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
        f"sec/tok={report.seconds_per_token:.4f}"
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
