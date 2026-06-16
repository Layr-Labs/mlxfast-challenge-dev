"""Pinned versions, model paths, harness identity. FROZEN.

This file is the source of truth for what versions the harness was
designed against. The CLI verifies the harness's own content hash
against an expected value at startup, so any modification here will
cause `mlxfast run` to refuse to run (a soft safety check —
the real one is the server's re-computation).

When the model or mlx-lm version is bumped, the harness hash changes
and participants see a clear error. Bump the EXPECTED_HARNESS_HASH
placeholder to a real value once the harness is server-signed.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

MLX_VERSION = "0.31.2"
MLX_LM_MIN_VERSION = "0.31.2"
MLX_LM_MAX_VERSION = "0.32.0"
MLX_VLM_VERSION = "0.6.3"

# Reference model. The harness downloads this to
# `mlxfast/reference_weights/` on first run.
REFERENCE_MODEL_REPO = "mlx-community/DeepSeek-V4-Flash-4bit"
REFERENCE_MODEL_DIRNAME = "DeepSeek-V4-Flash-4bit"

# DeepSeek V4 Flash model constants (from config.json).
VOCAB_SIZE = 129280
NUM_HIDDEN_LAYERS = 43
N_ROUTED_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 6

# Modifiable surface. The harness loads by module path from the
# participant's working directory (prepended to sys.path).
MODIFIABLE_DIR = Path("mlx_models/deepseek_v4")

# Output paths (relative to participant's working directory).
PARTICIPANT_WEIGHTS_DIR = Path("weights")
TRANSFORM_SCRIPT = Path("transform.py")
RESULTS_FILE = Path("results.tsv")
SCORE_FILE = Path("score.json")

# Reference weights (managed by `mlxfast weights`).
REFERENCE_WEIGHTS_DIR = Path("mlxfast/reference_weights")
TOKENIZER_DIR = Path("mlxfast/tokenizer")

# Measurement parameters.
# DECODE_LENGTH: number of autoregressive tokens measured per run.
# PREFILL_PROMPT_LENGTH: length of the prompt used for the prefill
#   latency measurement. Longer than the correctness seed prompt so
#   that prefill timing is dominated by the actual computation rather
#   than framework overhead.
# PROMPT_SEED_PREFIX_LENGTH: length of the seed prompt used for the
#   correctness gate and as the starting context for decode timing.
DECODE_LENGTH = 512
PREFILL_PROMPT_LENGTH = 512
PROMPT_SEED_PREFIX_LENGTH = 32

# Numerical tolerance for the correctness gate. The spec calls for
# bfloat16 numerical associativity — reordering of floating point
# operations is permitted, lossy approximation is not. We use
# 1e-2 as a generous bound that accounts for non-deterministic
# GPU reduction order; tighten to 1e-3 if you need stricter matching.
CORRECTNESS_EPSILON = 5e-3

# Scoring formula. Lower is better.
#   score = peak_ram_GB * bandwidth_GB_per_token * seconds_per_token
# All three axes are measured independently and stored in results.tsv.


def _harness_dir() -> Path:
    return Path(__file__).resolve().parent


def harness_root() -> Path:
    """The directory containing the frozen harness code.

    This is the path the self-hash check verifies. If a participant
    edits anything in here, the CLI will refuse to run.
    """
    return _harness_dir().parent  # mlxfast/


def compute_harness_hash() -> str:
    """SHA-256 of every .py file under the mlxfast/ package + version pins.

    Covers harness/, cli.py, _harness_runner.py, _sandbox.py, _self_hash.py
    so that modifying any part of the harness changes the hash.
    """
    h = hashlib.sha256()
    # Include the version pins so changing them changes the hash.
    h.update(f"mlx={MLX_VERSION}\nmlx-lm>={MLX_LM_MIN_VERSION}\n".encode())
    for path in sorted(harness_root().rglob("*.py")):
        h.update(path.read_bytes())
    return h.hexdigest()


# Set by the server when the participant installs the harness wheel.
# If unset (local dev), the CLI accepts any harness hash — useful for
# iterating on the harness itself, dangerous for leaderboard integrity.
EXPECTED_HARNESS_HASH = os.environ.get("MLXFAST_EXPECTED_HARNESS_HASH", "")
