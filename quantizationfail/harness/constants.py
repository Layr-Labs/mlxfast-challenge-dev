"""Pinned versions, model paths, harness identity. FROZEN.

This file is the source of truth for what versions the harness was
designed against. The CLI verifies the harness's own content hash
against an expected value at startup, so any modification here will
cause `quantizationfail run` to refuse to run (a soft safety check —
the real one is the server's re-computation).

When the model or mlx-lm version is bumped, the harness hash changes
and participants see a clear error. Bump the EXPECTED_HARNESS_HASH
placeholder to a real value once the harness is server-signed.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

# Pinned versions. mlx-lm 0.21.0 does not exist; the real minimum
# is 0.31.2 (when gemma4.py was added). The challenge spec says
# mlx-lm==0.21.0, which we treat as a forward-looking pin that
# the server will resolve at submission time.
MLX_VERSION = "0.31.1"
MLX_LM_MIN_VERSION = "0.31.2"
MLX_LM_MAX_VERSION = "0.32.0"

# Reference model. The harness downloads this to
# `quantizationfail/reference_weights/` on first run.
REFERENCE_MODEL_REPO = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
REFERENCE_MODEL_DIRNAME = "gemma-4-26B-A4B-it-qat-4bit"

# Modifiable surface. The harness loads these by file path from the
# participant's working directory (not from site-packages).
MODIFIABLE_DIR = Path("mlx_models/gemma4")
MODEL_FILE = MODIFIABLE_DIR / "model.py"

# Output paths (relative to participant's working directory).
PARTICIPANT_WEIGHTS_DIR = Path("weights")
TRANSFORM_SCRIPT = Path("transform.py")
RESULTS_FILE = Path("results.tsv")
SCORE_FILE = Path("score.json")

# Reference weights (managed by `quantizationfail weights`).
REFERENCE_WEIGHTS_DIR = Path("quantizationfail/reference_weights")
TOKENIZER_DIR = Path("quantizationfail/tokenizer")

# Measurement parameters. The challenge spec says 512-token decode runs.
DECODE_LENGTH = 512
PROMPT_SEED_PREFIX_LENGTH = 32

# Numerical tolerance for the correctness gate. The spec calls for
# bfloat16 numerical associativity — reordering of floating point
# operations is permitted, lossy approximation is not. We use
# 1e-2 as a generous bound that accounts for non-deterministic
# GPU reduction order; tighten to 1e-3 if you need stricter matching.
CORRECTNESS_EPSILON = 1e-2

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
    return _harness_dir().parent  # quantizationfail/


def compute_harness_hash() -> str:
    """SHA-256 of every .py file under harness/ + the version pin
    constants. The CLI computes this on each run and compares to the
    EXPECTED_HARNESS_HASH env var or a server-supplied manifest."""
    h = hashlib.sha256()
    # Include the version pins so changing them changes the hash.
    h.update(f"mlx={MLX_VERSION}\nmlx-lm>={MLX_LM_MIN_VERSION}\n".encode())
    for path in sorted(_harness_dir().rglob("*.py")):
        h.update(path.read_bytes())
    return h.hexdigest()


# Set by the server when the participant installs the harness wheel.
# If unset (local dev), the CLI accepts any harness hash — useful for
# iterating on the harness itself, dangerous for leaderboard integrity.
EXPECTED_HARNESS_HASH = os.environ.get("QUANTIZATIONFAIL_EXPECTED_HARNESS_HASH", "")
