"""Harness tamper detection. FROZEN.

The CLI runs this on every `mlxfast run` to verify that
the installed harness matches the hash the server expects. If a
participant has modified any file in `mlxfast/`, the hash
will not match and the CLI refuses to run.

For local development, the hash check is bypassed by setting
MLXFAST_SKIP_HASH_CHECK=1 in the environment. The leaderboard
submission path NEVER sets this.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .harness import constants


def verify() -> None:
    """Compare the installed harness's hash against the expected one.

    Raises SystemExit if they don't match.
    """
    if os.environ.get("MLXFAST_SKIP_HASH_CHECK") == "1":
        return

    expected = constants.EXPECTED_HARNESS_HASH
    if not expected:
        # No expected hash pinned. In dev mode this is fine.
        return

    actual = constants.compute_harness_hash()
    if actual != expected:
        print(
            f"ERROR: harness hash mismatch.\n"
            f"  expected: {expected[:16]}...\n"
            f"  actual:   {actual[:16]}...\n"
            f"\n"
            f"The harness has been modified. This is fine for local\n"
            f"experimentation if you are iterating on the harness\n"
            f"itself, but local results will not match the leaderboard.\n"
            f"To skip this check, set MLXFAST_SKIP_HASH_CHECK=1.",
            file=sys.stderr,
        )
        sys.exit(2)


def current_hash() -> str:
    return constants.compute_harness_hash()
