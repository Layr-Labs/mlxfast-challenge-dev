#!/usr/bin/env python3
"""Helpers for the benchmark.json shell wrappers."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PARTICIPANT_SOURCE_PATHS = [
    Path("transform.py"),
    Path("mlx_models/gemma4/linear.py"),
    Path("mlx_models/gemma4/experts.py"),
    Path("mlx_models/gemma4/model.py"),
    Path("mlx_models/gemma4/weights.py"),
]


def source_hash() -> str:
    h = hashlib.sha256()
    for path in PARTICIPANT_SOURCE_PATHS:
        if not path.exists():
            h.update(str(path).encode())
            h.update(b"\0MISSING\0")
            continue
        h.update(str(path).encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("source-hash", help="Hash participant-editable source files")

    args = parser.parse_args()
    if args.command == "source-hash":
        print(source_hash())
    else:
        parser.error("unknown command")


if __name__ == "__main__":
    main()
