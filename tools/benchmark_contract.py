#!/usr/bin/env python3
"""Helpers for the benchmark.json shell wrappers."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def source_hash() -> str:
    """Hash the file that determines weights/: transform.py only.

    The transformed weights are a pure function of transform.py and the
    reference checkpoint — NOT of the participant's inference code in
    mlx_models/.  Hashing only transform.py means an mlx_models-only
    submission reuses the cached weights/ (no re-transform), which is what
    lets the macOS benchmark runner skip the reference checkpoint entirely.
    """
    h = hashlib.sha256()
    path = Path("transform.py")
    if not path.exists():
        h.update(b"\0MISSING\0")
    else:
        h.update(path.read_bytes())
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
