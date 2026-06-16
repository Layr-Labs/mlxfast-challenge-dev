#!/usr/bin/env python3
"""Helpers for the benchmark.json shell wrappers."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def source_hash() -> str:
    """Hash all participant-editable source files: transform.py + mlx_models/**/*.py."""
    h = hashlib.sha256()
    paths = [Path("transform.py"), *sorted(Path("mlx_models").rglob("*.py"))]
    for path in paths:
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
