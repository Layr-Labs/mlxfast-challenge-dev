"""Harness subprocess runner.

The harness (`mlxfast run`) executes the actual model
loading and measurement in a clean subprocess. The subprocess:

  - Has a fresh Python interpreter, so the global `mlx.nn.Linear`
    patch from a previous run doesn't leak.
  - Has its own `getrusage`-tracked peak RSS, so the peak RAM
    measurement reflects the harness's footprint and not the CLI's.
  - Imports the harness from the installed location (verified by
    hash), not from the participant's repo.

This module is the bridge: the CLI calls `run_in_subprocess`,
which spawns a fresh Python that imports `mlxfast.harness.run`
and calls `run(weights, note, secret)`. The result is returned as
JSON via stdout.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .harness import constants


SUBPROCESS_SCRIPT = '''
import json
import sys
import os

# Run from the participant's working directory so relative paths
# in transform.py and mlx_models/ resolve correctly.
os.chdir({cwd!r})

# Add the participant's working directory to sys.path so
# `import mlx_models.deepseek_v4` resolves to their modifiable surface.
sys.path.insert(0, {cwd!r})

from mlxfast.harness import run as harness_run

report = harness_run.run(
    weights_path=__import__("pathlib").Path({weights!r}),
    note={note!r},
    secret={secret!r},
)
print("__MLXFAST_RESULT__" + json.dumps(report.to_tsv_row().split("\\t")))
'''


def run_in_subprocess(
    weights_path: Path,
    note: str,
    secret: str = "",
    cwd: Optional[Path] = None,
    python: str = sys.executable,
) -> dict:
    """Run the harness in a fresh subprocess and return the parsed report."""
    cwd = (cwd or Path.cwd()).resolve()
    script = SUBPROCESS_SCRIPT.format(
        cwd=str(cwd),
        weights=str(weights_path.resolve()),
        note=note,
        secret=secret,
    )

    proc = subprocess.run(
        [python, "-c", script],
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour hard cap
    )

    # The harness prints "__MLXFAST_RESULT__<json>" on success.
    for line in proc.stdout.splitlines():
        if line.startswith("__MLXFAST_RESULT__"):
            payload = line.removeprefix("__MLXFAST_RESULT__")
            values = json.loads(payload)
            header = constants.RESULTS_FILE.read_text().splitlines()[0].split("\t") if constants.RESULTS_FILE.exists() else []
            # If the header hasn't been written yet, return a raw dict.
            _COLUMNS = [
                "timestamp", "commit", "note", "peak_ram_gb",
                "bandwidth_gb_per_tok", "decode_sec_per_tok",
                "prefill_sec_per_tok", "score", "passed",
                "num_layers", "first_failing_layer", "max_abs_diff",
                "bandwidth_source", "harness_hash",
            ]
            if len(values) == len(_COLUMNS):
                return dict(zip(_COLUMNS, values))
            return {"raw": values}

    # Harness didn't produce a result line. Something went wrong.
    return {
        "error": "harness did not produce a result",
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
    }
