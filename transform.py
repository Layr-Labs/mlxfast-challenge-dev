"""transform.py — identity pass-through. PARTICIPANT REPLACES.

This is the offline weight transform. It runs ONCE, produces the
participant's `weights/` directory from `reference_weights/`, and
is then content-hashed by the harness for provenance.

The baseline version below is a no-op: it copies the reference
weights to weights/ unchanged. A submission with this transform
will get the same score as the published baseline.

Replace `main()` with your actual transform. Constraints enforced
by the harness sandbox:

  - Only reads from `quantizationfail/reference_weights/`
  - Only writes to `weights/`
  - No network, no subprocess, no env reads, no clock reads
  - Same inputs must produce byte-equal output (re-verified on
    every `quantizationfail run`)

The harness discovers the model architecture from
`reference_weights/gemma-4-26B-A4B-it-qat-4bit/config.json` (passed
through into `weights/config.json` with `model_file` pointing
back at the participant's modifiable model.py).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REFERENCE = Path("quantizationfail/reference_weights/gemma-4-26B-A4B-it-qat-4bit")
OUTPUT = Path("weights")


def main():
    """Identity pass-through. Replace with your transform."""
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Copy config.json and rewrite the model_file path so mlx-lm
    # loads the participant's modifiable model.py.
    src_config = REFERENCE / "config.json"
    with open(src_config) as f:
        config = json.load(f)
    config["model_file"] = "../mlx_models/gemma4/model.py"
    with open(OUTPUT / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer files (they're not weights; transform.py is
    # allowed to copy them through).
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = REFERENCE / name
        if src.exists():
            shutil.copy2(src, OUTPUT / name)

    # Copy safetensors unchanged.
    for src in sorted(REFERENCE.glob("*.safetensors")):
        dst = OUTPUT / src.name
        print(f"copying {src.name} ({src.stat().st_size / 1e9:.2f} GB)")
        shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
