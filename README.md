# quantizationfail

A benchmark arena for memory-bandwidth-optimal LLM inference. Run Gemma 4 26B MoE without materializing the full expert weights.

See [CHALLENGE.md](CHALLENGE.md) for the full problem statement, scoring formula, and approach space.

## Quickstart

```bash
# Install (creates a venv with mlx + mlx-lm + the CLI)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Download the 4-bit reference weights (~18 GB, one-time)
quantizationfail weights

# Run the baseline (no changes) — should match the published baseline score
quantizationfail run --note "baseline, no changes"

# Edit the modifiable surface and iterate
vim mlx_models/gemma4/linear.py
python transform.py
quantizationfail run --note "my schema v1"
```

Results append to `results.tsv`; finite passing runs also write `score.json`.
The harness re-verifies that your `transform.py` is reproducible on every run.

## The modifiable surface

You can edit exactly four files:

| File | Role |
|---|---|
| `mlx_models/gemma4/linear.py` | The `Linear` class used everywhere — attention projections, MLP gate/up/down, etc. The primary compute target. |
| `mlx_models/gemma4/experts.py` | The MoE expert block (`SwitchGLU` + `QuantizedSwitchLinear`). The dominant bandwidth target in 26B-A4B. |
| `mlx_models/gemma4/model.py` | The top-level `Model` class. Layer structure, attention pattern, MoE activation. |
| `mlx_models/gemma4/weights.py` | The `load_weights(model, weights_path)` function. How safetensors are read and mapped onto the model. |

Plus:

- `transform.py` — your offline weight transform. Pure function of `quantizationfail/reference_weights/`.
- `weights/` — the output of `transform.py`. The harness reads from here.

The frozen `mlx_models/gemma4/__init__.py` is the only wiring point. It patches the upstream `mlx_lm.models.gemma4` and `mlx_lm.models.switch_layers` modules with the classes from your 4 files, then re-exports the patched module as `mlx_models.gemma4`. You don't edit `__init__.py`.

## The shadow package

`mlx_models/gemma4/__init__.py` does this at import time:

1. Imports upstream `mlx_lm.models.gemma4` and its submodules.
2. Loads `linear.py` and patches `mlx.nn.Linear` globally with the participant's `Linear` class.
3. Loads `experts.py` and patches `mlx_lm.models.switch_layers.{SwitchGLU,SwitchLinear,QuantizedSwitchLinear}` with the participant's classes.
4. Loads `model.py` and rebinds `mlx_lm.models.gemma4_text.Model` to the participant's `Model` class.
5. Re-exports the patched module under our package name.

The process is dedicated to running this one model, so the global `mlx.nn.Linear` patch is safe.

## Scoring

```
score = peak_ram_GB × bandwidth_GB_per_token × seconds_per_token
```

All three axes are measured independently. Correctness is a hard gate — failing submissions are not scored. See CHALLENGE.md for details.

## Architecture

- `quantizationfail/` — the frozen CLI + harness. Installed as the `quantizationfail` (or short alias `qfail`) command. The CLI verifies the harness's content hash on every run.
- `mlx_models/gemma4/` — the 4 modifiable files plus the frozen `__init__.py` that wires them.
- `quantizationfail/reference_weights/` — the reference 4-bit checkpoint, downloaded by `quantizationfail weights`.
- `transform.py` — your offline weight transform.
- `weights/` — the output of your transform. The harness loads from here.
- `results.tsv` — your local experiment log.

## Versions

- `mlx==0.31.1`
- `mlx-lm>=0.31.2,<0.32` (gemma4.py was added in v0.31.2)
- Python 3.11+
- Apple Silicon (M2 or newer), 24 GB+ unified memory
