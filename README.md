# mlxfast

A benchmark arena for memory-bandwidth-optimal LLM inference. Run Gemma 4 26B MoE without materializing the full expert weights.

See [CHALLENGE.md](CHALLENGE.md) for the full problem statement, scoring formula, and approach space.

## Quickstart

```bash
# Install (creates a venv with mlx + mlx-lm + the CLI)
python -m venv .venv && source .venv/bin/activate
pip install -e . sentencepiece

# Download the 4-bit QAT reference weights (~18 GB, one-time)
mlxfast weights

# Copy the reference weights into weights/ as your starting point
REF=mlxfast/reference_weights/gemma-4-26B-A4B-it-qat-4bit
cp $REF/config.json weights/config.json
python -c "
import json
c = json.load(open('weights/config.json'))
c['model_file'] = '../mlx_models/gemma4/model.py'
json.dump(c, open('weights/config.json','w'), indent=2)
"
for f in $REF/*.safetensors; do ln -sf "../$f" "weights/$(basename $f)"; done
ln -sf "../$REF/tokenizer.json" weights/tokenizer.json
ln -sf "../$REF/tokenizer_config.json" weights/tokenizer_config.json

# Run the baseline — should match the published baseline score
MLXFAST_SKIP_HASH_CHECK=1 mlxfast run --note "baseline" --skip-transform-verify

# Edit the modifiable surface and iterate
vim mlx_models/gemma4/linear.py
python transform.py
MLXFAST_SKIP_HASH_CHECK=1 mlxfast run --note "my schema v1"
```

Results append to `results.tsv`. View them with:

```bash
column -t -s $'\t' results.tsv
```

## The modifiable surface

You can edit exactly four files:

| File | Role |
|---|---|
| `mlx_models/gemma4/linear.py` | The `Linear` class used everywhere — attention projections, MLP gate/up/down, etc. The primary compute target. |
| `mlx_models/gemma4/experts.py` | The MoE expert block (`SwitchGLU` + `QuantizedSwitchLinear`). The dominant bandwidth target in 26B-A4B. |
| `mlx_models/gemma4/model.py` | The top-level `Model` class. Layer structure, attention pattern, MoE activation. |
| `mlx_models/gemma4/weights.py` | The `load_weights(model, weights_path)` function. How safetensors are read and mapped onto the model. |

Plus:

- `transform.py` — your offline weight transform. Pure function of `mlxfast/reference_weights/`.
- `weights/` — the output of `transform.py`. The harness reads from here.

The frozen `mlx_models/gemma4/__init__.py` is the only wiring point. It patches `mlx.nn.Linear` and `mlx_lm.models.switch_layers` with your classes at import time. You don't edit `__init__.py`.

## Scoring

```
score = peak_ram_GB × bandwidth_GB_per_token × decode_sec_per_token × prefill_sec_per_token
```

All four axes are measured independently. Correctness is a hard gate — failing submissions are not scored. See CHALLENGE.md for details.

**Baseline (M5 Max 128 GB, QAT 4-bit):**

| Peak RAM | Bandwidth | Decode | Prefill | Score |
|---|---|---|---|---|
| 27.1 GB | 13.52 GB/tok | 0.0141 s/tok (~71 tok/s) | 0.00128 s/tok (~780 tok/s) | 0.0066 |

## Architecture

- `mlxfast/` — the frozen CLI + harness. Installed as the `mlxfast` (or short alias `qfail`) command.
- `mlx_models/gemma4/` — the 4 modifiable files plus the frozen `__init__.py` that wires them.
- `mlxfast/reference_weights/` — the reference QAT 4-bit checkpoint, downloaded by `mlxfast weights`.
- `transform.py` — your offline weight transform.
- `weights/` — the output of your transform. The harness loads from here.
- `results.tsv` — your local experiment log.
- `score.json` — written after each finite passing run (benchmark contract format).

## Requirements

- Apple Silicon Mac (M2 or newer), 24 GB+ unified memory
- macOS Sequoia or later
- Python 3.11+
- `mlx==0.31.1`, `mlx-lm>=0.31.2,<0.32`
