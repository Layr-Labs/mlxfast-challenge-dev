"""mlxfast CLI.

Subcommands:
  - run       Run the harness on the current weights/ and modifiable
             surface. Appends to results.tsv.
  - submit    Package transform.py + weights/ + modifiable surface
             hashes and upload to the leaderboard server. (Stub:
             prints what it would send.)
  - weights   Download the reference weights to
             mlxfast/reference_weights/. Idempotent.
  - login     Store the API key in ~/.config/mlxfast/.
             (Stub.)
  - clone     Initialize a local working directory from the
             challenge template. (Stub: prints a hint.)
  - verify    Re-run transform.py in a sandbox and check that
             weights/ is byte-equal to a clean re-run.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import _self_hash
from .harness import constants

app = typer.Typer(
    name="mlxfast",
    help="Benchmark arena for memory-bandwidth-optimal LLM inference",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


@app.command()
def run(
    note: str = typer.Option("", "--note", "-n", help="Free-form note recorded in results.tsv"),
    secret: str = typer.Option("", "--secret", help="Server-supplied secret for prompt seed (default: local deterministic)"),
    skip_transform_verify: bool = typer.Option(False, "--skip-transform-verify", help="Skip the sandboxed transform.py re-run (faster, less safe)"),
    weights: Path = typer.Option(constants.PARTICIPANT_WEIGHTS_DIR, "--weights", "-w", help="Path to the participant's transformed weights"),
    score_path: Path = typer.Option(constants.SCORE_FILE, "--score-path", help="Path to write JSON score output"),
):
    """Run the harness: correctness + bandwidth + latency.

    Appends to results.tsv and writes score.json for finite passing runs.
    """
    from . import _harness_runner
    from . import _sandbox

    _self_hash.verify()
    if score_path:
        score_path.unlink(missing_ok=True)

    if not skip_transform_verify:
        if constants.TRANSFORM_SCRIPT.exists():
            console.print(f"[bold]Verifying transform.py reproducibility...[/bold]")
            t0 = time.perf_counter()
            ok, msg = _sandbox.verify_transform(
                constants.TRANSFORM_SCRIPT,
                constants.REFERENCE_WEIGHTS_DIR / constants.REFERENCE_MODEL_DIRNAME,
                weights,
            )
            elapsed = time.perf_counter() - t0
            if not ok:
                console.print(f"[red]Transform verification FAILED ({elapsed:.1f}s):[/red]\n{msg}")
                raise typer.Exit(1)
            console.print(f"[green]Transform verified ({elapsed:.1f}s).[/green] hash={str(msg)[:16]}...")

    console.print(f"[bold]Running harness...[/bold]")
    t0 = time.perf_counter()
    report = _harness_runner.run_in_subprocess(
        weights_path=weights,
        note=note,
        secret=secret,
    )
    elapsed = time.perf_counter() - t0

    if "error" in report and "raw" not in report:
        console.print(f"[red]Harness failed:[/red]\n{report.get('error')}\n{report.get('stderr', '')}")
        raise typer.Exit(1)

    _append_to_results_tsv(report)
    _write_score_json(report, score_path)
    _print_report_table(report, harness_seconds=elapsed)


@app.command()
def submit(
    note: str = typer.Option("", "--note", "-n", help="Submission note"),
    weights: Path = typer.Option(constants.PARTICIPANT_WEIGHTS_DIR, "--weights", "-w"),
):
    """Package and submit to the leaderboard server.

    STUB: in this build, prints what it would send. The real server
    endpoint is configured via MLXFAST_API_URL.
    """
    from . import _sandbox

    _self_hash.verify()

    if not weights.exists():
        console.print(f"[red]No weights at {weights}. Run `python transform.py` first.[/red]")
        raise typer.Exit(1)

    # Re-verify transform.py is reproducible.
    if constants.TRANSFORM_SCRIPT.exists():
        ok, msg = _sandbox.verify_transform(
            constants.TRANSFORM_SCRIPT,
            constants.REFERENCE_WEIGHTS_DIR / constants.REFERENCE_MODEL_DIRNAME,
            weights,
        )
        if not ok:
            console.print(f"[red]Transform verification FAILED:[/red]\n{msg}")
            raise typer.Exit(1)

    payload = _build_submission_payload(weights, note)

    api_url = os.environ.get("MLXFAST_API_URL", "")
    if not api_url:
        console.print(
            f"[yellow]MLXFAST_API_URL is not set.[/yellow]\n"
            f"Submission payload (would POST to {{api_url}}/submit):\n"
        )
        console.print_json(data=payload)
        return

    console.print(f"[bold]Submitting to {api_url}/submit...[/bold]")
    # Real implementation: POST payload via httpx. Out of scope for the
    # client-only build; the server is built separately.
    raise typer.Exit(0)


@app.command()
def weights(
    force: bool = typer.Option(False, "--force", help="Re-download even if present"),
):
    """Download the reference DeepSeek-V4-Flash 4-bit weights.

    Downloads mlx-community/DeepSeek-V4-Flash-4bit to
    mlxfast/reference_weights/. This is a one-time ~141 GB
    download. Idempotent.
    """
    target = constants.REFERENCE_WEIGHTS_DIR / constants.REFERENCE_MODEL_DIRNAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        console.print(f"[green]Reference weights already at {target}.[/green]")
        return

    console.print(
        f"[bold]Downloading {constants.REFERENCE_MODEL_REPO}...[/bold]\n"
        f"This is ~18 GB. Set force=True to re-download."
    )
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=constants.REFERENCE_MODEL_REPO,
            local_dir=str(target),
            allow_patterns=[
                "*.json",
                "model*.safetensors",
                "tokenizer*",
                "*.tiktoken",
                "tiktoken.model",
                "*.txt",
            ],
        )
        console.print(f"[green]Downloaded to {target}.[/green]")
    except ImportError:
        console.print(
            f"[red]huggingface_hub not installed. Run:[/red]\n"
            f"  pip install huggingface_hub"
        )
        raise typer.Exit(1)


@app.command()
def login(
    api_key: str = typer.Argument(..., help="API key from mlxfast.com/account"),
):
    """Store the API key for submissions. STUB in this build."""
    cred_dir = Path.home() / ".config" / "mlxfast"
    cred_dir.mkdir(parents=True, exist_ok=True)
    cred_file = cred_dir / "credentials"
    cred_file.write_text(json.dumps({"api_key": api_key, "stored_at": time.time()}))
    cred_file.chmod(0o600)
    console.print(f"[green]Credentials stored at {cred_file}.[/green]")
    console.print(f"[yellow]Note: server endpoint not yet wired. Set MLXFAST_API_URL when the server is live.[/yellow]")


@app.command()
def clone():
    """Initialize a local working directory from the challenge template. STUB."""
    cwd = Path.cwd()
    required = [
        constants.MODIFIABLE_DIR / "deepseek_v4.py",
        constants.MODIFIABLE_DIR / "language.py",
        constants.MODIFIABLE_DIR / "config.py",
        constants.MODIFIABLE_DIR / "hyper_connection.py",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        console.print(f"[red]Missing files in {cwd}:[/red]")
        for m in missing:
            console.print(f"  - {m}")
        console.print(
            f"\n[yellow]Run this in a fresh clone of the mlxfast-challenge repo.[/yellow]"
        )
        raise typer.Exit(1)
    console.print(f"[green]Already initialized at {cwd}.[/green]")


@app.command()
def verify_transform(
    weights: Path = typer.Option(constants.PARTICIPANT_WEIGHTS_DIR, "--weights", "-w"),
):
    """Re-run transform.py in a sandbox and verify byte-equal output."""
    from . import _sandbox

    if not constants.TRANSFORM_SCRIPT.exists():
        console.print(f"[red]No transform.py at {constants.TRANSFORM_SCRIPT}.[/red]")
        raise typer.Exit(1)
    if not (constants.REFERENCE_WEIGHTS_DIR / constants.REFERENCE_MODEL_DIRNAME).exists():
        console.print(
            f"[red]Reference weights not found at "
            f"{constants.REFERENCE_WEIGHTS_DIR / constants.REFERENCE_MODEL_DIRNAME}.[/red]\n"
            f"Run `mlxfast weights` first."
        )
        raise typer.Exit(1)

    console.print(f"[bold]Verifying transform.py...[/bold]")
    t0 = time.perf_counter()
    ok, msg = _sandbox.verify_transform(
        constants.TRANSFORM_SCRIPT,
        constants.REFERENCE_WEIGHTS_DIR / constants.REFERENCE_MODEL_DIRNAME,
        weights,
    )
    elapsed = time.perf_counter() - t0
    if not ok:
        console.print(f"[red]FAIL ({elapsed:.1f}s):[/red]\n{msg}")
        raise typer.Exit(1)
    console.print(f"[green]OK ({elapsed:.1f}s).[/green] hash={str(msg)[:16]}...")


def _append_to_results_tsv(report: dict) -> None:
    """Append a row to results.tsv in the participant's working directory."""
    results = constants.RESULTS_FILE
    _COLUMNS = [
        "timestamp", "commit", "note", "peak_ram_gb",
        "bandwidth_gb_per_tok", "decode_sec_per_tok", "prefill_sec_per_tok",
        "score", "passed", "num_layers", "first_failing_layer", "max_abs_diff",
        "bandwidth_source", "harness_hash",
    ]

    if not results.exists():
        results.write_text("\t".join(_COLUMNS) + "\n")

    row = "\t".join([str(report.get(k, "")) for k in _COLUMNS])
    with open(results, "a") as f:
        f.write(row + "\n")


def _write_score_json(report: dict, score_path: Path) -> None:
    """Write score.json in the benchmark.json contract format.

    Always written — even for failed or errored runs — so CI can
    distinguish a correctness failure from a missing-file harness error.
    Failed runs use score=null so Yukon/CI can detect them.
    """
    if "raw" in report:
        return

    score = _finite_float(report.get("score"))
    passed = report.get("passed", "0") in ("1", True, "true", "True")

    payload = {
        "score": score,  # null for failed/errored runs
        "passed": passed,
        "metrics": {
            "peak_ram_gb": _float_metric(report, "peak_ram_gb"),
            "bandwidth_gb_per_token": _float_metric(report, "bandwidth_gb_per_tok"),
            "decode_seconds_per_token": _float_metric(report, "decode_sec_per_tok"),
            "prefill_seconds_per_token": _float_metric(report, "prefill_sec_per_tok"),
            "passed_correctness": passed,
            "num_layers": _int_metric(report, "num_layers"),
            "first_failing_layer": _optional_int_metric(report, "first_failing_layer"),
            "max_abs_diff": _float_metric(report, "max_abs_diff"),
            "bandwidth_source": report.get("bandwidth_source", ""),
            "error": report.get("error") or "",
            "commit": report.get("commit", ""),
            "timestamp": report.get("timestamp", ""),
            "harness_hash": report.get("harness_hash", ""),
        },
    }
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _float_metric(report: dict, key: str) -> float:
    value = report.get(key, "")
    if value in ("", "-"):
        return 0.0
    return float(value)


def _int_metric(report: dict, key: str) -> int:
    value = report.get(key, "")
    if value in ("", "-"):
        return 0
    return int(value)


def _optional_int_metric(report: dict, key: str) -> Optional[int]:
    value = report.get(key, "")
    if value in ("", "-"):
        return None
    return int(value)


def _print_report_table(report: dict, harness_seconds: float) -> None:
    table = Table(title="Harness Report", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    if "raw" in report:
        console.print_json(data=report["raw"])
        return

    passed = report.get("passed", "0") in ("1", True, "true")
    score = report.get("score", "inf")
    if passed:
        try:
            score_v = float(score)
            score_s = f"{score_v:.4f}"
        except ValueError:
            score_s = str(score)
    else:
        score_s = "[red]inf (correctness failed)[/red]"

    rows = [
        ("Status", "[green]PASS[/green]" if passed else "[red]FAIL[/red]"),
        ("Score", score_s),
        ("Peak RAM (GB)", report.get("peak_ram_gb", "?")),
        ("Bandwidth (GB/tok)", report.get("bandwidth_gb_per_tok", "?")),
        ("Decode sec/token", report.get("decode_sec_per_tok", "?")),
        ("Prefill sec/token", report.get("prefill_sec_per_tok", "?")),
        ("Layers checked", report.get("num_layers", "?")),
        ("First failing layer", report.get("first_failing_layer", "-")),
        ("Max abs diff", report.get("max_abs_diff", "?")),
        ("Bandwidth source", report.get("bandwidth_source", "?")),
        ("Harness hash", str(report.get("harness_hash", "?"))[:16] + "..."),
        ("Harness wall time", f"{harness_seconds:.1f}s"),
    ]
    for k, v in rows:
        table.add_row(k, str(v))
    console.print(table)


def _build_submission_payload(weights: Path, note: str) -> dict:
    """Build the dict that would be POSTed to /submit."""
    import hashlib

    def _hash_file(p: Path) -> str:
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()

    modifiable_hashes = {}
    for f in [
        constants.MODIFIABLE_DIR / "deepseek_v4.py",
        constants.MODIFIABLE_DIR / "language.py",
        constants.MODIFIABLE_DIR / "config.py",
        constants.MODIFIABLE_DIR / "hyper_connection.py",
    ]:
        if f.exists():
            modifiable_hashes[str(f)] = _hash_file(f)

    weights_hashes = {}
    for p in sorted(weights.rglob("*.safetensors")):
        weights_hashes[str(p.relative_to(weights))] = _hash_file(p)

    return {
        "transform_source": constants.TRANSFORM_SCRIPT.read_text() if constants.TRANSFORM_SCRIPT.exists() else None,
        "transform_hash": _hash_file(constants.TRANSFORM_SCRIPT) if constants.TRANSFORM_SCRIPT.exists() else None,
        "modifiable_hashes": modifiable_hashes,
        "weights_hashes": weights_hashes,
        "harness_hash": _self_hash.current_hash(),
        "note": note,
        "timestamp": time.time(),
    }


if __name__ == "__main__":
    app()
