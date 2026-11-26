"""Command line: `uv run pocketform-distill <stage> --config configs/<run>.yaml`."""

from __future__ import annotations

from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Distil a big VLM into a small form reader.")


@app.command()
def synth(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="Run config YAML")],
    data_root: Annotated[Path, typer.Option("--data", help="Root folder for stage outputs")] = Path("data"),
) -> None:
    """Stage 1: generate synthetic filled forms with truth labels."""
    import json

    from pocketform_distill.config import load_run_config
    from pocketform_distill.synth import run_synth

    cfg = load_run_config(config)
    out = run_synth(cfg, data_root)
    splits = Counter(json.loads(line)["split"] for line in (out / "manifest.jsonl").read_text().splitlines())
    typer.echo(f"Wrote {cfg.synth.num_forms} forms to {out} ({dict(splits)})")


@app.command("version")
def show_version() -> None:
    """Print the package version."""
    try:
        typer.echo(f"pocketform-distill {version('pocketform-distill')}")
    except PackageNotFoundError:
        typer.echo("pocketform-distill unknown")
