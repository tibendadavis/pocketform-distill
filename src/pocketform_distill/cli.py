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
    sections = len(cfg.load_form().sections)
    typer.echo(f"Wrote {cfg.synth.num_photos} photos x {sections} section(s) to {out} ({dict(splits)})")


@app.command()
def teacher(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="Run config YAML")],
    data_root: Annotated[Path, typer.Option("--data", help="Root folder for stage outputs")] = Path("data"),
    limit: Annotated[int | None, typer.Option("--limit", "-n", help="Label at most N new photos (try it out)")] = None,
    section: Annotated[str | None, typer.Option("--section", "-s", help="Only this section's photos")] = None,
) -> None:
    """Stage 2: the teacher reads every synth photo; saves its answers and probabilities. Resumable."""
    import json
    import time

    from pocketform_distill.config import load_run_config
    from pocketform_distill.teacher import run_teacher

    cfg = load_run_config(config)
    t0 = time.monotonic()
    skipped_before: list[int] = []  # photos already labelled before the first new one (they took no time)

    def progress(done: int, total: int, rec: dict) -> None:
        if not skipped_before:
            skipped_before.append(done - 1)
        new = done - skipped_before[0]
        left = (time.monotonic() - t0) / new * (total - done)
        right = sum(rec["correct"].values())
        typer.echo(f"[{done}/{total}] {rec['id']}: {right}/{len(rec['correct'])} fields right, "
                   f"{rec['seconds']:.0f} s, ~{left / 60:.0f} min left")

    out = run_teacher(cfg, data_root, limit=limit, section=section, progress=progress)
    summary = json.loads((out / "run.json").read_text())["summary"]
    for sid, s in summary["sections"].items():
        typer.echo(f"section {sid}: {s['field_accuracy']:.1%} of {s['fields']} fields right "
                   f"({s['photos']} photos, {s['one_line']} one-line replies)")
    typer.echo(f"teacher field accuracy: {summary['field_accuracy']:.1%} -> {out}")


@app.command()
def dataset(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="Run config YAML")],
    data_root: Annotated[Path, typer.Option("--data", help="Root folder for stage outputs")] = Path("data"),
) -> None:
    """Stage 3: training examples from synth truth (hard labels) + teacher probabilities (soft labels)."""
    import json

    from pocketform_distill.config import load_run_config
    from pocketform_distill.dataset import run_dataset

    out = run_dataset(load_run_config(config), data_root)
    for sid, s in json.loads((out / "run.json").read_text())["sections"].items():
        share = s["soft_share_of_value_positions"]
        typer.echo(f"section {sid}: {s['photos']} photos ({s['with_teacher']} with teacher), teacher right on "
                   f"{s['teacher_right']}/{s['fields']} fields, soft labels on "
                   f"{'-' if share is None else f'{share:.0%}'} of value positions, "
                   f"{s['target_tokens'] / s['photos']:.0f} answer tokens per photo")
    typer.echo(f"-> {out}")


@app.command()
def prompt(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="Run config YAML")],
    section: Annotated[str | None, typer.Option("--section", "-s", help="Section id (default: all sections)")] = None,
) -> None:
    """Print the exact prompt every stage sends for each section of this run's form, and its sha256."""
    from pocketform_distill.config import load_run_config
    from pocketform_distill.prompt import build_prompt, prompt_info

    spec = load_run_config(config).load_form()
    for s in [spec.section(section)] if section else spec.sections:
        info = prompt_info(spec, s)
        typer.echo(build_prompt(spec, s) + "\n")
        typer.echo(f"# section={s.id} kind={info['kind']} chars={info['chars']} sha256={info['sha256']}\n", err=True)


@app.command("version")
def show_version() -> None:
    """Print the package version."""
    try:
        typer.echo(f"pocketform-distill {version('pocketform-distill')}")
    except PackageNotFoundError:
        typer.echo("pocketform-distill unknown")
