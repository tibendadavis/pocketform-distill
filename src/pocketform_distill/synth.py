"""Stage 1: synth. Fake filled forms with known truth.

Writes to data/<run_name>/synth/:
    {id}.png         the synthetic "phone photo"
    {id}.json        truth keyed by field id, null = left empty
    manifest.jsonl   one line per form: id, split (train/val), seed, plus the writer and photo defects drawn
                     from that seed (for tracing bad images; regenerating only needs the seed)
    run.json         config + git commit this ran with

Never reads real forms (private/): they are the test set only.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

from pocketform_distill.augment import apply_photo_defects
from pocketform_distill.config import FormSpec, RunConfig, SynthConfig
from pocketform_distill.provenance import write_run_info
from pocketform_distill.render import Writer
from pocketform_distill.values import sample_value, written_text


def list_fonts(fonts_dir: Path) -> list[Path]:
    fonts = sorted(Path(fonts_dir).glob("*.ttf"))  # sorted: glob order differs between machines
    if not fonts:
        raise FileNotFoundError(f"no .ttf handwriting fonts in {fonts_dir}")
    return fonts


def make_truth(spec: FormSpec, cfg: SynthConfig, rng: random.Random) -> dict[str, int | str | None]:
    return {f.id: None if rng.random() < cfg.empty_rate else sample_value(f, rng) for f in spec.fields}


def _rounded(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_rounded(v) for v in x]
    return x


def generate_form(spec: FormSpec, blank: Image.Image, cfg: SynthConfig, fonts: list[Path], seed: int
                  ) -> tuple[Image.Image, dict[str, int | str | None], dict[str, Any]]:
    """One synthetic form: sample values, handwrite them, then photograph it badly.

    Returns the image, the truth, and what was drawn at random (writer + defects) for the manifest.
    """
    rng = random.Random(seed)
    truth = make_truth(spec, cfg, rng)
    writer = Writer.random(fonts, cfg.ink_colors, rng)
    ink = Image.new("L", blank.size, 0)
    for f in spec.fields:
        if truth[f.id] is not None:
            writer.write(ink, f, written_text(f, truth[f.id]), rng)
    img, defects = apply_photo_defects(writer.apply(blank, ink), cfg.augment, rng)
    writer_info = asdict(writer) | {"font": Path(writer.font).name}
    return img, truth, _rounded({"writer": writer_info, "defects": defects})


def val_indices(n: int, val_fraction: float, seed: int) -> set[int]:
    """Exactly round(n * val_fraction) forms go to val, picked by seed (not per form, so the count is exact)."""
    return set(random.Random(seed).sample(range(n), round(n * val_fraction)))


def run_synth(config: RunConfig, data_root: Path = Path("data")) -> Path:
    spec = config.load_form()
    cfg = config.synth
    fonts = list_fonts(cfg.fonts_dir)
    blank = spec.open_template()

    out = data_root / config.run_name / "synth"
    if out.exists():
        shutil.rmtree(out)  # this stage owns the folder; stale forms from a bigger earlier run must not linger
    out.mkdir(parents=True)

    val = val_indices(cfg.num_forms, cfg.val_fraction, cfg.seed)
    manifest = []
    for i in range(cfg.num_forms):
        form_id = f"{i:05d}"
        seed = cfg.seed + i
        img, truth, drawn = generate_form(spec, blank, cfg, fonts, seed)
        img.save(out / f"{form_id}.png")
        (out / f"{form_id}.json").write_text(json.dumps(truth, indent=2) + "\n")
        manifest.append({"id": form_id, "split": "val" if i in val else "train", "seed": seed, **drawn})

    (out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    write_run_info(out, "synth", config, form=spec.model_dump(mode="json"), fonts=[f.name for f in fonts],
                   template_sha256=hashlib.sha256(spec.template_path.read_bytes()).hexdigest())
    return out
