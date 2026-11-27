"""Stage 1: synth. Fake filled-in section photos with known truth.

A sectioned form is photographed one section at a time, so every synthetic image is one section, written on that
section's blank template. `synth.num_photos` images are made per section.

Writes to data/<run_name>/synth/:
    {id}.png         the synthetic "phone photo" of one section; id = <section>-<n>, e.g. b-00042
    {id}.json        truth for that section's fields, keyed by field id, null = left empty
    manifest.jsonl   one line per photo: id, section, split (train/val), seed, plus the writer and photo defects
                     drawn from that seed (for tracing bad images; regenerating only needs the seed)
    run.json         config + git commit (+ template hashes) this ran with

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
from pocketform_distill.config import RunConfig, Section, SynthConfig
from pocketform_distill.provenance import write_run_info
from pocketform_distill.render import Writer
from pocketform_distill.values import sample_value, written_text


def list_fonts(fonts_dir: Path) -> list[Path]:
    fonts = sorted(Path(fonts_dir).glob("*.ttf"))  # sorted: glob order differs between machines
    if not fonts:
        raise FileNotFoundError(f"no .ttf handwriting fonts in {fonts_dir}")
    return fonts


def make_truth(section: Section, cfg: SynthConfig, rng: random.Random) -> dict[str, int | str | None]:
    return {f.id: None if rng.random() < cfg.empty_rate else sample_value(f, rng) for f in section.fields}


def _rounded(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_rounded(v) for v in x]
    return x


def generate_photo(section: Section, blank: Image.Image, cfg: SynthConfig, fonts: list[Path], seed: int
                   ) -> tuple[Image.Image, dict[str, int | str | None], dict[str, Any]]:
    """One synthetic section photo: sample values, handwrite them, then photograph it badly.

    Returns the image, the truth, and what was drawn at random (writer + defects) for the manifest.
    """
    rng = random.Random(seed)
    truth = make_truth(section, cfg, rng)
    writer = Writer.random(fonts, cfg.ink_colors, rng)
    ink = Image.new("L", blank.size, 0)
    for f in section.fields:
        if truth[f.id] is not None:
            writer.write(ink, f, written_text(f, truth[f.id]), rng)
    img, defects = apply_photo_defects(writer.apply(blank, ink), cfg.augment, rng)
    writer_info = asdict(writer) | {"font": Path(writer.font).name}
    return img, truth, _rounded({"writer": writer_info, "defects": defects})


def photo_seed(base_seed: int, section_id: str, n: int | str) -> int:
    """Seed for photo n of a section, from the base seed, the section id and n only.

    Hashing (instead of base + position * count + n) means adding, removing or reordering sections, or raising
    num_photos, never changes a photo that already exists, so its teacher outputs stay valid. A 64-bit number
    from sha256: different inputs effectively never collide.
    """
    digest = hashlib.sha256(f"{base_seed}:{section_id}:{n}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def val_indices(n: int, val_fraction: float, seed: int) -> set[int]:
    """Exactly round(n * val_fraction) photos go to val, picked by seed (not per photo, so the count is exact)."""
    return set(random.Random(seed).sample(range(n), round(n * val_fraction)))


def run_synth(config: RunConfig, data_root: Path = Path("data")) -> Path:
    spec = config.load_form()
    cfg = config.synth
    fonts = list_fonts(cfg.fonts_dir)

    out = data_root / config.run_name / "synth"
    if out.exists():
        shutil.rmtree(out)  # this stage owns the folder; stale photos from a bigger earlier run must not linger
    out.mkdir(parents=True)

    manifest = []
    for section in spec.sections:
        blank = section.open_template()
        val = val_indices(cfg.num_photos, cfg.val_fraction, photo_seed(cfg.seed, section.id, "val"))
        for i in range(cfg.num_photos):
            photo_id = f"{section.id}-{i:05d}"
            seed = photo_seed(cfg.seed, section.id, i)
            img, truth, drawn = generate_photo(section, blank, cfg, fonts, seed)
            img.save(out / f"{photo_id}.png")
            (out / f"{photo_id}.json").write_text(json.dumps(truth, indent=2) + "\n")
            split = "val" if i in val else "train"
            manifest.append({"id": photo_id, "section": section.id, "split": split, "seed": seed, **drawn})

    (out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    templates = {s.id: hashlib.sha256(s.template_path.read_bytes()).hexdigest() for s in spec.sections}
    write_run_info(out, "synth", config, form=spec.model_dump(mode="json"), fonts=[f.name for f in fonts],
                   template_sha256=templates)
    return out
