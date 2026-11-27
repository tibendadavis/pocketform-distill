"""Typed run config and form spec.

Pydantic validates the YAML once at load time, so later stages get typed objects instead of raw dicts and a
misspelt key fails loudly (``extra="forbid"``) instead of silently using a default.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------------------------------------------
# Form spec (forms/<name>/form.yaml)
# ---------------------------------------------------------------------------------------------------------------

Box = tuple[int, int, int, int]  # x, y, width, height in pixels of the template image
RGB = tuple[int, int, int]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FormField(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str
    type: Literal["text", "integer", "date", "choice"]
    box: Box
    # integer: optional range; digits = number of printed digit cells ("comb" boxes), one digit per cell.
    # date: min/max are dates (YAML `2024-01-01`), format is how the writer writes it (truth is always ISO).
    min: int | date | None = None
    max: int | date | None = None
    digits: int | None = Field(default=None, ge=1, le=12)
    format: str = "%d/%m/%Y"
    options: list[str] | None = None  # choice
    values: list[str] | None = None  # text: realistic values to sample from

    @model_validator(mode="after")
    def _check_type_specific(self) -> FormField:
        if self.type == "choice" and not self.options:
            raise ValueError(f"{self.id}: choice fields need `options`")
        if self.type != "choice" and self.options:
            raise ValueError(f"{self.id}: `options` only applies to choice fields")
        if self.type != "text" and self.values:
            raise ValueError(f"{self.id}: `values` only applies to text fields")
        if self.digits is not None and self.type != "integer":
            raise ValueError(f"{self.id}: `digits` only applies to integer fields")
        want = {"integer": int, "date": date}.get(self.type)
        for name in ("min", "max"):
            v = getattr(self, name)
            if v is None:
                continue
            if want is None or not isinstance(v, want) or (want is int and isinstance(v, bool)):
                raise ValueError(f"{self.id}: `{name}` must be {'a date' if want is date else 'an integer'}")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"{self.id}: min > max")
        if self.type == "integer" and self.digits and self.max is not None and len(str(self.max)) > self.digits:
            raise ValueError(f"{self.id}: max {self.max} does not fit in {self.digits} digit cells")
        if self.box[2] <= 0 or self.box[3] <= 0:
            raise ValueError(f"{self.id}: box width and height must be > 0")
        return self


class Section(Strict):
    """A titled block of fields. Most paper forms are a stack of these (A. Visit, B. Screening, ...)."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str | None = None
    fields: list[FormField]


class FormSpec(Strict):
    name: str
    # Layout family. Only "sectioned" exists today; other families (e.g. a register with repeated rows) can be
    # added later without changing what downstream stages see: they only use `fields`.
    kind: Literal["sectioned"] = "sectioned"
    description: str = ""
    version: str | None = None
    # Required: a flat, straight-on image of the BLANK form next to form.yaml (scan or export of the print file).
    # Synth writes handwriting onto it and then adds the photo defects itself. Boxes are in its pixels.
    template_image: str
    sections: list[Section]
    dhis2: dict[str, Any] | None = None  # mapping used by the export stage later

    folder: Path | None = Field(default=None, exclude=True)  # set by load_form, not read from YAML

    @property
    def fields(self) -> list[FormField]:
        return [f for s in self.sections for f in s.fields]

    @model_validator(mode="after")
    def _check_ids(self) -> FormSpec:
        ids = [f.id for f in self.fields]
        if not ids:
            raise ValueError("form has no fields")
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate field ids: {dupes}")
        return self

    @property
    def template_path(self) -> Path:
        return (self.folder or Path(".")) / self.template_image

    def open_template(self) -> Image.Image:
        """The blank template, upright (phone and scanner images often store rotation in EXIF), as RGB."""
        with Image.open(self.template_path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")


def load_form(folder: str | Path) -> FormSpec:
    """Load and validate a form folder: form.yaml plus its template image, with every box inside the image."""
    folder = Path(folder)
    spec = FormSpec.model_validate(yaml.safe_load((folder / "form.yaml").read_text()))
    spec.folder = folder
    if not spec.template_path.exists():
        raise FileNotFoundError(
            f"{spec.template_path} not found. Every form needs a blank template image: scan or export the empty "
            "form, save it there, then get the boxes with scripts/pick_boxes.py."
        )
    w, h = spec.open_template().size
    for f in spec.fields:
        x, y, bw, bh = f.box
        if x < 0 or y < 0 or x + bw > w or y + bh > h:
            raise ValueError(f"{f.id}: box {list(f.box)} is outside the {w}x{h} template image")
    return spec


# ---------------------------------------------------------------------------------------------------------------
# Run config (configs/<run>.yaml)
# ---------------------------------------------------------------------------------------------------------------


class TeacherConfig(Strict):
    backend: Literal["ollama"] = "ollama"
    model: str
    top_logprobs: int = Field(default=20, ge=0, le=20)
    think: bool = False


class ImageConfig(Strict):
    max_pixels: int = Field(gt=0)


class StudentConfig(Strict):
    model: str
    method: Literal["lora", "full"] = "lora"


class Aug(Strict):
    p: float = Field(ge=0, le=1)  # probability this defect is applied to a given form


class PaperTintAug(Aug):
    colors: list[RGB] = [(232, 182, 156), (240, 205, 180), (236, 226, 210), (226, 226, 222)]
    jitter: int = 8


class ShadingAug(Aug):
    strength: tuple[float, float] = (0.05, 0.25)


class RotateAug(Aug):
    max_deg: float = 3.0


class PerspectiveAug(Aug):
    max_shift: float = Field(default=0.04, ge=0, le=0.2)  # corner shift as a fraction of the image size


class BlurAug(Aug):
    radius: tuple[float, float] = (0.3, 1.2)


class BrightnessContrastAug(Aug):
    brightness: tuple[float, float] = (0.85, 1.1)
    contrast: tuple[float, float] = (0.5, 1.0)


class NoiseAug(Aug):
    sigma: tuple[float, float] = (1.0, 6.0)


class JpegAug(Aug):
    quality: tuple[int, int] = (50, 90)


class AugmentConfig(Strict):
    paper_tint: PaperTintAug = PaperTintAug(p=0.9)
    shading: ShadingAug = ShadingAug(p=0.7)
    rotate: RotateAug = RotateAug(p=0.8)
    perspective: PerspectiveAug = PerspectiveAug(p=0.7)
    blur: BlurAug = BlurAug(p=0.6)
    brightness_contrast: BrightnessContrastAug = BrightnessContrastAug(p=0.8)
    noise: NoiseAug = NoiseAug(p=0.5)
    jpeg: JpegAug = JpegAug(p=1.0)


class SynthConfig(Strict):
    num_forms: int = Field(gt=0)
    seed: int = 0
    empty_rate: float = Field(default=0.1, ge=0, le=1)  # chance each field is left blank (truth = null)
    val_fraction: float = Field(default=0.1, ge=0, lt=1)
    fonts_dir: Path = Path("assets/fonts/handwriting")
    ink_colors: list[RGB] = [(20, 20, 28), (28, 40, 110), (55, 55, 60)]  # ballpoint black, blue, grey
    augment: AugmentConfig = AugmentConfig()


class TrainConfig(Strict):
    soft_label_weight: float = Field(ge=0, le=1)


class TargetsConfig(Strict):
    field_accuracy: float = Field(ge=0, le=1)


class ExportConfig(Strict):
    format: Literal["json"] = "json"


class RunConfig(Strict):
    run_name: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    form: Path
    teacher: TeacherConfig
    image: ImageConfig
    student: StudentConfig
    synth: SynthConfig
    train: TrainConfig
    targets: TargetsConfig
    export: ExportConfig

    def load_form(self) -> FormSpec:
        return load_form(self.form)


def load_run_config(path: str | Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(Path(path).read_text()))
