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
    # text     handwritten words on a line or in a box
    # integer  a count; with `cells`, one digit per printed cell, right-aligned
    # code     an identifier of letters and/or digits (e.g. a facility code), copied exactly; leading zeros kept
    # date     a date written in `format`; truth is ISO (YYYY-MM-DD)
    # choice   one of `options`, written as a word
    # mark     one of `options`, chosen by filling / ticking / crossing that option's printed circle or box
    type: Literal["text", "integer", "code", "date", "choice", "mark"]
    # The writing area in template pixels. For `mark` fields it can be left out: it is then the area around
    # all `option_boxes`.
    box: Box | None = None  # always set after validation
    # integer / code: number of printed cells ("comb" box), one character per cell.
    cells: int | None = Field(default=None, ge=1, le=16)
    # integer: optional range. date: min/max are dates (YAML `2024-01-01`), format is how the writer writes it.
    min: int | date | None = None
    max: int | date | None = None
    format: str = "%d/%m/%Y"
    # code: what synthetic codes look like, one character per position. A = letter, 9 = digit, * = letter or
    # digit, anything else is printed as is (e.g. "AA-9999"). Default: "*" for every cell.
    pattern: str | None = None
    options: list[str] | None = None  # choice, mark
    option_boxes: list[Box] | None = None  # mark: one box per option (same order), around its circle
    values: list[str] | None = None  # text, code: realistic values to sample from instead of generating

    @model_validator(mode="after")
    def _check_type_specific(self) -> FormField:
        t = self.type
        if t in ("choice", "mark") and not self.options:
            raise ValueError(f"{self.id}: {t} fields need `options`")
        if t not in ("choice", "mark") and self.options:
            raise ValueError(f"{self.id}: `options` only applies to choice and mark fields")
        if t == "mark":
            if not self.option_boxes or len(self.option_boxes) != len(self.options):
                raise ValueError(f"{self.id}: mark fields need one `option_boxes` entry per option "
                                 f"({len(self.options)})")
            if any(b[2] <= 0 or b[3] <= 0 for b in self.option_boxes):
                raise ValueError(f"{self.id}: option box width and height must be > 0")
            if self.box is None:
                x0 = min(b[0] for b in self.option_boxes)
                y0 = min(b[1] for b in self.option_boxes)
                x1 = max(b[0] + b[2] for b in self.option_boxes)
                y1 = max(b[1] + b[3] for b in self.option_boxes)
                self.box = (x0, y0, x1 - x0, y1 - y0)
        elif self.option_boxes:
            raise ValueError(f"{self.id}: `option_boxes` only applies to mark fields")
        if self.box is None:
            raise ValueError(f"{self.id}: `box` is required")
        if t not in ("text", "code") and self.values:
            raise ValueError(f"{self.id}: `values` only applies to text and code fields")
        if self.cells is not None and t not in ("integer", "code"):
            raise ValueError(f"{self.id}: `cells` only applies to integer and code fields")
        if self.pattern is not None:
            if t != "code":
                raise ValueError(f"{self.id}: `pattern` only applies to code fields")
            if self.cells and len(self.pattern) != self.cells:
                raise ValueError(f"{self.id}: pattern {self.pattern!r} has {len(self.pattern)} characters but "
                                 f"there are {self.cells} cells")
        if t == "code" and self.cells and self.values and any(len(v) > self.cells for v in self.values):
            raise ValueError(f"{self.id}: a value is longer than its {self.cells} cells")
        want = {"integer": int, "date": date}.get(t)
        for name in ("min", "max"):
            v = getattr(self, name)
            if v is None:
                continue
            if want is None or not isinstance(v, want) or (want is int and isinstance(v, bool)):
                raise ValueError(f"{self.id}: `{name}` must be {'a date' if want is date else 'an integer'}")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"{self.id}: min > max")
        if t == "integer" and self.cells and self.max is not None and len(str(self.max)) > self.cells:
            raise ValueError(f"{self.id}: max {self.max} does not fit in {self.cells} digit cells")
        if self.box[2] <= 0 or self.box[3] <= 0:
            raise ValueError(f"{self.id}: box width and height must be > 0")
        return self


class Group(Strict):
    """Related fields inside a section, under one printed heading (e.g. "MUAC", "Bilateral oedema")."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str | None = None
    fields: list[FormField]


class Section(Strict):
    """One part of a sectioned form: the area inside its three printed corner markers.

    A section is what the user photographs and what the pipeline processes as one unit: synth makes images of
    one section, the prompt asks for one section's fields, and the phone reads one section per photo. Smaller
    photos mean fewer image tokens and fewer fields to confuse.
    """

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str
    code: str | None = None  # printed section code, e.g. NUTR-B
    description: str = ""
    # Required: a flat, straight-on image of this section of the BLANK form, cropped to its corner markers (export
    # of the print file, or a scanner-app shot). Synth writes handwriting onto it, then adds photo defects.
    # Boxes are in its pixels.
    template_image: str
    groups: list[Group]

    folder: Path | None = Field(default=None, exclude=True)  # the form folder, set by load_form

    @property
    def fields(self) -> list[FormField]:
        return [f for g in self.groups for f in g.fields]

    @property
    def template_path(self) -> Path:
        return (self.folder or Path(".")) / self.template_image

    def open_template(self) -> Image.Image:
        """The blank template, upright (phone and scanner images often store rotation in EXIF), as RGB."""
        with Image.open(self.template_path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")


class FormSpec(Strict):
    """A whole paper form. Its `kind` says how it is photographed and laid out.

    Only "sectioned" exists today: several sections printed on one page, each marked by three corner squares,
    photographed and read one section at a time. A later kind (e.g. one photo of a whole single-layout form) gets
    its own spec shape and prompt builder.
    """

    name: str
    kind: Literal["sectioned"] = "sectioned"
    description: str = ""
    version: str | None = None
    sections: list[Section]
    dhis2: dict[str, Any] | None = None  # mapping used by the export stage later

    @property
    def fields(self) -> list[FormField]:
        """Every field on the form, all sections."""
        return [f for s in self.sections for f in s.fields]

    def section(self, section_id: str | None = None) -> Section:
        """A section by id. With no id, the only section (an error if the form has several)."""
        if section_id is None:
            if len(self.sections) != 1:
                raise ValueError(f"{self.name} has several sections; pick one of {[s.id for s in self.sections]}")
            return self.sections[0]
        for s in self.sections:
            if s.id == section_id:
                return s
        raise ValueError(f"no section {section_id!r} in {self.name}; sections: {[s.id for s in self.sections]}")

    @model_validator(mode="after")
    def _check_ids(self) -> FormSpec:
        section_ids = [s.id for s in self.sections]
        if not section_ids:
            raise ValueError("form has no sections")
        if len(set(section_ids)) != len(section_ids):
            raise ValueError(f"duplicate section ids: {section_ids}")
        # Field ids are unique across the whole form, so exported values never collide between sections.
        ids = [f.id for f in self.fields]
        if not all(s.fields for s in self.sections):
            raise ValueError("every section needs at least one field")
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate field ids: {dupes}")
        return self


def load_form(folder: str | Path) -> FormSpec:
    """Load and validate a form folder: form.yaml plus one template image per section, every box inside it."""
    folder = Path(folder)
    spec = FormSpec.model_validate(yaml.safe_load((folder / "form.yaml").read_text()))
    for section in spec.sections:
        section.folder = folder
        if not section.template_path.exists():
            raise FileNotFoundError(
                f"{section.template_path} not found. Every section needs a blank template image: scan or export "
                "the empty section, save it there, then get the boxes with scripts/pick_boxes.py."
            )
        w, h = section.open_template().size
        for f in section.fields:
            for x, y, bw, bh in [f.box, *(f.option_boxes or [])]:
                if x < 0 or y < 0 or x + bw > w or y + bh > h:
                    raise ValueError(f"{f.id}: box {[x, y, bw, bh]} is outside the {w}x{h} template of section "
                                     f"{section.id}")
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
    num_photos: int = Field(gt=0)  # synthetic photos per section
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
