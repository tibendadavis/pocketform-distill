"""Phone-photo defects applied to a clean synthetic form.

Each defect is one small function with its own probability in ``synth.augment`` of the run config. They run in
the order of ``STEPS`` (roughly the order things happen physically: light on paper -> camera pose -> lens ->
sensor -> file). The defaults were tuned by eye against a real phone photo of a filled form: warm pink cast,
low contrast (black ink reads as ~RGB 110,70,55), slight tilt and softness, and JPEG artefacts.
"""

from __future__ import annotations

import io
import math
import random
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from pocketform_distill.config import (
    AugmentConfig,
    BlurAug,
    BrightnessContrastAug,
    JpegAug,
    NoiseAug,
    PaperTintAug,
    PerspectiveAug,
    RotateAug,
    ShadingAug,
)


def _edge_fill(img: Image.Image) -> tuple[int, int, int]:
    """Colour for areas uncovered by rotation/perspective: a darker version of the page edge (the table)."""
    patch = np.asarray(img.crop((0, 0, 8, 8)), dtype=np.float32).reshape(-1, 3).mean(0)
    return tuple(int(c * 0.8) for c in patch)


def paper_tint(img: Image.Image, c: PaperTintAug, rng: random.Random) -> Image.Image:
    """Warm indoor light / auto white balance: white paper photographs as pink, cream or grey."""
    base = rng.choice(c.colors)
    tint = np.array([max(0, min(255, v + rng.randint(-c.jitter, c.jitter))) for v in base], np.float32) / 255
    arr = np.asarray(img, dtype=np.float32) * tint
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def shading(img: Image.Image, c: ShadingAug, rng: random.Random) -> Image.Image:
    """Uneven light: one side of the page is darker (lamp falloff, the phone's or hand's shadow)."""
    w, h = img.size
    angle = rng.uniform(0, 2 * math.pi)
    strength = rng.uniform(*c.strength)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xx / w - 0.5) * math.cos(angle) + (yy / h - 0.5) * math.sin(angle)  # -0.7..0.7 along the direction
    gain = 1 - strength * (t - t.min()) / (t.max() - t.min())
    arr = np.asarray(img, dtype=np.float32) * gain[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def rotate(img: Image.Image, c: RotateAug, rng: random.Random) -> Image.Image:
    """Page not square to the camera frame: small in-plane rotation."""
    return img.rotate(rng.uniform(-c.max_deg, c.max_deg), resample=Image.Resampling.BICUBIC,
                      fillcolor=_edge_fill(img))


def _perspective_coeffs(dst: list[tuple[float, float]], src: list[tuple[float, float]]) -> list[float]:
    """Solve the 8 numbers Pillow needs to map each output pixel (dst) back to an input pixel (src)."""
    rows, rhs = [], []
    for (x, y), (sx, sy) in zip(dst, src):
        rows.append([x, y, 1, 0, 0, 0, -sx * x, -sx * y])
        rows.append([0, 0, 0, x, y, 1, -sy * x, -sy * y])
        rhs += [sx, sy]
    return np.linalg.solve(np.array(rows, np.float64), np.array(rhs, np.float64)).tolist()


def perspective(img: Image.Image, c: PerspectiveAug, rng: random.Random) -> Image.Image:
    """Phone held at an angle, not parallel to the page: the form becomes a slight trapezoid."""
    w, h = img.size
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    dx, dy = c.max_shift * w, c.max_shift * h
    moved = [(x + rng.uniform(-dx, dx), y + rng.uniform(-dy, dy)) for x, y in corners]
    return img.transform(img.size, Image.Transform.PERSPECTIVE, _perspective_coeffs(moved, corners),
                         resample=Image.Resampling.BICUBIC, fillcolor=_edge_fill(img))


def blur(img: Image.Image, c: BlurAug, rng: random.Random) -> Image.Image:
    """Slight defocus or hand shake."""
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(*c.radius)))


def brightness_contrast(img: Image.Image, c: BrightnessContrastAug, rng: random.Random) -> Image.Image:
    """Auto-exposure: the photo comes out washed out (low contrast) or a bit dark/bright."""
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(*c.contrast))
    return ImageEnhance.Brightness(img).enhance(rng.uniform(*c.brightness))


def noise(img: Image.Image, c: NoiseAug, rng: random.Random) -> Image.Image:
    """Sensor noise from shooting in a dim room."""
    np_rng = np.random.default_rng(rng.getrandbits(32))
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + np_rng.normal(0, rng.uniform(*c.sigma), arr.shape).astype(np.float32)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def jpeg(img: Image.Image, c: JpegAug, rng: random.Random) -> Image.Image:
    """Lossy compression when the phone saves the photo or a messaging app resends it (blocky edges)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=rng.randint(*c.quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


STEPS: list[tuple[str, Callable]] = [
    ("paper_tint", paper_tint),
    ("shading", shading),
    ("rotate", rotate),
    ("perspective", perspective),
    ("blur", blur),
    ("brightness_contrast", brightness_contrast),
    ("noise", noise),
    ("jpeg", jpeg),
]


def apply_photo_defects(img: Image.Image, cfg: AugmentConfig, rng: random.Random) -> tuple[Image.Image, list[str]]:
    """Apply each defect with its own probability. Returns the image and the names of the defects applied."""
    applied = []
    for name, fn in STEPS:
        step_cfg = getattr(cfg, name)
        if rng.random() < step_cfg.p:
            img = fn(img, step_cfg, rng)
            applied.append(name)
    return img, applied
