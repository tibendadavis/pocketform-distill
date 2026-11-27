"""Phone-photo defects applied to a clean synthetic form.

Each defect is one small function with its own probability in ``synth.augment`` of the run config. They run in
the order of ``STEPS`` (roughly the order things happen physically: light on paper -> camera pose -> lens ->
sensor -> file). Each returns the image and the random parameters it drew, which synth records per form in
manifest.jsonl so a bad-looking image can be traced to its settings. The defaults were tuned by eye against a real phone photo of a filled form: warm pink cast,
low contrast (black ink reads as ~RGB 110,70,55), slight tilt and softness, and JPEG artefacts.
"""

from __future__ import annotations

import io
import math
import random
from collections.abc import Callable
from typing import Any

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

Out = tuple[Image.Image, dict[str, Any]]


def _edge_fill(img: Image.Image) -> tuple[int, int, int]:
    """Colour for areas uncovered by rotation/perspective: a darker version of the page edge (the table)."""
    patch = np.asarray(img.crop((0, 0, 8, 8)), dtype=np.float32).reshape(-1, 3).mean(0)
    return tuple(int(c * 0.8) for c in patch)


def paper_tint(img: Image.Image, c: PaperTintAug, rng: random.Random) -> Out:
    """Warm indoor light / auto white balance: white paper photographs as pink, cream or grey."""
    base = rng.choice(c.colors)
    color = [max(0, min(255, v + rng.randint(-c.jitter, c.jitter))) for v in base]
    arr = np.asarray(img, dtype=np.float32) * (np.array(color, np.float32) / 255)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8)), {"color": color}


def shading(img: Image.Image, c: ShadingAug, rng: random.Random) -> Out:
    """Uneven light: one side of the page is darker (lamp falloff, the phone's or hand's shadow)."""
    w, h = img.size
    angle = rng.uniform(0, 2 * math.pi)
    strength = rng.uniform(*c.strength)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xx / w - 0.5) * math.cos(angle) + (yy / h - 0.5) * math.sin(angle)  # -0.7..0.7 along the direction
    gain = 1 - strength * (t - t.min()) / (t.max() - t.min())
    arr = np.asarray(img, dtype=np.float32) * gain[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8)), {"direction_deg": math.degrees(angle),
                                                                "strength": strength}


def rotate(img: Image.Image, c: RotateAug, rng: random.Random) -> Out:
    """Page not square to the camera frame: small in-plane rotation."""
    deg = rng.uniform(-c.max_deg, c.max_deg)
    return img.rotate(deg, resample=Image.Resampling.BICUBIC, fillcolor=_edge_fill(img)), {"deg": deg}


def _perspective_coeffs(dst: list[tuple[float, float]], src: list[tuple[float, float]]) -> list[float]:
    """Solve the 8 numbers Pillow needs to map each output pixel (dst) back to an input pixel (src)."""
    rows, rhs = [], []
    for (x, y), (sx, sy) in zip(dst, src):
        rows.append([x, y, 1, 0, 0, 0, -sx * x, -sx * y])
        rows.append([0, 0, 0, x, y, 1, -sy * x, -sy * y])
        rhs += [sx, sy]
    return np.linalg.solve(np.array(rows, np.float64), np.array(rhs, np.float64)).tolist()


def perspective(img: Image.Image, c: PerspectiveAug, rng: random.Random) -> Out:
    """Phone held at an angle, not parallel to the page: the form becomes a slight trapezoid."""
    w, h = img.size
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    dx, dy = c.max_shift * w, c.max_shift * h
    moved = [(x + rng.uniform(-dx, dx), y + rng.uniform(-dy, dy)) for x, y in corners]
    out = img.transform(img.size, Image.Transform.PERSPECTIVE, _perspective_coeffs(moved, corners),
                        resample=Image.Resampling.BICUBIC, fillcolor=_edge_fill(img))
    return out, {"corners": [[x, y] for x, y in moved]}


def blur(img: Image.Image, c: BlurAug, rng: random.Random) -> Out:
    """Slight defocus or hand shake."""
    radius = rng.uniform(*c.radius)
    return img.filter(ImageFilter.GaussianBlur(radius)), {"radius": radius}


def brightness_contrast(img: Image.Image, c: BrightnessContrastAug, rng: random.Random) -> Out:
    """Auto-exposure: the photo comes out washed out (low contrast) or a bit dark/bright."""
    contrast, brightness = rng.uniform(*c.contrast), rng.uniform(*c.brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return ImageEnhance.Brightness(img).enhance(brightness), {"contrast": contrast, "brightness": brightness}


def noise(img: Image.Image, c: NoiseAug, rng: random.Random) -> Out:
    """Sensor noise from shooting in a dim room."""
    np_rng = np.random.default_rng(rng.getrandbits(32))
    arr = np.asarray(img, dtype=np.float32)
    sigma = rng.uniform(*c.sigma)
    arr = arr + np_rng.normal(0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8)), {"sigma": sigma}


def jpeg(img: Image.Image, c: JpegAug, rng: random.Random) -> Out:
    """Lossy compression when the phone saves the photo or a messaging app resends it (blocky edges)."""
    buf = io.BytesIO()
    quality = rng.randint(*c.quality)
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), {"quality": quality}


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


def apply_photo_defects(img: Image.Image, cfg: AugmentConfig, rng: random.Random) -> Out:
    """Apply each defect with its own probability. Returns the image and {defect: parameters} for those applied."""
    applied = {}
    for name, fn in STEPS:
        step_cfg = getattr(cfg, name)
        if rng.random() < step_cfg.p:
            img, applied[name] = fn(img, step_cfg, rng)
    return img, applied
