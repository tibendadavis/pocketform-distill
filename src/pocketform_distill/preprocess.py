"""The one image preprocessing, used by every stage that shows an image to a model (teacher, train, eval, phone).

Distillation compares teacher and student on the same image, so they must get the same pixels: an image is turned
upright (phone photos often store rotation in EXIF), converted to RGB, shrunk only if it has more than
`image.max_pixels`, and padded so both sides are a multiple of 32. Why 32: the student (Qwen3.5, Qwen2-VL image processor) cuts images into 16 px patches
and merges 2 x 2 of them into one image token, and resizes anything that is not a multiple of 32 itself. Already
being one means its processor leaves the image alone, and the phone can reproduce it exactly.

Fewer pixels = fewer image tokens (~1 per 1024 px) = faster on the phone, but small handwriting gets harder to read:
`max_pixels` is that trade-off, and it must not change between teacher, training, eval and export.
"""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

from PIL import Image, ImageOps

FACTOR = 32  # patch size 16 x merge size 2, from the student's preprocessor_config.json
MIN_PIXELS = 65536  # the student processor's smallest image (size.shortest_edge in preprocessor_config.json)


def target_size(width: int, height: int, max_pixels: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(scaled size, padded size) for an image: scaled keeps the shape and is shrunk only if needed; padded rounds
    both sides UP to a multiple of FACTOR, and the two fit in max_pixels together.

    Padding instead of resizing to reach a multiple of 32 keeps the pixels as they are: a 1234 x 516 section photo
    would otherwise be resampled to 1216 x 512, which slightly blurs 25 px digit cells (teacher accuracy on the
    nutrition form's section D dropped from ~90% to 85% with that resample).
    """
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    if width * height * scale * scale < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / (width * height))  # tiny image: up to the student's minimum
    while True:
        w, h = max(1, round(width * scale)), max(1, round(height * scale))
        pw, ph = math.ceil(w / FACTOR) * FACTOR, math.ceil(h / FACTOR) * FACTOR
        if pw * ph <= max(max_pixels, MIN_PIXELS) or scale < 0.01:
            return (w, h), (pw, ph)
        scale *= 0.99


def prepare_image(image: Image.Image | str | Path, max_pixels: int) -> Image.Image:
    """Upright RGB image at the size every model sees: shrunk to fit max_pixels if needed, then padded right and
    bottom to multiples of 32 with the image's own edge colour (a plain strip, no new detail)."""
    if not isinstance(image, Image.Image):
        with Image.open(image) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
    else:
        image = ImageOps.exif_transpose(image).convert("RGB")
    scaled, padded = target_size(*image.size, max_pixels)
    if scaled != image.size:
        image = image.resize(scaled, Image.Resampling.LANCZOS)
    if padded == image.size:
        return image
    edge = image.crop((image.width - 1, 0, image.width, image.height)).resize((1, 1), Image.Resampling.BOX)
    out = Image.new("RGB", padded, edge.getpixel((0, 0)))
    out.paste(image, (0, 0))
    return out


def png_bytes(image: Image.Image) -> bytes:
    """Lossless, metadata-free PNG: the same image always gives the same bytes (and the same hash)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def image_sha256(image: Image.Image) -> str:
    return hashlib.sha256(png_bytes(image)).hexdigest()
