"""Write handwriting into the field boxes of a blank template.

Handwriting is drawn into a separate greyscale "ink" layer (0 = no ink, 255 = full ink) and composited onto the
paper at the end. Keeping ink separate makes pen colour and pressure one step instead of a property of every glyph.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from pocketform_distill.config import RGB, FormField

CELL_GAP = 4  # px between cells in a comb box
MARK_STYLES = ("fill", "scribble", "tick", "cross")


@lru_cache(maxsize=64)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def cells(f: FormField) -> list[tuple[int, int, int, int]]:
    """Boxes the writer fills: one per character for comb fields (`cells`), else the whole field box."""
    x, y, w, h = f.box
    if not f.cells:
        return [f.box]
    cw = (w - CELL_GAP * (f.cells - 1)) / f.cells
    return [(round(x + i * (cw + CELL_GAP)), y, round(cw), h) for i in range(f.cells)]


# ---------------------------------------------------------------------------------------------------------------
# Handwriting
# ---------------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _digit_height_at_100(path: str) -> float:
    _, top, _, bottom = _font(path, 100).getbbox("8")
    return float(bottom - top)


def _font_for_height(path: str, px: float) -> ImageFont.FreeTypeFont:
    """Font size whose digits are `px` tall. Fonts differ a lot in how big "size 40" looks."""
    return _font(path, max(6, round(px * 100 / _digit_height_at_100(path))))


@dataclass(frozen=True)
class Writer:
    """One person filling one form: same pen and handwriting style for every field on it."""

    font: str
    ink: RGB
    size: float  # glyph height as a fraction of the cell height
    slant: float  # degrees; people lean consistently
    thick: bool  # heavier pen / more pressure
    jitter: float  # how untidy: scales random offsets and rotations
    mark: str  # how this person marks an option circle: fill, scribble, tick or cross

    @classmethod
    def random(cls, fonts: list[Path], ink_colors: list[RGB], rng: random.Random) -> Writer:
        base = rng.choice(ink_colors)
        ink = tuple(max(0, min(255, c + rng.randint(-10, 10))) for c in base)
        return cls(
            font=str(rng.choice(fonts)),
            ink=ink,
            size=rng.uniform(0.5, 0.75),
            slant=rng.uniform(-6, 6),
            thick=rng.random() < 0.35,
            jitter=rng.uniform(0.5, 1.5),
            mark=rng.choice(MARK_STYLES),
        )

    def _paste_glyphs(self, ink: Image.Image, text: str, font: ImageFont.FreeTypeFont, center: tuple[float, float],
                      angle: float, strength: int) -> None:
        left, top, right, bottom = font.getbbox(text, anchor="ls")
        pad = 6
        tile = Image.new("L", (right - left + 2 * pad, bottom - top + 2 * pad), 0)
        ImageDraw.Draw(tile).text((pad - left, pad - top), text, font=font, fill=strength, anchor="ls")
        if self.thick:
            tile = tile.filter(ImageFilter.MaxFilter(3))
        tile = tile.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        x = round(center[0] - tile.width / 2)
        y = round(center[1] - tile.height / 2)
        region = (x, y, x + tile.width, y + tile.height)
        ink.paste(ImageChops.lighter(ink.crop(region), tile), region)

    def write(self, ink: Image.Image, f: FormField, text: str, rng: random.Random) -> None:
        """Write `text` into field `f` on the ink layer, with random size, offset and rotation inside the box."""
        strength = rng.randint(200, 255)  # pen pressure for this field
        j = self.jitter
        if f.cells:
            # Comb boxes: one character per cell. Counts are right-aligned, as people fill tally boxes ("_ 2 1");
            # codes start at the first cell.
            boxes = cells(f)[-len(text):] if f.type == "integer" else cells(f)[:len(text)]
            for (cx, cy, cw, ch), digit in zip(boxes, text):
                font = _font_for_height(self.font, ch * self.size * rng.uniform(0.9, 1.1))
                center = (cx + cw / 2 + rng.gauss(0, cw * 0.06 * j), cy + ch / 2 + rng.gauss(0, ch * 0.05 * j))
                self._paste_glyphs(ink, digit, font, center, -self.slant + rng.gauss(0, 3 * j), strength)
            return
        x, y, w, h = f.box
        font = _font_for_height(self.font, h * self.size * rng.uniform(0.85, 1.05))
        while font.getlength(text) > w * 0.92 and font.size > 8:
            font = _font(self.font, font.size - 1)
        slack = max(0.0, w * 0.95 - font.getlength(text))
        cx = x + w * 0.03 + rng.uniform(0, slack) * rng.uniform(0.1, 0.8) + font.getlength(text) / 2
        cy = y + h / 2 + rng.gauss(0, h * 0.05 * j)
        self._paste_glyphs(ink, text, font, (cx, cy), -self.slant * 0.4 + rng.gauss(0, 1 * j), strength)

    def mark_option(self, ink: Image.Image, f: FormField, option: str, rng: random.Random) -> None:
        """Mark `option` of a mark field in its printed circle, in this writer's style."""
        x, y, w, h = f.option_boxes[f.options.index(option)]
        strength = rng.randint(200, 255)
        j = self.jitter
        r = min(w, h) / 2
        cx = x + w / 2 + rng.gauss(0, r * 0.12 * j)
        cy = y + h / 2 + rng.gauss(0, r * 0.12 * j)
        width = max(2, round(r * (0.3 if self.thick else 0.2)))
        draw = ImageDraw.Draw(ink)
        if self.mark == "fill":  # blacken the circle, a bit bigger or smaller than it
            s = r * rng.uniform(0.65, 1.0)
            draw.ellipse((cx - s, cy - s * rng.uniform(0.85, 1.1), cx + s, cy + s), fill=strength)
        elif self.mark == "scribble":  # quick zigzag strokes over the circle
            s = r * 0.8
            pts = [(cx - s + 2 * s * (i % 2) + rng.gauss(0, r * 0.1), cy - s + i * 2 * s / 6 + rng.gauss(0, r * 0.08))
                   for i in range(7)]
            draw.line(pts, fill=strength, width=width, joint="curve")
        elif self.mark == "tick":  # short down stroke, long up stroke; often overshoots the circle
            s = r * rng.uniform(0.9, 1.4)
            draw.line([(cx - s * 0.7, cy), (cx - s * 0.15, cy + s * 0.6), (cx + s * 0.9, cy - s * 0.9)],
                      fill=strength, width=width, joint="curve")
        else:  # cross
            s = r * rng.uniform(0.8, 1.3)
            for dx in (-1, 1):
                draw.line([(cx - dx * s, cy - s + rng.gauss(0, r * 0.1)), (cx + dx * s, cy + s)], fill=strength,
                          width=width)

    def apply(self, page: Image.Image, ink: Image.Image) -> Image.Image:
        """Composite the ink layer onto the page in this writer's pen colour."""
        return Image.composite(Image.new("RGB", page.size, self.ink), page, ink)
