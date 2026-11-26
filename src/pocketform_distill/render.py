"""Draw a blank form from its spec, and write handwriting into field boxes.

Handwriting is drawn into a separate greyscale "ink" layer (0 = no ink, 255 = full ink) and composited onto the
paper at the end. Keeping ink separate makes pen colour and pressure one step instead of a property of every glyph.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from pocketform_distill.config import RGB, FormField, FormSpec, Printed

PRINT_FONTS = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "print"
CELL_GAP = 4  # px between digit cells in a comb box


@lru_cache(maxsize=64)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _print_font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    return _font(str(PRINT_FONTS / ("Lato-Bold.ttf" if bold else "Lato-Regular.ttf")), size)


def cells(f: FormField) -> list[tuple[int, int, int, int]]:
    """Boxes the writer fills: one per digit for comb integer fields, else the whole field box."""
    x, y, w, h = f.box
    if not f.digits:
        return [f.box]
    cw = (w - CELL_GAP * (f.digits - 1)) / f.digits
    return [(round(x + i * (cw + CELL_GAP)), y, round(cw), h) for i in range(f.digits)]


# ---------------------------------------------------------------------------------------------------------------
# Blank template
# ---------------------------------------------------------------------------------------------------------------


def _draw_printed(d: ImageDraw.ImageDraw, item: Printed) -> None:
    if item.text is not None:
        d.text(item.at, item.text, font=_print_font(item.size, item.bold), fill=item.color, anchor=item.anchor)
    elif item.rect is not None:
        x, y, w, h = item.rect
        d.rectangle((x, y, x + w, y + h), fill=item.fill, outline=item.color if item.width else None,
                    width=item.width)
    elif item.line is not None:
        d.line(item.line, fill=item.color, width=item.width)


def draw_blank(spec: FormSpec) -> Image.Image:
    """Blank form from the spec: printed decoration, alignment markers, labels and field boxes."""
    img = Image.new("RGB", spec.page, spec.paper)
    d = ImageDraw.Draw(img)
    for item in spec.printed:
        _draw_printed(d, item)
    for x, y, w, h in spec.markers:
        # Ring + solid centre, like the fiducials printed on many survey forms.
        d.rectangle((x, y, x + w, y + h), outline=(20, 20, 20), width=max(2, w // 8))
        m = w // 4
        d.rectangle((x + m, y + m, x + w - m, y + h - m), fill=(20, 20, 20))
    for section in spec.sections:
        for item in section.printed:
            _draw_printed(d, item)
        for f in section.fields:
            x, y, w, h = f.box
            if f.print_label:
                d.text((x - 12, y + h // 2), f.label, font=_print_font(min(18, max(10, h // 2)), False),
                       fill=(60, 60, 60), anchor="rm")
            for cx, cy, cw, ch in cells(f):
                d.rectangle((cx, cy, cx + cw, cy + ch), outline=(70, 70, 70), width=1)
    return img


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
        if f.digits:
            # Comb boxes: one digit per cell, right-aligned, as people fill tally boxes ("_ 2 1").
            for (cx, cy, cw, ch), digit in zip(cells(f)[-len(text):], text):
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

    def apply(self, page: Image.Image, ink: Image.Image) -> Image.Image:
        """Composite the ink layer onto the page in this writer's pen colour."""
        return Image.composite(Image.new("RGB", page.size, self.ink), page, ink)
