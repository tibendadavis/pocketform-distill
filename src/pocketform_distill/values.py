"""Random but valid field values for synthetic forms.

Every function takes an explicit ``random.Random`` so the caller controls the seed: no global randomness.
"""

from __future__ import annotations

import math
import random
import string
from datetime import date, timedelta

from pocketform_distill.config import FormField

DEFAULT_INT_MAX = 999
DEFAULT_DATE_MIN = date(2020, 1, 1)
DEFAULT_DATE_MAX = date(2026, 12, 31)

_SYLLABLES = ["ka", "ki", "mu", "na", "ba", "lo", "re", "sa", "to", "wa", "ni", "ga", "da", "me", "ru", "po"]


def int_range(f: FormField) -> tuple[int, int]:
    lo = f.min if isinstance(f.min, int) else 0
    hi = f.max if isinstance(f.max, int) else (10**f.cells - 1 if f.cells else DEFAULT_INT_MAX)
    return lo, hi


INT_SKEW = 1.5  # 1 = log-uniform; higher = more small numbers


def sample_int(f: FormField, rng: random.Random) -> int:
    """Skewed towards small counts, but every digit length still appears.

    Log-uniform (INT_SKEW = 1) gives each order of magnitude (0-9, 10-99, 100-999) equal probability. Raising the
    uniform draw to INT_SKEW = 1.5 tilts that further: a 0-999 field gives ~48% one-digit, ~28% two-digit and ~24%
    three-digit values (uniform sampling would give 1% / 9% / 90%). Tally sheets are mostly small counts.
    """
    lo, hi = int_range(f)
    u = math.log(hi - lo + 2) * rng.random() ** INT_SKEW
    return min(hi, lo + int(math.exp(u)) - 1)


def sample_date(f: FormField, rng: random.Random) -> date:
    lo = f.min if isinstance(f.min, date) else DEFAULT_DATE_MIN
    hi = f.max if isinstance(f.max, date) else DEFAULT_DATE_MAX
    return lo + timedelta(days=rng.randint(0, (hi - lo).days))


def sample_text(f: FormField, rng: random.Random) -> str:
    if f.values:
        return rng.choice(f.values)
    words = ["".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 3))).capitalize()
             for _ in range(rng.randint(1, 2))]
    return " ".join(words)


# No I or O: handwritten, they look exactly like 1 and 0, so no reader (teacher, student or person) could tell them
# apart and the truth would teach guessing. Many real code schemes leave them out for the same reason. If a form's
# codes do use them, list real codes in `values`.
CODE_CHARS = {"A": "".join(c for c in string.ascii_uppercase if c not in "IO"), "9": string.digits}


def sample_code(f: FormField, rng: random.Random) -> str:
    """A code from `values`, or built from `pattern` (A = letter, 9 = digit, * = either, else literal).

    Without a pattern every cell gets a letter or digit (50/50), so all cells are filled, as codes are.
    """
    if f.values:
        return rng.choice(f.values)
    pattern = f.pattern or "*" * (f.cells or rng.randint(4, 8))
    out = []
    for p in pattern:
        if p == "*":
            p = rng.choice("A9")
        out.append(rng.choice(CODE_CHARS[p]) if p in CODE_CHARS else p)
    return "".join(out)


def sample_value(f: FormField, rng: random.Random) -> int | str:
    """The truth value as it is stored in {id}.json (dates as ISO strings, codes and options as strings)."""
    if f.type == "integer":
        return sample_int(f, rng)
    if f.type == "date":
        return sample_date(f, rng).isoformat()
    if f.type in ("choice", "mark"):
        return rng.choice(f.options)
    if f.type == "code":
        return sample_code(f, rng)
    return sample_text(f, rng)


def written_text(f: FormField, value: int | str) -> str:
    """What the writer puts on paper for a truth value."""
    if f.type == "date":
        return date.fromisoformat(value).strftime(f.format)
    return str(value)
