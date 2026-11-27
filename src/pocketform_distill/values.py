"""Random but valid field values for synthetic forms.

Every function takes an explicit ``random.Random`` so the caller controls the seed: no global randomness.
"""

from __future__ import annotations

import math
import random
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


def sample_value(f: FormField, rng: random.Random) -> int | str:
    """The truth value as it is stored in {id}.json (dates as ISO strings)."""
    if f.type == "integer":
        return sample_int(f, rng)
    if f.type == "date":
        return sample_date(f, rng).isoformat()
    if f.type == "choice":
        return rng.choice(f.options)
    return sample_text(f, rng)


def written_text(f: FormField, value: int | str) -> str:
    """What the writer puts on paper for a truth value."""
    if f.type == "date":
        return date.fromisoformat(value).strftime(f.format)
    return str(value)
