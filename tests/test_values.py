import random
from datetime import date

from pocketform_distill.config import FormField
from pocketform_distill.values import sample_value, written_text

N = 2000


def field(**kw) -> FormField:
    return FormField(id="f", label="F", box=(0, 0, 10, 10), **kw)


def samples(f: FormField, seed: int = 0) -> list:
    rng = random.Random(seed)
    return [sample_value(f, rng) for _ in range(N)]


def test_integer_respects_min_max():
    vals = samples(field(type="integer", min=3, max=40))
    assert all(isinstance(v, int) and 3 <= v <= 40 for v in vals)
    assert {3, 40} <= set(vals)  # both ends are reachable


def test_integer_defaults_fit_digit_cells():
    vals = samples(field(type="integer", cells=2))
    assert all(0 <= v <= 99 for v in vals)


def test_integer_counts_skew_small():
    vals = samples(field(type="integer", min=0, max=999))
    one_digit = sum(v < 10 for v in vals) / N
    three_digit = sum(v >= 100 for v in vals) / N
    assert one_digit > 0.35  # uniform would give 1%
    assert 0.1 < three_digit < one_digit  # big values still appear, just less often


def test_date_in_range_and_iso():
    vals = samples(field(type="date", min=date(2025, 1, 1), max=date(2025, 3, 31)))
    parsed = [date.fromisoformat(v) for v in vals]
    assert all(date(2025, 1, 1) <= d <= date(2025, 3, 31) for d in parsed)


def test_date_is_written_in_field_format():
    f = field(type="date", format="%d/%m/%Y")
    assert written_text(f, "2025-03-07") == "07/03/2025"


def test_choice_only_from_options():
    vals = samples(field(type="choice", options=["a", "b", "c"]))
    assert set(vals) == {"a", "b", "c"}


def test_text_from_values_or_generated():
    assert set(samples(field(type="text", values=["X", "Y"]))) == {"X", "Y"}
    generated = samples(field(type="text"))
    assert all(isinstance(v, str) and v.strip() for v in generated)
    assert len(set(generated)) > 100


def test_same_seed_same_values():
    f = field(type="integer", max=500)
    assert samples(f, seed=7) == samples(f, seed=7)
    assert samples(f, seed=7) != samples(f, seed=8)
