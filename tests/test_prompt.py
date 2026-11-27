import json
from pathlib import Path
from typing import get_args

import pytest

from pocketform_distill.config import FormSpec, load_form
from pocketform_distill.prompt import BUILDERS, build_prompt, format_answer, parse_answer, prompt_info

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def spec() -> FormSpec:
    return load_form(ROOT / "forms/_template")


def test_every_form_kind_has_a_prompt():
    kinds = set(get_args(FormSpec.model_fields["kind"].annotation))
    assert kinds == set(BUILDERS), "add a builder in prompt.BUILDERS for each FormSpec kind"


def test_prompt_lists_every_field_in_order(spec):
    text = build_prompt(spec)
    positions = [text.index(f"- {f.id}: {f.label}") for f in spec.fields]
    assert positions == sorted(positions)


def test_prompt_is_stable_and_ignores_boxes(spec):
    # Re-picking boxes must not change the prompt, or teacher outputs would have to be regenerated.
    moved = spec.model_copy(deep=True)
    moved.sections[0].fields[0].box = (1, 1, 5, 5)
    assert build_prompt(spec) == build_prompt(spec) == build_prompt(moved)
    assert prompt_info(spec)["sha256"] == prompt_info(moved)["sha256"]


def test_prompt_hash_changes_when_wording_inputs_change(spec):
    relabelled = spec.model_copy(deep=True)
    relabelled.sections[0].fields[0].label = "Site name"
    assert prompt_info(spec)["sha256"] != prompt_info(relabelled)["sha256"]


def test_example_line_is_in_the_answer_format(spec):
    line = next(x for x in build_prompt(spec).splitlines() if x.startswith("Example"))
    example = json.loads(line[line.index("{"):])
    assert list(example) == [f.id for f in spec.fields[:3]]
    assert None in example.values()


def test_format_answer_is_one_line_in_field_order(spec):
    truth = {"shift": "evening", "site_name": "Hill Camp", "arrivals_count": 3}  # out of order, keys missing
    text = format_answer(spec, truth)
    assert "\n" not in text
    assert list(json.loads(text)) == [f.id for f in spec.fields]
    assert json.loads(text)["visit_date"] is None


def test_parse_round_trips_format(spec):
    truth = {"site_name": "Hill Camp", "visit_date": "2025-01-02", "shift": "morning",
             "arrivals_count": 0, "departures_count": None}
    assert parse_answer(spec, format_answer(spec, truth)) == truth


def test_parse_tolerates_fences_and_pretty_json(spec):
    reply = '```json\n{\n    "site_name": "River Post",\n    "extra": 1\n}\n```'
    got = parse_answer(spec, reply)
    assert got["site_name"] == "River Post"
    assert "extra" not in got
    assert got["shift"] is None  # missing keys become None


@pytest.mark.parametrize("reply", ["", "I cannot read this form.", '{"site_name": '])
def test_parse_rejects_non_json(spec, reply):
    with pytest.raises(ValueError):
        parse_answer(spec, reply)
