import json
from pathlib import Path
from typing import get_args

import pytest

from pocketform_distill.config import FormSpec, Section, load_form
from pocketform_distill.prompt import BUILDERS, build_prompt, format_answer, parse_answer, prompt_info, rows

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def spec() -> FormSpec:
    return load_form(ROOT / "forms/_template")


@pytest.fixture
def section(spec) -> Section:
    return spec.section("a")


def test_every_form_kind_has_a_prompt():
    kinds = set(get_args(FormSpec.model_fields["kind"].annotation))
    assert kinds == set(BUILDERS), "add a builder in prompt.BUILDERS for each FormSpec kind"


def test_prompt_names_the_section_and_lists_its_fields_in_order(spec, section):
    text = build_prompt(spec, section)
    assert text.startswith('This is a phone photo of section A of the paper form "_template"')
    positions = [text.index(f"- {f.id}: {f.label}") for f in section.fields]
    assert positions == sorted(positions)
    assert "## A  Visit" in text and "## B  Tally" in text  # groups become headings


def test_each_section_gets_its_own_prompt_with_only_its_fields():
    spec = grid_spec()
    prompts = {s.id: build_prompt(spec, s) for s in spec.sections}
    assert "a_m" in prompts["s"] and "other_n" not in prompts["s"]
    assert "other_n" in prompts["t"] and "a_m" not in prompts["t"]
    assert prompt_info(spec, spec.section("s"))["sha256"] != prompt_info(spec, spec.section("t"))["sha256"]


def test_prompt_is_stable_when_boxes_are_re_picked(spec, section):
    # Re-picking boxes must not change the prompt, or teacher outputs would have to be regenerated.
    # Only which row a field is in matters, not its exact coordinates.
    moved = spec.model_copy(deep=True)
    field = moved.section("a").fields[0]
    x, y, w, h = field.box
    field.box = (x + 7, y - 4, w - 10, h + 3)
    assert build_prompt(spec, section) == build_prompt(spec, section) == build_prompt(moved, moved.section("a"))
    assert prompt_info(spec, section)["sha256"] == prompt_info(moved, moved.section("a"))["sha256"]


def grid_spec() -> FormSpec:
    """Section s: a 2x2 tally grid (labels "<row>, <column>") plus one field on its own row. Section t: one field."""
    def f(fid, label, x, y):
        return {"id": fid, "label": label, "type": "integer", "digits": 3, "box": [x, y, 90, 40]}

    return FormSpec.model_validate({"name": "grid", "sections": [
        {"id": "s", "title": "Tally", "template_image": "template-s.png", "groups": [
            {"id": "cases", "title": "Cases", "fields": [
                f("a_m", "Cases A, Male", 300, 100), f("a_f", "Cases A, Female", 410, 102),
                f("b_m", "Cases B, Male", 300, 160), f("b_f", "Cases B, Female", 410, 158),
                f("total", "Total", 300, 220),
            ]}]},
        {"id": "t", "title": "Other", "template_image": "template-t.png", "groups": [
            {"id": "other", "fields": [f("other_n", "Other", 300, 100)]}]},
    ]})


def test_rows_group_side_by_side_fields():
    fields = grid_spec().section("s").fields
    assert [[f.id for f in r] for r in rows(fields)] == [["a_m", "a_f"], ["b_m", "b_f"], ["total"]]


def test_rows_need_left_to_right_order():
    fields = grid_spec().section("s").fields
    a_m, a_f = fields[:2]
    a_m.box, a_f.box = a_f.box, a_m.box  # form.yaml order is now right to left: don't claim "left to right"
    assert [f.id for f in rows(fields)[0]] == ["a_m"]


def test_grid_prompt_lists_rows_and_warns_about_neighbouring_groups():
    spec = grid_spec()
    text = build_prompt(spec, spec.section("s"))
    assert "- row Cases A: a_m (Male) | a_f (Female)" in text
    assert "- total: Total" in text
    assert "never join digits from neighbouring groups" in text


def test_no_group_warning_without_side_by_side_combs(spec, section):
    assert "neighbouring groups" not in build_prompt(spec, section)  # template: each comb field on its own row


def test_prompt_hash_changes_when_wording_inputs_change(spec, section):
    relabelled = spec.model_copy(deep=True)
    relabelled.section("a").fields[0].label = "Site name"
    assert prompt_info(spec, section)["sha256"] != prompt_info(relabelled, relabelled.section("a"))["sha256"]


def test_example_line_is_in_the_answer_format(spec, section):
    line = next(x for x in build_prompt(spec, section).splitlines() if x.startswith("Example"))
    example = json.loads(line[line.index("{"):])
    assert list(example) == [f.id for f in section.fields[:3]]
    assert None in example.values()


def test_format_answer_is_one_line_in_field_order(section):
    truth = {"shift": "evening", "site_name": "Hill Camp", "arrivals_count": 3}  # out of order, keys missing
    text = format_answer(section, truth)
    assert "\n" not in text
    assert list(json.loads(text)) == [f.id for f in section.fields]
    assert json.loads(text)["visit_date"] is None


def test_parse_round_trips_format(section):
    truth = {"site_name": "Hill Camp", "visit_date": "2025-01-02", "shift": "morning",
             "arrivals_count": 0, "departures_count": None}
    assert parse_answer(section, format_answer(section, truth)) == truth


def test_parse_tolerates_fences_and_pretty_json(section):
    reply = '```json\n{\n    "site_name": "River Post",\n    "extra": 1\n}\n```'
    got = parse_answer(section, reply)
    assert got["site_name"] == "River Post"
    assert "extra" not in got
    assert got["shift"] is None  # missing keys become None


@pytest.mark.parametrize("reply", ["", "I cannot read this form.", '{"site_name": '])
def test_parse_rejects_non_json(section, reply):
    with pytest.raises(ValueError):
        parse_answer(section, reply)
