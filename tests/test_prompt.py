import json
from pathlib import Path
from typing import get_args

import pytest

from pocketform_distill.config import FormSpec, Section, load_form
from pocketform_distill.prompt import (
    BUILDERS,
    answer_layout,
    build_prompt,
    format_answer,
    parse_answer,
    prompt_info,
    rows,
)

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
    assert "Cases A" in prompts["s"] and "other_n" not in prompts["s"]
    assert "other_n" in prompts["t"] and "Cases A" not in prompts["t"]
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
        return {"id": fid, "label": label, "type": "integer", "cells": 3, "box": [x, y, 90, 40]}

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
    assert "Columns of each list (2 values), left to right: Male | Female" in text  # same columns: said once
    assert "- a: Cases A" in text and "- b: Cases B" in text
    assert "- total: Total" in text
    assert "never join digits from neighbouring groups" in text


def test_a_row_is_one_key_with_a_list():
    section = grid_spec().section("s")
    assert [(e.key, e.is_row) for e in answer_layout(section)] == [("a", True), ("b", True), ("total", False)]
    text = format_answer(section, {"a_m": 1, "b_f": 4, "total": 9})
    assert text == '{"a": [1, null], "b": [null, 4], "total": 9}'
    assert parse_answer(section, text) == {"a_m": 1, "a_f": None, "b_m": None, "b_f": 4, "total": 9}


def test_rows_with_different_columns_list_them_per_row():
    spec = grid_spec()
    spec.section("s").fields[3].label = "Cases B, Other"
    text = build_prompt(spec, spec.section("s"))
    assert "- a (2 values: Male | Female): Cases A" in text and "- b (2 values: Male | Other): Cases B" in text
    assert "Columns of each list" not in text


def test_row_without_a_shared_id_prefix_keeps_field_keys():
    section = grid_spec().section("s")
    section.groups[0].fields[1].id = "x_f"  # a_m, x_f: nothing in common
    keys = [e.key for e in answer_layout(section)]
    assert keys == ["a_m", "x_f", "b", "total"]


def test_two_rows_with_the_same_prefix_keep_field_keys():
    section = grid_spec().section("s")
    section.fields[0].id, section.fields[1].id = "a_y_m", "a_y_f"
    section.fields[2].id, section.fields[3].id = "a_y_n", "a_y_g"  # both rows would get the key "a_y"
    keys = [e.key for e in answer_layout(section)]
    assert keys == ["a_y_m", "a_y_f", "a_y_n", "a_y_g", "total"]


def test_parse_reads_short_lists_and_field_ids():
    section = grid_spec().section("s")
    got = parse_answer(section, '{"a": [5], "b_m": 2, "b": "oops", "total": null}')
    assert got == {"a_m": 5, "a_f": None, "b_m": 2, "b_f": None, "total": None}


def test_rows_never_mix_types():
    section = grid_spec().section("s")
    section.fields[1].type = "code"
    assert [[f.id for f in r] for r in rows(section.fields)][:2] == [["a_m"], ["a_f"]]


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


def test_mark_and_code_fields_get_their_hints_and_rules():
    spec = FormSpec.model_validate({"name": "mc", "sections": [{"id": "m", "title": "M", "template_image": "t.png",
        "groups": [{"id": "g", "fields": [
            {"id": "region", "label": "Region", "type": "mark", "options": ["North", "South"],
             "option_boxes": [[10, 10, 9, 9], [40, 10, 9, 9]]},
            {"id": "mfl", "label": "MFL code", "type": "code", "cells": 9, "box": [10, 60, 200, 30]},
        ]}]}]})
    section = spec.section()
    text = build_prompt(spec, section)
    assert "- region: Region [the option whose circle is marked, one of: North, South]" in text
    assert "- mfl: MFL code [code, 9 letters/digits, one per box]" in text
    assert "codes as strings" in text and "marked options as the option text" in text
    example = json.loads(text[text.index('{"region"'):].splitlines()[0])
    assert example == {"region": "North", "mfl": None}
    assert parse_answer(section, '{"region": "South", "mfl": "0012AB345"}') == {"region": "South",
                                                                               "mfl": "0012AB345"}


def test_parse_undoes_harmless_type_slips():
    spec = FormSpec.model_validate({"name": "n", "sections": [{"id": "m", "title": "M", "template_image": "t.png",
        "groups": [{"id": "g", "fields": [
            {"id": "day", "label": "Day", "type": "mark", "options": ["1", "2"],
             "option_boxes": [[10, 10, 9, 9], [40, 10, 9, 9]]},
            {"id": "code", "label": "Code", "type": "code", "cells": 4, "box": [10, 60, 90, 30]},
            {"id": "n", "label": "N", "type": "integer", "box": [10, 100, 90, 30]},
        ]}]}]})
    got = parse_answer(spec.section(), '{"day": 2, "code": 1234, "n": "7"}')
    assert got == {"day": "2", "code": "1234", "n": 7}
    assert parse_answer(spec.section(), '{"day": 3, "code": null, "n": "seven"}')["day"] == 3  # wrong stays wrong


def table_section(columns: list[tuple[str, str]]) -> Section:
    """One table row "cured" with the given (id suffix, column label) columns, side by side."""
    fields = [{"id": f"cured_{c}", "label": f"Cured, {label}", "type": "integer", "cells": 2,
               "box": [100 + 60 * i, 50, 50, 40]} for i, (c, label) in enumerate(columns)]
    spec = FormSpec.model_validate({"name": "t", "sections": [
        {"id": "d", "title": "D", "template_image": "t.png", "groups": [{"id": "g", "fields": fields}]}]})
    return spec.section()


AGES = [("6_11m", "6-11 m"), ("12_23m", "12-23 m"), ("24_59m", "24-59 m")]


def test_long_rows_split_where_the_id_words_change():
    cols = [(f"{p}_{s}_{a}", f"{p.upper()} {sl} {al}") for p in ("mam", "sam") for s, sl in (("m", "Male"),
            ("f", "Female")) for a, al in AGES]
    section = table_section(cols)  # 12 columns, like the nutrition form's section D
    layout = answer_layout(section)
    assert [e.key for e in layout] == ["cured_mam_m", "cured_mam_f", "cured_sam_m", "cured_sam_f"]
    assert all(e.is_row and len(e.fields) == 3 for e in layout)
    text = build_prompt(FormSpec.model_validate({"name": "t", "sections": [section.model_dump()]}), section)
    # Different columns per list: each names its own (the shorter shared header cost 12 points on the teacher)
    assert "- cured_mam_f (3 values: MAM Female 6-11 m | MAM Female 12-23 m | MAM Female 24-59 m): Cured" in text
    assert "Columns of each list" not in text
    truth = {f.id: i for i, f in enumerate(section.fields)}
    answer = format_answer(section, truth)
    assert answer.startswith('{"cured_mam_m": [0, 1, 2], "cured_mam_f": [3, 4, 5]')
    assert parse_answer(section, answer) == truth


def test_a_row_without_split_points_stays_one_list():
    section = table_section([("referred", "Referred"), ("completed", "Completed"), ("incomplete", "Incomplete"),
                             ("unknown", "Unknown")])
    assert [(e.key, len(e.fields)) for e in answer_layout(section)] == [("cured", 4)]


def test_short_rows_are_not_split():
    section = table_section([(f"m_{a}", f"Male {al}") for a, al in AGES])
    assert [e.key for e in answer_layout(section)] == ["cured_m"]
