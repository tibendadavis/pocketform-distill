"""The one prompt, and the answer format that goes with it.

Every stage that talks to a model (teacher, dataset, train, eval, export, the phone) gets its prompt from
``build_prompt(spec, section)``: one prompt per photo, and for a sectioned form a photo is one section. Distillation only works if they all see the same text: the teacher's probabilities are
"what comes next given this exact prompt and image", and the student is trained to match them under the same prompt.

One builder per form ``kind``. The builder turns a form spec and one section into the prompt text, so every
section of every form of that kind gets the same wording with its own field list. A new kind = a new ``kind`` value in config.FormSpec plus a builder registered in
``BUILDERS`` (a test fails if a kind has no builder).

The answer format is defined here too because it is part of the same contract: ``format_answer`` produces the exact
target text the student learns (and the teacher is asked for), ``parse_answer`` reads a model's reply back.

Answer shape: one-line JSON. A field on its own is ``"field_id": value``. Side-by-side fields of one type (a table
row) are answered as short lists, ``"cured_mam_m": [3, null, 12]``, values left to right. A list's key is the part
of its field ids they all start with, so name grid fields ``<row>_<column>`` (``cured_mam_m_6_11m``). A row of more
than MAX_LIST fields is split where the next word of its ids changes (``cured_mam_m_*`` | ``cured_mam_f_*`` | ...).
Why: each field id costs ~8 tokens and the phone generates every one, but one list per long row made qwen3.5:9b lose
its place (a skipped empty column shifts every later value). Measured on 6 synthetic photos of the nutrition form's
section D (6 rows x 12 columns): one key per field 83% of fields right, ~1360 answer tokens; one 12-value list per
row 78%, ~255 tokens; lists of <= 3: 90%, ~400 tokens. If a row's fields share no prefix, or two lists would get the
same key, those fields keep their own keys.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pocketform_distill.config import FormField, FormSpec, Group, Section


def _date_pattern(fmt: str) -> str:
    """strftime format -> what a person sees, e.g. %d/%m/%Y -> DD/MM/YYYY."""
    return fmt.replace("%d", "DD").replace("%m", "MM").replace("%Y", "YYYY").replace("%y", "YY")


def _describe(f: FormField) -> str:
    if f.type == "integer":
        return f"whole number, up to {f.cells} digits, one digit per box" if f.cells else "whole number"
    if f.type == "code":
        return f"code, {f.cells} letters/digits, one per box" if f.cells else "code of letters/digits"
    if f.type == "date":
        return f"date, written {_date_pattern(f.format)}"
    if f.type == "choice":
        return "one of: " + ", ".join(f.options)
    if f.type == "mark":
        return "the option whose circle is marked, one of: " + ", ".join(f.options)
    return "handwritten text"


_EXAMPLE_INTS = (12, None, 7, 3, 25, 0)


def _example_value(f: FormField, i: int) -> Any:
    """A made-up but valid value for the example line. Fixed, not random: the prompt must never change."""
    if i == 1:
        return None  # show what an empty field looks like
    if f.type == "integer":
        return _EXAMPLE_INTS[i % len(_EXAMPLE_INTS)]
    if f.type == "date":
        return "2025-01-31"
    if f.type in ("choice", "mark"):
        return f.options[0]
    if f.type == "code":
        if f.values:
            return f.values[0]
        pattern = f.pattern or "*" * (f.cells or 6)
        return "".join({"A": "K", "9": "4", "*": "K4"[k % 2]}.get(c, c) for k, c in enumerate(pattern))
    return f.values[0] if f.values else "Example"


def rows(fields: list[FormField]) -> list[list[FormField]]:
    """Group consecutive fields of one type that sit side by side (boxes overlap vertically, left to right).

    Only the grouping reaches the prompt and the answer, not coordinates, so re-picking boxes doesn't change them
    unless a field moves to another row.
    """
    out: list[list[FormField]] = []
    for f in fields:
        if out:
            prev = out[-1][-1]
            overlap = min(prev.box[1] + prev.box[3], f.box[1] + f.box[3]) - max(prev.box[1], f.box[1])
            if (overlap > min(prev.box[3], f.box[3]) / 2 and f.box[0] > prev.box[0]
                    and f.type == prev.type and f.type != "mark"):
                out[-1].append(f)
                continue
        out.append([f])
    return out


def _common_prefix(ids: list[str]) -> str:
    """Longest run of leading "_"-separated words that every id starts with ("" if none)."""
    words = [i.split("_") for i in ids]
    common = []
    for parts in zip(*words):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    # Keep at least one word that tells the fields apart, so a prefix is never a whole id.
    if any(len(w) == len(common) for w in words):
        common = common[:-1]
    return "_".join(common)


MAX_LIST = 3  # longest list in an answer before a row is split (see the module docstring for the measurements)


def _split_row(row: list[FormField]) -> list[list[FormField]]:
    """Split a long row where the next word of its field ids changes, until every part has <= MAX_LIST fields.

    A row that has no such split point stays whole (its fields' next words are all different, e.g. ref_mam_referred,
    ref_mam_completed, ...): splitting it would only give single fields.
    """
    if len(row) <= MAX_LIST:
        return [row]
    depth = len(_common_prefix([f.id for f in row]).split("_"))
    runs: list[list[FormField]] = []
    for f in row:
        if runs and runs[-1][-1].id.split("_")[:depth + 1] == f.id.split("_")[:depth + 1]:
            runs[-1].append(f)
        else:
            runs.append([f])
    if len(runs) == 1 or all(len(r) == 1 for r in runs):
        return [row]
    return [part for r in runs for part in _split_row(r)]


@dataclass(frozen=True)
class Entry:
    """One key of the answer: a single field, or a row of side-by-side fields answered as one list."""

    key: str
    fields: list[FormField]
    is_row: bool
    group: str


def answer_layout(section: Section) -> list[Entry]:
    """The keys of a section's answer, in order (see the module docstring for the rule)."""
    candidates: list[tuple[str, list[FormField]]] = []
    for group in section.groups:
        for row in rows(group.fields):
            candidates += [(group.id, part) for part in _split_row(row)]
    keys = [_common_prefix([f.id for f in row]) if len(row) > 1 else "" for _, row in candidates]
    field_ids = {f.id for f in section.fields}
    out: list[Entry] = []
    for (group, row), key in zip(candidates, keys):
        if key and keys.count(key) == 1 and key not in field_ids:
            out.append(Entry(key, row, True, group))
        else:
            out += [Entry(f.id, [f], False, group) for f in row]
    return out


def _example(section: Section) -> str:
    """One line in the exact answer format. Without it the teacher answers in fenced, pretty-printed JSON, whose
    tokens don't line up with the one-line target the student learns (checked on qwen3.5:9b). Shows the first
    entries: three single fields, or two when a row (a list) is among them."""
    layout = answer_layout(section)
    shown = layout[:2] if any(e.is_row for e in layout[:3]) else layout[:3]
    out: dict[str, Any] = {}
    for i, e in enumerate(shown):
        if e.is_row:
            out[e.key] = [_example_value(f, i + k) for k, f in enumerate(e.fields)]
        else:
            out[e.key] = _example_value(e.fields[0], i)
    return json.dumps(out, ensure_ascii=False)


def _split_label(row: list[FormField]) -> tuple[str | None, list[str]]:
    """Labels written "<row>, <column>" with one shared row part -> (row, columns); else (None, labels)."""
    split = [f.label.rsplit(", ", 1) for f in row]
    if all(len(s) == 2 for s in split) and len({s[0] for s in split}) == 1:
        return split[0][0], [s[1] for s in split]
    return None, [f.label for f in row]


def _list_names(e: Entry) -> tuple[str, list[str]]:
    """A list's name and its full column names ("Cured", [MAM Male 6-11 m, ...]).

    Each list keeps its full column names even when they repeat across lists. Measured on qwen3.5:9b, section D:
    moving the shared words into the name and listing "6-11 m | 12-23 m | 24-59 m" once saved ~850 prompt tokens
    but dropped fields right from 90% to 78%.
    """
    name, cols = _split_label(e.fields)
    return name or "row", cols


def _describe_all(fields: list[FormField]) -> str:
    """One hint for a list: its fields' shared hint, or e.g. "whole numbers, up to 2/2/4 digits, ..." for a date
    written in DD / MM / YYYY cells."""
    hints = {_describe(f) for f in fields}
    if len(hints) == 1:
        return hints.pop()
    if all(f.type == "integer" and f.cells for f in fields):
        return f"whole numbers, up to {'/'.join(str(f.cells) for f in fields)} digits, one digit per box"
    return "; ".join(_describe(f) for f in fields)


def _group_lines(group: Group, entries: list[Entry]) -> list[str]:
    hints = {_describe(f) for f in group.fields}
    shared = hints.pop() if len(hints) == 1 else None
    lines = [f"## {group.title or group.id}" + (f" [all: {shared}]" if shared else "")]
    names = {e.key: _list_names(e) for e in entries if e.is_row}
    tails = {tuple(t) for _, t in names.values()}
    one_header = len(names) > 1 and len(tails) == 1
    if one_header:  # every list has the same columns: say them once
        cols = tails.pop()
        lines.append(f"Columns of each list ({len(cols)} values), left to right: " + " | ".join(cols))
    for e in entries:
        hint = "" if shared else f" [{_describe_all(e.fields)}]"
        if e.is_row:
            name, cols = names[e.key]
            what = "" if one_header else f" ({len(cols)} values: {' | '.join(cols)})"
            lines.append(f"- {e.key}{what}: {name}{hint}")
        else:
            lines.append(f"- {e.key}: {e.fields[0].label}{hint}")
    return lines


def sectioned_prompt(spec: FormSpec, section: Section) -> str:
    """Prompt for one section of a sectioned form (one photo = one section, inside its corner markers).

    Kept short on purpose: the phone pays for every prompt token on every photo. A type hint shared by a whole
    group is stated once on its heading, fields side by side are listed as one row (left to right, which also
    tells the model where each value is), and only the answer rules for types the section has are included.
    """
    form = f'"{spec.name}"' + (f" ({spec.description})" if spec.description else "")
    title = f'"{section.title}"' + (f" ({section.code})" if section.code else "")
    lines = [f"This is a phone photo of section {section.id.upper()} of the paper form {form}: {title}."]
    if section.description:
        lines.append(section.description.rstrip(".") + ".")
    lines += ["Read the handwritten value of every field listed below.", "",
              "Fields, by group and row (left to right):"]
    layout = answer_layout(section)
    for group in section.groups:
        lines += _group_lines(group, [e for e in layout if e.group == group.id])
    if any(sum(bool(f.cells) for f in row) > 1 for g in section.groups for row in rows(g.fields)):
        # Measured on qwen3.5:9b: without this, "4" in one comb next to "851" in the next was read as 4851.
        lines.append("Each field is its own group of boxes. Groups in a row sit side by side: read each group on "
                     "its own and never join digits from neighbouring groups.")

    types = {f.type for f in section.fields}
    rules = [
        ("integer", "- numbers as JSON numbers, e.g. 21"),
        ("code", '- codes as strings, copied character by character (keep leading zeros)'),
        ("date", '- dates as "YYYY-MM-DD"'),
        ("text", "- text as strings"),
        ("choice", "- choices as the option text"),
        ("mark", "- marked options as the option text (the circle that is filled, ticked or crossed)"),
    ]
    lines += ["", "Answer with one line of JSON: an object with exactly these keys, in this order."]
    lines += [rule for t, rule in rules if t in types]
    if any(e.is_row for e in layout):
        # Seen on qwen3.5:9b: a skipped empty column, or a 2-digit number split into two values, shifts every later
        # value in the list.
        lines.append("- a list has exactly one value per column, left to right. An empty column is null, never "
                     "skipped. The digits in one column's boxes are one number.")
    lines += ["- a field left empty: null",
              f"Example of the shape (values made up, real answers have all keys): {_example(section)}",
              "If a value is hard to read, give your best reading. No other text, no code fences."]
    return "\n".join(lines)


BUILDERS: dict[str, Callable[[FormSpec, Section], str]] = {
    "sectioned": sectioned_prompt,
}


def build_prompt(spec: FormSpec, section: Section) -> str:
    return BUILDERS[spec.kind](spec, section)


def prompt_info(spec: FormSpec, section: Section) -> dict[str, str | int]:
    """What a stage records in run.json so later stages can check they use the same prompt."""
    text = build_prompt(spec, section)
    return {"kind": spec.kind, "section": section.id, "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "chars": len(text)}


def format_answer(section: Section, truth: dict[str, Any]) -> str:
    """The exact target text for one section photo: one-line JSON in answer_layout order, null for empty."""
    out: dict[str, Any] = {}
    for e in answer_layout(section):
        values = [truth.get(f.id) for f in e.fields]
        out[e.key] = values if e.is_row else values[0]
    return json.dumps(out, ensure_ascii=False)


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def parse_answer(section: Section, text: str) -> dict[str, Any]:
    """A model's reply about one section photo -> {field id: value}. Tolerates code fences and surrounding text.

    Missing keys become None and unknown keys are dropped, so callers always get every field id. A row list that
    is too short is padded with None (too long: cut). A field given under its own id instead of its row's key is
    accepted too. Raises ValueError if there is no JSON object at all.
    """
    text = _FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the answer")
    data = json.loads(text[start:end + 1])  # text from "{" to "}" can only decode to an object
    out: dict[str, Any] = {}
    for e in answer_layout(section):
        value = data.get(e.key)
        if not e.is_row:
            out[e.key] = value
            continue
        values = value if isinstance(value, list) else []
        for k, f in enumerate(e.fields):
            out[f.id] = values[k] if k < len(values) else data.get(f.id)
    return {f.id: _normalise(f, out.get(f.id)) for f in section.fields}


def _normalise(f: FormField, value: Any) -> Any:
    """Undo harmless type slips: an option or code given as a number ("campaign day": 2 for "2"), a count as a
    numeric string. Anything else is returned as it is, so a wrong answer still scores as wrong."""
    if value is None or isinstance(value, bool):
        return value
    if f.type in ("choice", "mark", "code") and isinstance(value, int | float):
        text = str(int(value)) if float(value).is_integer() else str(value)
        return text if f.type == "code" or text in f.options else value
    if f.type == "integer" and isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return value
