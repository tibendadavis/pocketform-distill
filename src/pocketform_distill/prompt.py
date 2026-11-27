"""The one prompt, and the answer format that goes with it.

Every stage that talks to a model (teacher, dataset, train, eval, export, the phone) gets its prompt from
``build_prompt(spec)``. Distillation only works if they all see the same text: the teacher's probabilities are
"what comes next given this exact prompt and image", and the student is trained to match them under the same prompt.

One builder per form ``kind``. The builder turns a form spec into the prompt text, so each form gets the same
wording with its own field list. A new kind = a new ``kind`` value in config.FormSpec plus a builder registered in
``BUILDERS`` (a test fails if a kind has no builder).

The answer format is defined here too because it is part of the same contract: ``format_answer`` produces the exact
target text the student learns (and the teacher is asked for), ``parse_answer`` reads a model's reply back.

Every stage records ``prompt_info(spec)`` in its run.json. If the sha256 differs between stages, they did not
use the same prompt and their outputs must not be mixed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

from pocketform_distill.config import FormField, FormSpec


def _date_pattern(fmt: str) -> str:
    """strftime format -> what a person sees, e.g. %d/%m/%Y -> DD/MM/YYYY."""
    return fmt.replace("%d", "DD").replace("%m", "MM").replace("%Y", "YYYY").replace("%y", "YY")


def _describe(f: FormField) -> str:
    if f.type == "integer":
        return f"whole number, up to {f.digits} digits, one digit per box" if f.digits else "whole number"
    if f.type == "date":
        return f"date, written {_date_pattern(f.format)}"
    if f.type == "choice":
        return "one of: " + ", ".join(f.options)
    return "handwritten text"


def _example_value(f: FormField, i: int) -> Any:
    """A made-up but valid value for the example line. Fixed, not random: the prompt must never change."""
    if i == 1:
        return None  # show what an empty field looks like
    if f.type == "integer":
        return 12 if i == 0 else 7
    if f.type == "date":
        return "2025-01-31"
    if f.type == "choice":
        return f.options[0]
    return f.values[0] if f.values else "Example"


def _example(spec: FormSpec) -> str:
    """One line in the exact answer format. Without it the teacher answers in fenced, pretty-printed JSON, whose
    tokens don't line up with the one-line target the student learns (checked on qwen3.5:9b)."""
    first = spec.fields[:3]
    return json.dumps({f.id: _example_value(f, i) for i, f in enumerate(first)}, ensure_ascii=False)


def rows(fields: list[FormField]) -> list[list[FormField]]:
    """Group consecutive fields that sit side by side (boxes overlap vertically, left to right) into rows.

    Only the grouping reaches the prompt, not coordinates, so re-picking boxes doesn't change the prompt unless a
    field moves to another row.
    """
    out: list[list[FormField]] = []
    for f in fields:
        if out:
            prev = out[-1][-1]
            overlap = min(prev.box[1] + prev.box[3], f.box[1] + f.box[3]) - max(prev.box[1], f.box[1])
            if overlap > min(prev.box[3], f.box[3]) / 2 and f.box[0] > prev.box[0]:
                out[-1].append(f)
                continue
        out.append([f])
    return out


def _row_line(row: list[FormField], hint: bool) -> str:
    """One prompt line per row. Labels written "<row>, <column>" become `row <row>: id (column) | ...`."""
    def h(f: FormField) -> str:
        return f" [{_describe(f)}]" if hint else ""

    if len(row) == 1:
        return f"- {row[0].id}: {row[0].label}{h(row[0])}"
    split = [f.label.rsplit(", ", 1) for f in row]
    if all(len(s) == 2 for s in split) and len({s[0] for s in split}) == 1:
        return f"- row {split[0][0]}: " + " | ".join(f"{f.id} ({s[1]}){h(f)}" for f, s in zip(row, split))
    return "- row: " + " | ".join(f"{f.id} ({f.label}){h(f)}" for f in row)


def sectioned_prompt(spec: FormSpec) -> str:
    """Prompt for a form made of titled sections of fields.

    Kept short on purpose: the phone pays for every prompt token on every photo. A type hint shared by a whole
    section is stated once on its heading, fields side by side are listed as one row (left to right, which also
    tells the model where each value is), and only the answer rules for types the form has are included.
    """
    head = f'This is a phone photo of a paper form: "{spec.name}"'
    lines = [head + (f" ({spec.description})." if spec.description else "."),
             "Read the handwritten value of every field listed below.", "",
             "Fields, by section and row (left to right):"]
    side_by_side_combs = False
    for section in spec.sections:
        hints = {_describe(f) for f in section.fields}
        shared = hints.pop() if len(hints) == 1 else None
        lines.append(f"## {section.title or section.id}" + (f" [all: {shared}]" if shared else ""))
        for row in rows(section.fields):
            lines.append(_row_line(row, hint=shared is None))
            side_by_side_combs |= sum(bool(f.digits) for f in row) > 1
    if side_by_side_combs:
        # Measured on qwen3.5:9b: without this, "4" in one comb next to "851" in the next was read as 4851.
        lines.append("Each field is its own group of boxes. Groups in a row sit side by side: read each group on "
                     "its own and never join digits from neighbouring groups.")

    types = {f.type for f in spec.fields}
    rules = [
        ("integer", "- numbers as JSON numbers, e.g. 21"),
        ("date", '- dates as "YYYY-MM-DD"'),
        ("text", "- text as strings"),
        ("choice", "- choices as the option text"),
    ]
    lines += ["", "Answer with one line of JSON: an object with exactly these keys, in this order."]
    lines += [rule for t, rule in rules if t in types]
    lines += ["- a field left empty: null",
              f"Example of the shape (values made up, real answers have all keys): {_example(spec)}",
              "If a value is hard to read, give your best reading. No other text, no code fences."]
    return "\n".join(lines)


BUILDERS: dict[str, Callable[[FormSpec], str]] = {
    "sectioned": sectioned_prompt,
}


def build_prompt(spec: FormSpec) -> str:
    return BUILDERS[spec.kind](spec)


def prompt_info(spec: FormSpec) -> dict[str, str | int]:
    """What a stage records in run.json so later stages can check they use the same prompt."""
    text = build_prompt(spec)
    return {"kind": spec.kind, "sha256": hashlib.sha256(text.encode()).hexdigest(), "chars": len(text)}


def format_answer(spec: FormSpec, truth: dict[str, Any]) -> str:
    """The exact target text for a form: one-line JSON, keys in form.yaml order, null for empty."""
    return json.dumps({f.id: truth.get(f.id) for f in spec.fields}, ensure_ascii=False)


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def parse_answer(spec: FormSpec, text: str) -> dict[str, Any]:
    """A model's reply -> {field id: value}. Tolerates code fences and surrounding text.

    Missing keys become None and unknown keys are dropped, so callers always get every field id. Raises
    ValueError if there is no JSON object at all.
    """
    text = _FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the answer")
    data = json.loads(text[start:end + 1])  # text from "{" to "}" can only decode to an object
    return {f.id: data.get(f.id) for f in spec.fields}
