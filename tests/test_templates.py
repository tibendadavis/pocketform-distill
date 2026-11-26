from pathlib import Path

import pytest
from pydantic import ValidationError

from pocketform_distill.config import FormSpec, load_form, load_run_config

ROOT = Path(__file__).resolve().parent.parent


def test_run_config_template_loads():
    cfg = load_run_config(ROOT / "configs/_template.yaml")
    assert cfg.form == Path("forms/_template")
    assert cfg.export.format == "json"
    assert 0 <= cfg.synth.val_fraction < 1
    assert cfg.synth.augment.rotate.max_deg == 3


def test_form_template_loads_with_unique_field_ids():
    form = load_form(ROOT / "forms/_template")
    ids = [f.id for f in form.fields]
    assert ids
    assert len(ids) == len(set(ids))
    assert form.kind == "sectioned"
    assert form.template_image is None  # drawn from the spec, so the template needs no image assets
    assert {f.type for f in form.fields} == {"text", "date", "choice", "integer"}


def _form(**field) -> dict:
    base = {"id": "a", "label": "A", "type": "integer", "box": [10, 10, 50, 20]}
    return {"name": "t", "page": [200, 100], "sections": [{"id": "s", "fields": [base | field]}]}


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "choice"},  # choice without options
        {"options": ["x"]},  # options on a non-choice field
        {"min": 5, "max": 1},
        {"digits": 2, "max": 500},  # 500 does not fit in 2 cells
        {"box": [190, 10, 50, 20]},  # off the page
        {"type": "date", "min": 3},  # date range must be dates
        {"typo_key": 1},  # unknown keys fail instead of being ignored
    ],
)
def test_bad_form_specs_are_rejected(bad):
    with pytest.raises(ValidationError):
        FormSpec.model_validate(_form(**bad))


def test_duplicate_field_ids_are_rejected():
    spec = _form()
    spec["sections"].append({"id": "s2", "fields": spec["sections"][0]["fields"]})
    with pytest.raises(ValidationError, match="duplicate"):
        FormSpec.model_validate(spec)
