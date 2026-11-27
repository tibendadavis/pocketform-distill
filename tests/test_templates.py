from pathlib import Path

import pytest
from PIL import Image
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
    assert form.template_path.exists()  # the committed blank template.png
    assert {f.type for f in form.fields} == {"text", "date", "choice", "integer"}


def _form(**field) -> dict:
    base = {"id": "a", "label": "A", "type": "integer", "box": [10, 10, 50, 20]}
    return {"name": "t", "template_image": "template.png", "sections": [{"id": "s", "fields": [base | field]}]}


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "choice"},  # choice without options
        {"options": ["x"]},  # options on a non-choice field
        {"min": 5, "max": 1},
        {"digits": 2, "max": 500},  # 500 does not fit in 2 cells
        {"box": [10, 10, 0, 20]},  # zero width
        {"type": "date", "min": 3},  # date range must be dates
        {"typo_key": 1},  # unknown keys fail instead of being ignored
    ],
)
def test_bad_form_specs_are_rejected(bad):
    with pytest.raises(ValidationError):
        FormSpec.model_validate(_form(**bad))


def test_template_image_is_required():
    spec = _form()
    del spec["template_image"]
    with pytest.raises(ValidationError, match="template_image"):
        FormSpec.model_validate(spec)


def _write_form(folder: Path, box: list[int], image: bool = True) -> Path:
    import yaml

    folder.mkdir()
    (folder / "form.yaml").write_text(yaml.safe_dump(_form(box=box)))
    if image:
        Image.new("RGB", (200, 100), "white").save(folder / "template.png")
    return folder


def test_missing_template_image_file_fails_with_a_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="pick_boxes"):
        load_form(_write_form(tmp_path / "f", [10, 10, 50, 20], image=False))


def test_box_outside_template_image_is_rejected(tmp_path):
    assert load_form(_write_form(tmp_path / "ok", [140, 70, 50, 20])).fields[0].box == (140, 70, 50, 20)
    with pytest.raises(ValueError, match="outside the 200x100"):
        load_form(_write_form(tmp_path / "bad", [190, 10, 50, 20]))


def test_duplicate_field_ids_are_rejected():
    spec = _form()
    spec["sections"].append({"id": "s2", "fields": spec["sections"][0]["fields"]})
    with pytest.raises(ValidationError, match="duplicate"):
        FormSpec.model_validate(spec)
