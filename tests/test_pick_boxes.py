import importlib.util
from pathlib import Path

import yaml

from pocketform_distill.config import FormSpec

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("pick_boxes", ROOT / "scripts/pick_boxes.py")
pick_boxes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pick_boxes)


def test_write_boxes_updates_only_the_named_boxes(tmp_path):
    form_yaml = tmp_path / "form.yaml"
    original = (ROOT / "forms/_template/form.yaml").read_text()
    form_yaml.write_text(original)

    pick_boxes.write_boxes(form_yaml, {"site_name": (1, 2, 3, 4), "departures_count": (5, 6, 7, 8)})

    boxes = {f.id: f.box for f in FormSpec.model_validate(yaml.safe_load(form_yaml.read_text())).fields}
    before = {f.id: f.box for f in FormSpec.model_validate(yaml.safe_load(original)).fields}
    assert boxes["site_name"] == (1, 2, 3, 4)
    assert boxes["departures_count"] == (5, 6, 7, 8)
    assert {k: v for k, v in boxes.items() if k not in ("site_name", "departures_count")} == {
        k: v for k, v in before.items() if k not in ("site_name", "departures_count")
    }
    assert "# Form spec template" in form_yaml.read_text()  # comments survive
    assert (tmp_path / "form.yaml.bak").read_text() == original


def test_write_boxes_handles_one_line_fields(tmp_path):
    form_yaml = tmp_path / "form.yaml"
    form_yaml.write_text("fields:\n  - {id: a_1, label: A, type: integer, box: [1, 1, 1, 1]}\n"
                         "  - {id: a, label: B, type: integer, box: [2, 2, 2, 2]}\n")
    pick_boxes.write_boxes(form_yaml, {"a": (9, 9, 9, 9)})
    assert form_yaml.read_text().splitlines()[1:] == [
        "  - {id: a_1, label: A, type: integer, box: [1, 1, 1, 1]}",
        "  - {id: a, label: B, type: integer, box: [9, 9, 9, 9]}",
    ]
