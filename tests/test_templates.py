from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_run_config_template_loads():
    cfg = yaml.safe_load((ROOT / "configs/_template.yaml").read_text())
    assert cfg["form"] == "forms/_template"
    assert cfg["export"]["format"] == "json"


def test_form_template_loads_with_unique_field_ids():
    form = yaml.safe_load((ROOT / "forms/_template/form.yaml").read_text())
    ids = [f["id"] for f in form["fields"]]
    assert ids
    assert len(ids) == len(set(ids))
