import json
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pocketform_distill.config import AugmentConfig, RunConfig, load_run_config
from pocketform_distill.synth import generate_form, list_fonts, make_truth, run_synth, val_indices

ROOT = Path(__file__).resolve().parent.parent


def config(**synth) -> RunConfig:
    cfg = load_run_config(ROOT / "configs/_template.yaml")
    synth = {"num_forms": 4, "fonts_dir": ROOT / "assets/fonts/handwriting", **synth}
    return cfg.model_copy(update={"form": ROOT / "forms/_template", "synth": cfg.synth.model_copy(update=synth)})


def test_fonts_are_committed_with_licences():
    fonts = list_fonts(ROOT / "assets/fonts/handwriting")
    assert 3 <= len(fonts) <= 5
    for f in fonts:
        family = f.stem.split("-")[0]
        assert (f.parent / f"{family}-OFL.txt").exists(), f"missing OFL licence for {f.name}"


def test_writes_expected_files(tmp_path):
    cfg = config()
    out = run_synth(cfg, tmp_path)
    assert out == tmp_path / cfg.run_name / "synth"
    manifest = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert [m["id"] for m in manifest] == ["00000", "00001", "00002", "00003"]
    assert [m["seed"] for m in manifest] == [cfg.synth.seed + i for i in range(4)]
    field_ids = [f.id for f in cfg.load_form().fields]
    for m in manifest:
        assert list(json.loads((out / f"{m['id']}.json").read_text())) == field_ids
        with Image.open(out / f"{m['id']}.png") as img:
            assert img.mode == "RGB"
    assert json.loads((out / "run.json").read_text())["stage"] == "synth"


def test_same_config_gives_byte_identical_output(tmp_path):
    a = run_synth(config(), tmp_path / "a")
    b = run_synth(config(), tmp_path / "b")
    files = sorted(p.name for p in a.iterdir())
    assert files == sorted(p.name for p in b.iterdir())
    for name in files:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_different_seed_gives_different_forms(tmp_path):
    a = run_synth(config(seed=1), tmp_path / "a")
    b = run_synth(config(seed=2), tmp_path / "b")
    assert (a / "00000.png").read_bytes() != (b / "00000.png").read_bytes()


def test_rerun_removes_stale_forms(tmp_path):
    run_synth(config(num_forms=4), tmp_path)
    out = run_synth(config(num_forms=2), tmp_path)
    assert not (out / "00003.png").exists()


@pytest.mark.parametrize("n,frac", [(10, 0.3), (50, 0.1), (7, 0.0), (200, 0.25)])
def test_split_fraction_is_exact(n, frac):
    val = val_indices(n, frac, seed=0)
    assert len(val) == round(n * frac)
    assert val <= set(range(n))


def test_manifest_split_matches_val_fraction(tmp_path):
    out = run_synth(config(num_forms=10, val_fraction=0.3), tmp_path)
    splits = [json.loads(line)["split"] for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert splits.count("val") == 3 and splits.count("train") == 7


@pytest.mark.parametrize("rate", [0.0, 0.25, 0.6, 1.0])
def test_empty_rate(rate):
    cfg = config(empty_rate=rate)
    spec = cfg.load_form()
    rng = random.Random(0)
    values = [v for _ in range(400) for v in make_truth(spec, cfg.synth, rng).values()]
    empty = sum(v is None for v in values) / len(values)
    assert abs(empty - rate) < 0.03


def test_empty_fields_get_no_ink():
    # With every field empty and no photo defects, the form must be exactly the blank template.
    no_aug = AugmentConfig(**{k: {"p": 0.0} for k in AugmentConfig.model_fields})
    cfg = config(empty_rate=1.0, augment=no_aug)
    spec = cfg.load_form()
    blank = spec.open_template()
    img, truth, _ = generate_form(spec, blank, cfg.synth, list_fonts(cfg.synth.fonts_dir), seed=0)
    assert all(v is None for v in truth.values())
    assert np.array_equal(np.asarray(img), np.asarray(blank))


def test_filled_fields_get_ink_inside_their_box():
    no_aug = AugmentConfig(**{k: {"p": 0.0} for k in AugmentConfig.model_fields})
    cfg = config(empty_rate=0.0, augment=no_aug)
    spec = cfg.load_form()
    blank = spec.open_template()
    img, _, _ = generate_form(spec, blank, cfg.synth, list_fonts(cfg.synth.fonts_dir), seed=0)
    diff = np.abs(np.asarray(img, np.int16) - np.asarray(blank, np.int16)).sum(-1) > 0
    for f in spec.fields:
        x, y, w, h = f.box
        assert diff[y:y + h, x:x + w].any(), f"no ink in {f.id}"
