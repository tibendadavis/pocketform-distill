import copy
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from pocketform_distill.augment import STEPS
from pocketform_distill.config import AugmentConfig, RunConfig, load_run_config
from pocketform_distill.synth import generate_photo, list_fonts, make_truth, run_synth, val_indices

ROOT = Path(__file__).resolve().parent.parent
NO_AUG = AugmentConfig(**{k: {"p": 0.0} for k in AugmentConfig.model_fields})


def config(form: Path = ROOT / "forms/_template", **synth) -> RunConfig:
    cfg = load_run_config(ROOT / "configs/_template.yaml")
    synth = {"num_photos": 4, "fonts_dir": ROOT / "assets/fonts/handwriting", **synth}
    return cfg.model_copy(update={"form": form, "synth": cfg.synth.model_copy(update=synth)})


@pytest.fixture
def two_section_form(tmp_path) -> Path:
    """The template form plus a copy of its section as section "b" (own template file, own field ids)."""
    folder = tmp_path / "form"
    folder.mkdir()
    data = yaml.safe_load((ROOT / "forms/_template/form.yaml").read_text())
    b = copy.deepcopy(data["sections"][0])
    b["id"], b["template_image"] = "b", "template-b.png"
    for group in b["groups"]:
        for f in group["fields"]:
            f["id"] += "_b"
    data["sections"].append(b)
    (folder / "form.yaml").write_text(yaml.safe_dump(data))
    shutil.copy(ROOT / "forms/_template/template-a.png", folder / "template-a.png")
    shutil.copy(ROOT / "forms/_template/template-a.png", folder / "template-b.png")
    return folder


def manifest(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]


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
    lines = manifest(out)
    assert [m["id"] for m in lines] == ["a-00000", "a-00001", "a-00002", "a-00003"]
    assert {m["section"] for m in lines} == {"a"}
    assert [m["seed"] for m in lines] == [cfg.synth.seed + i for i in range(4)]
    field_ids = [f.id for f in cfg.load_form().section("a").fields]
    for m in lines:
        assert list(json.loads((out / f"{m['id']}.json").read_text())) == field_ids
        with Image.open(out / f"{m['id']}.png") as img:
            assert img.mode == "RGB"
    run = json.loads((out / "run.json").read_text())
    assert run["stage"] == "synth" and len(run["template_sha256"]["a"]) == 64


def test_each_section_gets_its_own_photos_truth_and_seeds(tmp_path, two_section_form):
    cfg = config(form=two_section_form, num_photos=3)
    out = run_synth(cfg, tmp_path)
    lines = manifest(out)
    assert [m["id"] for m in lines] == ["a-00000", "a-00001", "a-00002", "b-00000", "b-00001", "b-00002"]
    seeds = [m["seed"] for m in lines]
    assert len(set(seeds)) == len(seeds)  # no two photos share a seed, even across sections
    b_truth = json.loads((out / "b-00001.json").read_text())
    assert all(k.endswith("_b") for k in b_truth)  # a photo's truth only has its own section's fields
    assert set(json.loads((out / "run.json").read_text())["template_sha256"]) == {"a", "b"}


def test_manifest_records_writer_and_defects(tmp_path):
    out = run_synth(config(num_photos=6), tmp_path)
    step_names = [name for name, _ in STEPS]
    for m in manifest(out):
        assert m["writer"]["font"].endswith(".ttf")
        assert list(m["defects"]) == [n for n in step_names if n in m["defects"]]  # applied in STEPS order
        if "rotate" in m["defects"]:
            assert abs(m["defects"]["rotate"]["deg"]) <= 3
        assert 50 <= m["defects"]["jpeg"]["quality"] <= 90  # jpeg p = 1.0 in the template config


def test_photo_can_be_regenerated_from_its_manifest_seed(tmp_path):
    cfg = config()
    out = run_synth(cfg, tmp_path)
    m = manifest(out)[2]
    section = cfg.load_form().section(m["section"])
    img, truth, _ = generate_photo(section, section.open_template(), cfg.synth, list_fonts(cfg.synth.fonts_dir),
                                   m["seed"])
    assert np.array_equal(np.asarray(img), np.asarray(Image.open(out / f"{m['id']}.png")))
    assert truth == json.loads((out / f"{m['id']}.json").read_text())


def test_same_config_gives_byte_identical_output(tmp_path):
    a = run_synth(config(), tmp_path / "a")
    b = run_synth(config(), tmp_path / "b")
    files = sorted(p.name for p in a.iterdir())
    assert files == sorted(p.name for p in b.iterdir())
    for name in files:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_different_seed_gives_different_photos(tmp_path):
    a = run_synth(config(seed=1), tmp_path / "a")
    b = run_synth(config(seed=2), tmp_path / "b")
    assert (a / "a-00000.png").read_bytes() != (b / "a-00000.png").read_bytes()


def test_rerun_removes_stale_photos(tmp_path):
    run_synth(config(num_photos=4), tmp_path)
    out = run_synth(config(num_photos=2), tmp_path)
    assert not (out / "a-00003.png").exists()


@pytest.mark.parametrize("n,frac", [(10, 0.3), (50, 0.1), (7, 0.0), (200, 0.25)])
def test_split_fraction_is_exact(n, frac):
    val = val_indices(n, frac, seed=0)
    assert len(val) == round(n * frac)
    assert val <= set(range(n))


def test_val_share_is_exact_in_every_section(tmp_path, two_section_form):
    out = run_synth(config(form=two_section_form, num_photos=10, val_fraction=0.3), tmp_path)
    for section in ("a", "b"):
        splits = [m["split"] for m in manifest(out) if m["section"] == section]
        assert splits.count("val") == 3 and splits.count("train") == 7


@pytest.mark.parametrize("rate", [0.0, 0.25, 0.6, 1.0])
def test_empty_rate(rate):
    cfg = config(empty_rate=rate)
    section = cfg.load_form().section("a")
    rng = random.Random(0)
    values = [v for _ in range(400) for v in make_truth(section, cfg.synth, rng).values()]
    empty = sum(v is None for v in values) / len(values)
    assert abs(empty - rate) < 0.03


def test_empty_fields_get_no_ink():
    # With every field empty and no photo defects, the photo must be exactly the blank template.
    cfg = config(empty_rate=1.0, augment=NO_AUG)
    section = cfg.load_form().section("a")
    blank = section.open_template()
    img, truth, _ = generate_photo(section, blank, cfg.synth, list_fonts(cfg.synth.fonts_dir), seed=0)
    assert all(v is None for v in truth.values())
    assert np.array_equal(np.asarray(img), np.asarray(blank))


def test_filled_fields_get_ink_inside_their_box():
    cfg = config(empty_rate=0.0, augment=NO_AUG)
    section = cfg.load_form().section("a")
    blank = section.open_template()
    img, _, _ = generate_photo(section, blank, cfg.synth, list_fonts(cfg.synth.fonts_dir), seed=0)
    diff = np.abs(np.asarray(img, np.int16) - np.asarray(blank, np.int16)).sum(-1) > 0
    for f in section.fields:
        x, y, w, h = f.box
        assert diff[y:y + h, x:x + w].any(), f"no ink in {f.id}"
