import io
import json
from pathlib import Path

import pytest
from PIL import Image

from pocketform_distill.config import load_run_config
from pocketform_distill.preprocess import FACTOR
from pocketform_distill.prompt import format_answer
from pocketform_distill.synth import run_synth
from pocketform_distill.teacher import run_teacher

ROOT = Path(__file__).resolve().parent.parent


class FakeTeacher:
    """Answers with the truth (pretty-printed, like qwen3.5:9b does for long answers), except one wrong field."""

    def __init__(self, synth: Path, wrong: str | None = "arrivals_count"):
        self.synth, self.wrong, self.calls = synth, wrong, []

    def __call__(self, **kw):
        self.calls.append(kw)
        pid = self.order.pop(0)  # photos are sent in manifest order
        truth = json.loads((self.synth / f"{pid}.json").read_text())
        if self.wrong:
            truth[self.wrong] = 999
        reply = json.dumps(truth, indent=2)
        tokens = [{"token": c, "logprob": -0.01, "top_logprobs": [{"token": c, "logprob": -0.01}]} for c in reply]
        return {"message": {"content": reply}, "logprobs": tokens, "eval_count": len(tokens)}


def setup(tmp_path, num_photos=3):
    cfg = load_run_config(ROOT / "configs/_template.yaml")
    cfg = cfg.model_copy(update={"synth": cfg.synth.model_copy(update={
        "num_photos": num_photos, "fonts_dir": ROOT / "assets/fonts/handwriting"})})
    synth = run_synth(cfg, tmp_path)
    fake = FakeTeacher(synth)
    fake.order = [json.loads(line)["id"] for line in (synth / "manifest.jsonl").read_text().splitlines()]
    return cfg, synth, fake


def test_labels_every_photo_with_reply_logprobs_and_accuracy(tmp_path):
    cfg, _synth, fake = setup(tmp_path)
    out = run_teacher(cfg, tmp_path, chat=fake)
    assert out == tmp_path / cfg.run_name / "teacher"
    rec = json.loads((out / "a-00001.json").read_text())
    assert rec["parsed"] and not rec["one_line"]  # pretty-printed reply: parsed, but not the one-line target
    assert rec["logprobs"][0]["top_logprobs"]
    assert rec["correct"]["arrivals_count"] is False and rec["correct"]["site_name"] is True
    summary = json.loads((out / "run.json").read_text())["summary"]
    assert summary["sections"]["a"]["fields"] == 3 * 5
    assert summary["sections"]["a"]["fields_right"] == 3 * 4
    call = fake.calls[0]
    assert call["format"] == "json" and call["options"]["temperature"] == 0 and call["logprobs"] is True


def test_teacher_gets_the_preprocessed_image(tmp_path):
    cfg, _synth, fake = setup(tmp_path, num_photos=1)
    cfg = cfg.model_copy(update={"image": cfg.image.model_copy(update={"max_pixels": 100_000})})
    run_teacher(cfg, tmp_path, chat=fake)
    w, h = Image.open(io.BytesIO(fake.calls[0]["messages"][0]["images"][0])).size
    assert w * h <= 100_000 and w % FACTOR == 0 and h % FACTOR == 0


def test_rerun_skips_labelled_photos_and_relabels_when_the_teacher_changes(tmp_path):
    cfg, _synth, fake = setup(tmp_path)
    run_teacher(cfg, tmp_path, chat=fake)
    assert len(fake.calls) == 3
    run_teacher(cfg, tmp_path, chat=fake)
    assert len(fake.calls) == 3  # nothing new to do
    other = cfg.model_copy(update={"teacher": cfg.teacher.model_copy(update={"model": "another:1b"})})
    fake.order = ["a-00000", "a-00001", "a-00002"]
    run_teacher(other, tmp_path, chat=fake)
    assert len(fake.calls) == 6  # a different teacher's answers are not reused


def test_more_photos_only_labels_the_new_ones_and_stale_records_go(tmp_path):
    cfg, _, fake = setup(tmp_path, num_photos=2)
    run_teacher(cfg, tmp_path, chat=fake)
    cfg4, _synth, fake4 = setup(tmp_path, num_photos=4)
    out = run_teacher(cfg4, tmp_path, chat=FakeOrder(fake4, ["a-00002", "a-00003"]))
    assert len(fake4.calls) == 2  # a-00000 and a-00001 kept: same seed, same image, same prompt
    cfg1, _, fake1 = setup(tmp_path, num_photos=1)
    run_teacher(cfg1, tmp_path, chat=fake1)
    assert sorted(p.name for p in out.glob("a-*.json")) == ["a-00000.json"]


def FakeOrder(fake, order):
    fake.order = list(order)
    return fake


def test_limit_and_section(tmp_path):
    cfg, _synth, fake = setup(tmp_path)
    run_teacher(cfg, tmp_path, chat=fake, limit=1)
    assert len(fake.calls) == 1
    with pytest.raises(ValueError):
        run_teacher(cfg, tmp_path, chat=fake, section="zz")


def test_needs_synth_first(tmp_path):
    cfg = load_run_config(ROOT / "configs/_template.yaml")
    with pytest.raises(FileNotFoundError, match="synth"):
        run_teacher(cfg, tmp_path, chat=lambda **kw: None)


def test_one_line_reply_is_recognised(tmp_path):
    cfg, synth, _fake = setup(tmp_path, num_photos=1)
    section = cfg.load_form().section("a")

    def compact(**kw):
        truth = json.loads((synth / "a-00000.json").read_text())
        reply = format_answer(section, truth)
        return {"message": {"content": reply}, "logprobs": [{"token": reply, "logprob": 0.0}]}

    out = run_teacher(cfg, tmp_path, chat=compact)
    assert json.loads((out / "a-00000.json").read_text())["one_line"] is True
