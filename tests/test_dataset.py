import json
import re
from pathlib import Path

import pytest

from pocketform_distill.config import FormSpec, load_run_config
from pocketform_distill.dataset import build_example, run_dataset, value_spans
from pocketform_distill.prompt import format_answer
from pocketform_distill.synth import run_synth
from pocketform_distill.teacher import run_teacher

ROOT = Path(__file__).resolve().parent.parent
# Like the Qwen tokenizer where it matters: one token per digit, " null" one token, a space before a number its
# own token, punctuation runs merged ("],", '":').
TOKEN = re.compile(r' ?null|\d|[A-Za-z_]+| +|\n *|"|[^\w\s"]+')


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id(self, t: str) -> int:
        return self.vocab.setdefault(t, len(self.vocab) + 1)

    def pieces(self, text: str) -> list[re.Match]:
        return list(TOKEN.finditer(text))

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ms = self.pieces(text)
        return {"input_ids": [self._id(m.group()) for m in ms], "offset_mapping": [m.span() for m in ms]}

    def encode(self, text, add_special_tokens=False):
        return [self._id(m.group()) for m in self.pieces(text)]

    def convert_tokens_to_ids(self, piece):
        return self.unk_token_id


def teacher_record(text: str, tok: FakeTokenizer, unsure: str = "7") -> dict:
    """What run_teacher saves: the reply and, per token, the chosen token plus one alternative."""
    lps = [{"token": m.group(), "logprob": -0.1,
            "top_logprobs": [{"token": m.group(), "logprob": -0.1}, {"token": unsure, "logprob": -2.4}]}
           for m in tok.pieces(text)]
    return {"reply": text, "logprobs": lps}


def grid() -> FormSpec:
    def f(fid, label, x, y):
        return {"id": fid, "label": label, "type": "integer", "cells": 2, "box": [x, y, 50, 40]}

    return FormSpec.model_validate({"name": "g", "sections": [{"id": "s", "title": "S", "template_image": "t.png",
        "groups": [{"id": "g", "fields": [
            f("cured_m", "Cured, Male", 100, 50), f("cured_f", "Cured, Female", 160, 50),
            f("total", "Total", 100, 120),
        ]}]}]})


TRUTH = {"cured_m": 12, "cured_f": None, "total": 7}


def test_value_spans_in_one_line_and_pretty_answers():
    section = grid().section()
    one_line = format_answer(section, TRUTH)
    pretty = json.dumps({"cured": [12, None], "total": 7}, indent=2)
    for text in (one_line, pretty):
        spans = value_spans(text, section)
        assert {k: text[a:b] for k, (a, b) in spans.items()} == {"cured_m": "12", "cured_f": "null", "total": "7"}
    assert value_spans('{"total": 3}', section) == {"total": (10, 11)}  # missing keys are left out


# How qwen3.5:9b lays out long answers: one key per line, each list on one line.
PRETTY = '{\n    "cured": [12, null],\n    "total": 7\n}'


def test_pretty_teacher_reply_still_gives_soft_labels_on_every_value():
    section, tok = grid().section(), FakeTokenizer()
    example, stats = build_example(section, TRUTH, teacher_record(PRETTY, tok), tok)
    target = example["target"]
    assert target == '{"cured": [12, null], "total": 7}'
    soft_text = [tok.pieces(target)[p].group() for p in example["soft"]["pos"]]
    # Value tokens and the decisions around them: "[" before 12 (is the first item empty?), "1", "2", "," after it
    # (does the number go on?), " null" and the "]," after it, " " before 7 and "7". Not the "}" after 7: the reply
    # has a newline there, a different token, so that position keeps the hard label only.
    assert soft_text == ["[", "1", "2", ",", " null", "],", " ", "7"]
    assert stats["teacher_right"] == 3
    for text, ids, lps in zip(soft_text, example["soft"]["ids"], example["soft"]["logprobs"]):
        if text == "7":  # chosen and alternative are the same student token: merged into one
            assert len(ids) == 1
        else:  # chosen + the "7" alternative, most likely first
            assert len(ids) == len(lps) == 2 and lps[0] > lps[1]


def test_list_items_on_separate_lines_keep_soft_labels_on_the_digits():
    section, tok = grid().section(), FakeTokenizer()
    one_per_line = json.dumps({"cured": [12, None], "total": 7}, indent=2)
    example, _ = build_example(section, TRUTH, teacher_record(one_per_line, tok), tok)
    soft_text = [tok.pieces(example["target"])[p].group() for p in example["soft"]["pos"]]
    assert "1" in soft_text and "2" in soft_text and "7" in soft_text
    assert " null" not in soft_text  # the reply has "null" after a newline: a different token, so hard label


def test_misread_field_gets_no_soft_labels():
    section, tok = grid().section(), FakeTokenizer()
    wrong = format_answer(section, TRUTH | {"cured_m": 13})
    example, stats = build_example(section, TRUTH, teacher_record(wrong, tok), tok)
    assert stats["teacher_right"] == 2
    target_pieces = tok.pieces(example["target"])
    twelve = [k for k, m in enumerate(target_pieces) if m.group() in ("1", "2")][:2]
    assert not set(twelve) & set(example["soft"]["pos"])  # the student is never taught "13"


def test_no_teacher_means_hard_labels_only():
    section, tok = grid().section(), FakeTokenizer()
    example, stats = build_example(section, TRUTH, None, tok)
    assert example["soft"]["pos"] == [] and example["target_ids"]
    assert stats["soft"] == 0


def test_run_dataset_end_to_end(tmp_path):
    cfg = load_run_config(ROOT / "configs/_template.yaml")
    cfg = cfg.model_copy(update={"synth": cfg.synth.model_copy(update={
        "num_photos": 4, "val_fraction": 0.25, "fonts_dir": ROOT / "assets/fonts/handwriting"})})
    synth = run_synth(cfg, tmp_path)
    tok = FakeTokenizer()
    section = cfg.load_form().section("a")
    order = [json.loads(line)["id"] for line in (synth / "manifest.jsonl").read_text().splitlines()]

    def chat(**kw):
        reply = format_answer(section, json.loads((synth / f"{order.pop(0)}.json").read_text()))
        return {"message": {"content": reply}, "logprobs": teacher_record(reply, tok)["logprobs"]}

    run_teacher(cfg, tmp_path, chat=chat)
    out = run_dataset(cfg, tmp_path, tokenizer=tok)
    train = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    val = [json.loads(x) for x in (out / "val.jsonl").read_text().splitlines()]
    assert len(train) == 3 and len(val) == 1
    ex = train[0]
    assert ex["image"] == f"synth/{ex['id']}.png" and ex["soft"]["pos"]
    stats = json.loads((out / "run.json").read_text())["sections"]["a"]
    assert stats["teacher_right"] == stats["fields"] == 4 * 5
    assert stats["soft_share_of_value_positions"] > 0.8


def test_teacher_outputs_for_another_prompt_are_refused(tmp_path):
    cfg = load_run_config(ROOT / "configs/_template.yaml")
    cfg = cfg.model_copy(update={"synth": cfg.synth.model_copy(update={
        "num_photos": 1, "fonts_dir": ROOT / "assets/fonts/handwriting"})})
    synth = run_synth(cfg, tmp_path)
    teacher = tmp_path / cfg.run_name / "teacher"
    teacher.mkdir()
    (teacher / "a-00000.json").write_text(json.dumps({"prompt_sha256": "old", "reply": "{}", "logprobs": []}))
    with pytest.raises(ValueError, match="different prompt"):
        run_dataset(cfg, tmp_path, tokenizer=FakeTokenizer())
    assert synth.exists()
