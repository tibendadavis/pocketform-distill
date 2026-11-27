"""Stage 3: dataset. Turn synth truth + teacher answers into training examples for the student.

For each synthetic photo the student's target is the TRUTH in the one answer format (prompt.format_answer): one
line, short lists for table rows, null for empty. That is what the student learns to write on the phone, and the
hard label at every position.

Soft labels (the teacher's top-k probabilities, the part that makes this distillation and not just fine-tuning)
are added where they carry knowledge and line up exactly:

- only for fields the teacher read correctly (its value text equals the truth's). A misread field gets the hard
  label only, so the student never learns the teacher's mistakes.
- at the value's tokens, plus the token just before it (" " vs " null": empty or not?) and just after it ("," vs
  another digit: does the number go on?). Measured on qwen3.5:9b: keys were never < 99% sure (0 of 243 tokens),
  values were (61 of 191), and most uncertain punctuation was one of those two decisions.
- matched token by token: the teacher's reply may be laid out differently (pretty-printed over lines), but the
  text of a value and its neighbours is the same, so their tokens are too. A position only gets a soft label if
  the teacher's token there is exactly the student's.

Writes data/<run_name>/dataset/:
    train.jsonl, val.jsonl   one example per photo: id, section, image (path under data/<run_name>/), target text,
                             target_ids, and soft labels {pos, ids, logprobs} at target positions
    run.json                 config, git commit, prompt hashes, and coverage stats

Reads only data/<run_name>/synth/ and data/<run_name>/teacher/; never real forms.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from pocketform_distill.config import RunConfig, Section
from pocketform_distill.prompt import answer_layout, format_answer, prompt_info
from pocketform_distill.provenance import write_run_info
from pocketform_distill.token_mapping import to_student_id

Span = tuple[int, int]  # character start, end


def value_spans(text: str, section: Section) -> dict[str, Span]:
    """Where each field's value is in an answer (one line or pretty-printed): {field id: (start, end)}.

    Fields whose key is missing or whose value can't be read are left out.
    """
    dec = json.JSONDecoder()
    spans: dict[str, Span] = {}

    def skip_ws(i: int) -> int:
        while i < len(text) and text[i] in " \t\r\n":
            i += 1
        return i

    for e in answer_layout(section):
        m = re.search(r'"' + re.escape(e.key) + r'"\s*:\s*', text)
        if not m:
            continue
        i = m.end()
        try:
            if not e.is_row:
                _, j = dec.raw_decode(text, i)
                spans[e.fields[0].id] = (i, j)
                continue
            if text[i:i + 1] != "[":
                continue
            i += 1
            for f in e.fields:
                i = skip_ws(i)
                if text[i:i + 1] == "]":
                    break
                _, j = dec.raw_decode(text, i)
                spans[f.id] = (i, j)
                i = skip_ws(j)
                if text[i:i + 1] == ",":
                    i += 1
        except (json.JSONDecodeError, IndexError):
            continue
    return spans


def _positions(offsets: list[Span], span: Span) -> list[int]:
    """Token positions for a value span: the tokens covering it, plus the token right before and right after."""
    start, end = span
    inside = [k for k, (a, b) in enumerate(offsets) if a < end and b > start]
    if not inside:
        return []
    first, last = inside[0], inside[-1]
    out = inside[:]
    if first > 0 and offsets[first][0] == start:
        out.insert(0, first - 1)
    if last + 1 < len(offsets) and offsets[last][1] == end:
        out.append(last + 1)
    return out


def _soft(entry: dict, tok: Any) -> tuple[list[int], list[float]]:
    """A teacher position's top-k as student ids and log-probabilities. Teacher tokens that map to the same student
    id are merged; ones that map to none are dropped (the train stage renormalises what is left)."""
    probs: dict[int, float] = defaultdict(float)
    for alt in entry.get("top_logprobs") or [entry]:
        sid = to_student_id(alt, tok)
        if sid is not None:
            probs[sid] += math.exp(alt["logprob"])
    ids = sorted(probs, key=probs.get, reverse=True)
    return ids, [round(math.log(probs[i]), 5) for i in ids]


def build_example(section: Section, truth: dict[str, Any], teacher: dict | None, tok: Any) -> tuple[dict, dict]:
    """One training example and its alignment stats."""
    target = format_answer(section, truth)
    enc = tok(target, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    soft_pos: list[int] = []
    soft_ids: list[list[int]] = []
    soft_lps: list[list[float]] = []
    stats = {"fields": len(section.fields), "teacher_right": 0, "positions": 0, "soft": 0, "aligned_text": False}

    if teacher:
        tlp = teacher.get("logprobs") or []
        ttext = "".join(e["token"] for e in tlp)
        stats["aligned_text"] = ttext == teacher["reply"]  # token texts must add up to the reply
        tspans = value_spans(ttext, section) if stats["aligned_text"] else {}
        toffsets, pos = [], 0
        for e in tlp:
            toffsets.append((pos, pos + len(e["token"])))
            pos += len(e["token"])
        sspans = value_spans(target, section)
        done: set[int] = set()
        for f in section.fields:
            s_span, t_span = sspans.get(f.id), tspans.get(f.id)
            s_pos = _positions(offsets, s_span) if s_span else []
            stats["positions"] += len(s_pos)
            if not t_span or ttext[t_span[0]:t_span[1]] != target[s_span[0]:s_span[1]]:
                continue  # missing or misread: hard label only
            stats["teacher_right"] += 1
            t_pos = _positions(toffsets, t_span)
            if len(t_pos) != len(s_pos):
                continue
            for sp, tp in zip(s_pos, t_pos):
                a, b = offsets[sp]
                if sp in done or tlp[tp]["token"] != target[a:b]:
                    continue  # different token here (e.g. "]," vs "],\n" at a list's end): hard label
                sid, slp = _soft(tlp[tp], tok)
                if ids[sp] not in sid:
                    continue
                done.add(sp)
                soft_pos.append(sp)
                soft_ids.append(sid)
                soft_lps.append(slp)
                stats["soft"] += 1
    order = sorted(range(len(soft_pos)), key=soft_pos.__getitem__)
    example = {"target": target, "target_ids": ids,
               "soft": {"pos": [soft_pos[k] for k in order], "ids": [soft_ids[k] for k in order],
                        "logprobs": [soft_lps[k] for k in order]}}
    return example, stats


def _default_tokenizer(cfg: RunConfig) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg.student.model)


def run_dataset(cfg: RunConfig, data_root: Path, tokenizer: Any = None) -> Path:
    """Build data/<run_name>/dataset from synth + teacher. Returns that folder."""
    run = Path(data_root) / cfg.run_name
    synth, teacher_dir, out = run / "synth", run / "teacher", run / "dataset"
    if not (synth / "manifest.jsonl").exists():
        raise FileNotFoundError(f"{synth / 'manifest.jsonl'} not found: run `pocketform-distill synth` first")
    if not teacher_dir.exists():
        raise FileNotFoundError(f"{teacher_dir} not found: run `pocketform-distill teacher` first")
    spec = cfg.load_form()
    shas = {s.id: prompt_info(spec, s)["sha256"] for s in spec.sections}
    tok = tokenizer or _default_tokenizer(cfg)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files = {"train": (out / "train.jsonl").open("w"), "val": (out / "val.jsonl").open("w")}
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    try:
        for line in (synth / "manifest.jsonl").read_text().splitlines():
            photo = json.loads(line)
            section = spec.section(photo["section"])
            truth = json.loads((synth / f"{photo['id']}.json").read_text())
            tpath = teacher_dir / f"{photo['id']}.json"
            teacher = json.loads(tpath.read_text()) if tpath.exists() else None
            if teacher and teacher.get("prompt_sha256") != shas[section.id]:
                raise ValueError(f"{tpath} was made with a different prompt for section {section.id}: rerun "
                                 "`pocketform-distill teacher` (it relabels only what changed)")
            example, stats = build_example(section, truth, teacher, tok)
            files[photo["split"]].write(json.dumps({"id": photo["id"], "section": section.id,
                                                    "image": f"synth/{photo['id']}.png", **example}) + "\n")
            t = totals[section.id]
            t["photos"] += 1
            t["with_teacher"] += teacher is not None
            for k in ("fields", "teacher_right", "positions", "soft"):
                t[k] += stats[k]
            t["target_tokens"] += len(example["target_ids"])
    finally:
        for f in files.values():
            f.close()

    sections = {sid: dict(t) | {"soft_share_of_value_positions": round(t["soft"] / t["positions"], 4)
                                if t["positions"] else None} for sid, t in sorted(totals.items())}
    write_run_info(out, "dataset", cfg, prompts={s.id: prompt_info(spec, s) for s in spec.sections},
                   tokenizer=cfg.student.model, sections=sections)
    return out
