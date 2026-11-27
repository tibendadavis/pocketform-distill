"""Stage 2: teacher. The big model reads every synthetic photo; we keep its answer and its probabilities.

For each photo in data/<run_name>/synth/ the teacher (Ollama, `teacher.model`) gets the section's prompt and the
preprocessed image, and answers at temperature 0. Saved per photo, in data/<run_name>/teacher/{id}.json:

    reply       the teacher's text, exactly as generated
    logprobs    per generated token: the token, its log-probability and the `teacher.top_logprobs` most likely
                alternatives. These are the soft labels the student learns from (the dataset stage lines them up
                with the student's target)
    values      the reply parsed into {field id: value}
    correct     per field: does the teacher's value equal the synthetic truth?
    key         hash of everything the answer depends on (model, settings, prompt, image)

Resumable: a photo whose saved `key` still matches is skipped, so an interrupted run continues where it stopped
and raising `synth.num_photos` only labels the new photos. Changing the prompt, the teacher or the image settings
changes the key, so those photos are labelled again. Records for photos that no longer exist are removed.

Never reads real forms (private/): they are the test set only.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pocketform_distill.config import RunConfig
from pocketform_distill.preprocess import FACTOR, image_sha256, png_bytes, prepare_image
from pocketform_distill.prompt import build_prompt, format_answer, parse_answer, prompt_info
from pocketform_distill.provenance import write_run_info

Chat = Callable[..., Any]  # ollama.chat, or a fake in tests
Progress = Callable[[int, int, dict[str, Any]], None]  # (done, total, record) after each labelled photo


def _plain(x: Any) -> Any:
    """Ollama returns pydantic objects; store plain JSON."""
    if hasattr(x, "model_dump"):
        return _plain(x.model_dump(exclude_none=True))
    if isinstance(x, Mapping):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_plain(v) for v in x]
    return x


def _key(cfg: RunConfig, prompt_sha: str, image_sha: str) -> str:
    parts = {"model": cfg.teacher.model, "top_logprobs": cfg.teacher.top_logprobs, "think": cfg.teacher.think,
             "max_pixels": cfg.image.max_pixels, "factor": FACTOR, "prompt": prompt_sha, "image": image_sha,
             "format": "json", "temperature": 0}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _default_chat() -> Chat:
    import ollama

    return ollama.chat


def run_teacher(cfg: RunConfig, data_root: Path, chat: Chat | None = None, limit: int | None = None,
                section: str | None = None, progress: Progress | None = None) -> Path:
    """Label the synth photos of one run. Returns data/<run_name>/teacher."""
    synth = Path(data_root) / cfg.run_name / "synth"
    manifest_path = synth / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found: run `pocketform-distill synth` first")
    out = Path(data_root) / cfg.run_name / "teacher"
    out.mkdir(parents=True, exist_ok=True)
    chat = chat or _default_chat()

    spec = cfg.load_form()
    if section is not None:
        spec.section(section)  # unknown section id: fail now, with the list of ids
    photos = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    ids = {p["id"] for p in photos}
    for stale in out.glob("*.json"):
        if stale.name != "run.json" and stale.stem not in ids:
            stale.unlink()

    todo = [p for p in photos if section is None or p["section"] == section]
    prompts = {s.id: build_prompt(spec, s) for s in spec.sections}
    shas = {s.id: prompt_info(spec, s)["sha256"] for s in spec.sections}
    labelled = skipped = 0
    for photo in todo:
        if limit is not None and labelled >= limit:
            break
        sec = spec.section(photo["section"])
        image = prepare_image(synth / f"{photo['id']}.png", cfg.image.max_pixels)
        key = _key(cfg, shas[sec.id], image_sha256(image))
        path = out / f"{photo['id']}.json"
        if path.exists() and json.loads(path.read_text()).get("key") == key:
            skipped += 1
            continue

        t0 = time.monotonic()
        resp = _plain(chat(
            model=cfg.teacher.model,
            messages=[{"role": "user", "content": prompts[sec.id], "images": [png_bytes(image)]}],
            think=cfg.teacher.think,
            # JSON mode: never wraps the answer in code fences and keeps big answers on one line more often
            # (measured on qwen3.5:9b). The dataset stage does not need the exact one-line shape: it lines up
            # the teacher's probabilities with the student's target field by field.
            format="json",
            logprobs=True,
            top_logprobs=cfg.teacher.top_logprobs,
            options={"temperature": 0, "seed": 0},
        ))
        reply = resp["message"]["content"]
        truth = json.loads((synth / f"{photo['id']}.json").read_text())
        try:
            values = parse_answer(sec, reply)
            parsed = True
        except ValueError:
            values, parsed = {f.id: None for f in sec.fields}, False
        record = {
            "id": photo["id"], "section": sec.id, "split": photo["split"], "key": key,
            "model": cfg.teacher.model, "prompt_sha256": shas[sec.id], "image_size": list(image.size),
            "reply": reply, "parsed": parsed, "one_line": reply.strip() == format_answer(sec, values),
            "values": values, "correct": {k: values.get(k) == v for k, v in truth.items()},
            "eval_count": resp.get("eval_count"), "seconds": round(time.monotonic() - t0, 1),
            "logprobs": resp.get("logprobs") or [],
        }
        if not record["logprobs"]:
            raise RuntimeError("Ollama returned no logprobs: update Ollama (>= 0.12.11) and check the model")
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
        labelled += 1
        if progress:
            progress(labelled + skipped, len(todo), record)

    write_run_info(out, "teacher", cfg, prompts={s.id: prompt_info(spec, s) for s in spec.sections},
                   summary=summarise(out))
    return out


def summarise(out: Path) -> dict[str, Any]:
    """Teacher accuracy per section over every saved record (fields right / fields), and how often the reply was
    one line, parseable."""
    right: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    shape = defaultdict(lambda: [0, 0, 0])  # photos, parsed, one line
    for path in sorted(out.glob("*.json")):
        if path.name == "run.json":
            continue
        r = json.loads(path.read_text())
        right[r["section"]][0] += sum(r["correct"].values())
        right[r["section"]][1] += len(r["correct"])
        s = shape[r["section"]]
        s[0], s[1], s[2] = s[0] + 1, s[1] + r["parsed"], s[2] + r["one_line"]
    sections = {sid: {"photos": shape[sid][0], "fields_right": n, "fields": total,
                      "field_accuracy": round(n / total, 4) if total else None,
                      "parsed": shape[sid][1], "one_line": shape[sid][2]} for sid, (n, total) in sorted(right.items())}
    n = sum(v["fields_right"] for v in sections.values())
    total = sum(v["fields"] for v in sections.values())
    return {"sections": sections, "field_accuracy": round(n / total, 4) if total else None}
