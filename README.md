# pocketform-distill

Teach a phone-sized vision-language model to read one paper form layout by distilling it from a larger model.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![CI](https://github.com/tibendadavis/pocketform-distill/actions/workflows/ci.yml/badge.svg)](https://github.com/tibendadavis/pocketform-distill/actions/workflows/ci.yml)

> **Status: early.** Smoke tests and stage 1 (synth) work. Stages 2-6 are not built yet.

## Why

Paper forms are still common wherever connectivity is poor. A small model running on the phone can read them without
sending the photos anywhere, which keeps the data private and works offline. A model that small isn't accurate enough
out of the box, but distilling it from a large model on a single known layout can close most of that gap.

## How it works

```
form spec ──► 1. synth ──► 2. teacher ──► 3. dataset ──► 4. train ──► 5. eval ──► 6. export
              fake filled   teacher reads  align teacher   student      score vs    package for
              forms + truth each form,      tokens to       learns from  targets,    device
                            saves top-k     student token   truth + soft log to
                            probabilities   ids             labels       MLflow
```

The teacher (`qwen3.5:9b`) runs in Ollama for inference only. For every token it returns its top-k alternatives with
probabilities, and these are the soft labels. The student (`Qwen/Qwen3.5-0.8B`) runs in Hugging Face Transformers
and is trained to match those probabilities as well as the ground truth. Each stage reads the previous stage's files
from `data/<run-name>/`, so you can rerun any stage on its own.

The form, the models and the output format are set in YAML, so nothing in the core is tied to one form.

## Measured so far

These are smoke-test numbers from an M2 Pro, not training or accuracy results.

| Check | Result |
|---|---|
| Teacher (Ollama `qwen3.5:9b`) speed | ~29 tok/s |
| Teacher → student tokenizer mapping | 99.9%, PASS |
| Student (0.8B) on Apple GPU | ~31–35 tok/s, 1.7 GB |
| Student on CPU | works; ~0.2 tok/s |

## Quick start

```bash
uv sync
uv run hf download Qwen/Qwen3.5-0.8B
uv run pytest
uv run python scripts/smoke_test.py path/to/form.jpg
uv run python scripts/teacher_smoke_test.py path/to/form.jpg --teacher qwen3.5:9b
```

**Student smoke test:** `smoke_test.py <image> [--form forms/<form>] [--model <hf-id>] [--device auto|cuda|mps|cpu]
[--max-new-tokens 512]` runs the form's prompt (from `prompt.py`) on one image with greedy decoding. It prints the device, load time, memory use, the output
with tokens/s, and the top-5 probabilities for the first 5 output tokens. An untrained student gives poor output, so the
test passes if it runs and prints probabilities. The dtype is bfloat16 on GPUs that support it, float16 on older NVIDIA
GPUs, and float32 on CPU.

**Teacher smoke test:** `teacher_smoke_test.py <image> --teacher <tag> [--form forms/<form>] [--student <hf-id>]
[--top 20] [--show-prompt]` sends the form's prompt to Ollama with logprobs on and prints:

- the teacher's output and tokens/s
- for a synthetic image (truth `{id}.json` next to it): fields correct, and whether the reply is exactly the target
  format. It must be, or the teacher's tokens won't line up with what the student learns.
- the top alternatives for the first 5 tokens
- a **tokenizer check**: the share of teacher tokens that map to exactly one student token id. Ollama returns tokens
  as text and bytes rather than ids, and distillation compares the two models token by token, so the tokens must map
  back. Over 99% prints `PASS`, anything lower prints `CHECK`.
- the **"< 99% sure" count**: positions where the teacher's top choice had under 99% probability. It must be above 0,
  or the logprobs carry no soft-label signal.

## Terms

| Term | Meaning |
|---|---|
| **form** | The whole paper form, described by `forms/<form>/form.yaml`. Its `kind` says how it is laid out and photographed |
| **section** | For `kind: sectioned`: one part of the page, marked by three printed corner squares. The user photographs one section at a time, and every stage works on one section photo at a time: fewer image tokens, fewer fields to confuse |
| **group** | Related fields inside a section, under one printed heading (e.g. MUAC, Bilateral oedema) |
| **field** | One handwritten value, with a box on the section's template image |

Only `sectioned` exists today. A later kind, such as one photo of a whole single-layout form, would get its own
spec shape and prompt builder.

**One student per form.** A single student model learns to read every section of its form. The prompt tells it which
section the photo shows, so the phone ships one model per form, not one per section. Training data mixes all
sections, and eval reports accuracy per section as well as overall.

## Stage 1: synth

```bash
uv run pocketform-distill synth --config configs/_template.yaml
```

Writes fake filled-in **section photos** with known answers to `data/<run_name>/synth/`, `synth.num_photos` per
section:

| File | Contents |
|---|---|
| `{id}.png` | the synthetic "phone photo" of one section; `id` = `<section>-<n>`, e.g. `b-00042` |
| `{id}.json` | truth for that section's fields, keyed by field id; `null` = left empty |
| `manifest.jsonl` | one line per photo: `id`, `section`, `split` (`train`/`val`), `seed`, and the `writer` and `defects` drawn from that seed |
| `run.json` | the config, form spec, fonts, template image hash per section and git commit it ran with |

How a photo is made:

1. **Blank.** The section's `template_image`: a clean image of that section of the empty form.
2. **Values.** Each field is left empty with probability `synth.empty_rate`, otherwise it gets a random value that
   respects its type: integers within `min`/`max` and skewed towards small counts, dates within `min`/`max`, choices
   and marks from `options`, text from `values` if given, codes from `values` or `pattern` (letters and digits
   in every cell by default).
3. **Handwriting.** One "writer" per photo (one of the OFL handwriting fonts in `assets/fonts/handwriting/`, a pen
   colour, a slant, a neatness and a way of marking circles) writes each value with random size, offset and
   rotation inside its box. Comb fields (`cells: 3`) get one character per cell: right-aligned for integers, from
   the first cell for codes. A `mark` field gets a fill, scribble, tick or cross on the chosen option's circle.
4. **Photo defects.** Each defect under `synth.augment` is applied with its own probability `p`: paper tint, uneven
   shading, rotation (±3°), perspective, blur, brightness/contrast, sensor noise and JPEG (quality 50-90). The
   comments in `configs/_template.yaml` say which phone-photo flaw each one imitates.

Each photo's seed is derived from `synth.seed`, its section id and its number (a hash, recorded as `seed` in
the manifest), and the same config, templates and commit give byte-identical output on the same machine. Adding,
removing or reordering sections, or raising `num_photos`, never changes a photo that already exists, so its
teacher outputs stay valid. To trace a bad image, look up its line in `manifest.jsonl` (font, pen, and
every defect with its values). In every section exactly `round(num_photos * val_fraction)` photos are marked `val`. Rerunning the stage replaces its folder. Synth never reads
real photos, which are the test set only.

**Setting up a form:** every section needs a template image, `forms/<form>/template-<section>.png`: a flat,
straight-on image of that section of the **empty** form, cropped to its corner markers. Best is an export of the form's print file (PDF -> PNG, ~1000-1600 px wide). Next best is a scanner-app shot
of an unfilled copy (iPhone Notes > Scan Documents, Google Drive > Scan, Microsoft Lens), which flattens and
straightens it. Don't use a filled-in phone photo: its handwriting would sit under the synthetic handwriting, it is
already distorted before synth adds its own defects, and real filled forms are kept for testing only.

**Boxes for a template:** a box is where the handwriting for one field goes, in template pixels. Run
`uv run python scripts/pick_boxes.py --form forms/<form> --section <id> --write` (`--section` can be left out when
the form has one section). A window opens that section's template and walks its fields in `form.yaml` order. The bottom bar names the field and its label, and an orange dashed rectangle shows
where `form.yaml` currently puts it. For each field, click the top-left then the bottom-right corner of its writing
area. For a comb field, take all cells as one box. For a `mark` field the bar walks through its options: box each
option's printed circle. Enter keeps the orange box, u undoes, s skips, q finishes.
`--write` saves the boxes into `form.yaml` (previous version in `form.yaml.bak`); without it they are only printed.

## The prompt

```bash
uv run pocketform-distill prompt --config configs/_template.yaml
```

`src/pocketform_distill/prompt.py` is the only place a prompt is written. Every stage that talks to a model calls
`build_prompt(form spec, section)`: one prompt per photo, so per section for a sectioned form, and later the exported bundle will ship that exact text to the phone. Distillation needs this:
the teacher's probabilities are "what comes next, given this prompt and image", and the student learns to match them
under the same prompt.

- **One builder per form `kind`.** `sectioned` names the section and form, then lists the section's fields by group,
  row by row, left to right (fields whose
  boxes sit side by side form a row; labels written `"<row>, <column>"` give the row and column names), then the
  answer rules and a one-line example. When digit boxes sit side by side it adds one rule: never join digits
  across neighbouring groups. That rule and the row layout raised the teacher from 85.6% to 90.6% of fields on 8
  hard synthetic forms, with 94 fewer prompt tokens. Every section of every sectioned form gets the same wording with
  its own fields. A new kind means a new
  `kind` value plus a builder in `prompt.BUILDERS`; a test fails if one is missing.
- **The answer format lives next to it.** `format_answer` gives the exact target text: one line of JSON in
  `form.yaml` order, `null` for empty. A field on its own is `"field_id": value`. Side-by-side fields of one type
  (a table row) are answered as short lists, left to right: `"cured_mam_m": [3, null, 12]`. A list's key is the
  part of its field ids they all share, so name grid fields `<row>_<column>` (`cured_mam_m_6_11m`). A row of more
  than 3 fields is split where the next word of the ids changes (`cured_mam_m_*` | `cured_mam_f_*` | ...).
  Field ids are most of an answer's tokens and the phone generates every one, but one list per long row made the
  teacher lose its place. On 6 synthetic photos of the nutrition form's section D (6 rows x 12 columns):

  | Answer format | Fields right (`qwen3.5:9b`) | Answer tokens | Teacher time |
  |---|---|---|---|
  | one key per field | 83% | ~1360 | 35 s |
  | one list per row (12 values) | 78% | ~255 | 13 s |
  | lists of at most 3 (used) | 90% | ~400 | 18 s |

  Each list names its own columns in the prompt (`- cured_mam_f (3 values: MAM Female 6-11 m | ...): Cured`).
  Saying the shared `6-11 m | 12-23 m | 24-59 m` once instead saved ~850 prompt tokens but dropped D to 78%.
  `parse_answer` reads a reply back into `{field id: value}`. Without the example line, `qwen3.5:9b` answered in
  fenced, pretty-printed JSON that doesn't match the target token for token.
- **Checked by hash.** Each stage will record `prompt_info` (kind, section, sha256) in its `run.json`, so outputs made with
  different prompts can't be mixed by mistake. Labels and field order are part of the prompt; boxes are not, so
  re-picking boxes doesn't invalidate teacher outputs.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.12
- [Ollama](https://ollama.com) ≥ 0.12.11 (needed for logprobs), running, with the teacher pulled
- a GPU for the student, chosen automatically in this order:

| Machine | Device | Notes |
|---------|--------|-------|
| Linux + NVIDIA GPU | `cuda` | `uv sync` installs the CUDA build of PyTorch. Optional faster kernels: `uv pip install flash-linear-attention causal-conv1d` |
| Apple Silicon Mac | `mps` | |
| Anything else | `cpu` | Works, but far too slow to train on |

AMD GPUs (ROCm) need a ROCm build of PyTorch, which this project does not configure.

## Project structure

```
assets/fonts/                   OFL handwriting fonts with their licences
configs/_template.yaml          run config template: one form + teacher + student + synth + targets
forms/_template/                form spec template: form.yaml + template-a.png (fictional form, one section, two groups)
scripts/pick_boxes.py           click field boxes on a template image, print their coordinates
scripts/smoke_test.py           student smoke test
scripts/teacher_smoke_test.py   teacher smoke test
src/pocketform_distill/         the package (CLI: `pocketform-distill`)
  config.py                     typed run config + form spec (pydantic)
  prompt.py                     the prompt per form kind + the answer format (used by every stage)
  synth.py, values.py,          stage 1: values, handwriting, phone-photo defects
  render.py, augment.py
tests/                          offline unit tests (no model downloads)
private/, data/                 real photos and stage outputs (git-ignored)
```

## Configuration

To use your own form, copy `forms/_template/` to `forms/<your-form>/` and `configs/_template.yaml` to
`configs/<your-run>.yaml`, replace `template-a.png` with a blank section of your form (one `template-<section>.png` per section), pick the
boxes, then edit both. Your own form folders and run configs are git-ignored by default. To publish
one on purpose, add `!forms/<your-form>/` and `!configs/<your-run>.yaml` to `.gitignore`.

**Form spec** (`forms/<name>/form.yaml`)

| Key | Meaning |
|-----|---------|
| `name`, `description`, `version` | Identifies the form |
| `kind` | Layout family. Only `sectioned` exists today: sections photographed one at a time |
| `sections[].id`, `title`, `code`, `description` | One section: what one photo covers. `id` is used in photo ids and the prompt ("section B") |
| `sections[].template_image` | Required. Clean image of the empty section, `template-<id>.png` next to `form.yaml` (only `_template`'s are committed) |
| `sections[].groups[].id`, `title` | Related fields under one printed heading |
| `…groups[].fields[].id` | Unique across the whole form; the key used in outputs |
| `…fields[].label` | What the field means. In a grid write `"<row>, <column>"`, which the prompt uses for row and column names |
| `…fields[].type` | `text` (any handwriting), `integer` (a count), `code` (letters and/or digits copied exactly, leading zeros kept), `date`, `choice` (option written as a word), `mark` (option chosen by filling / ticking / crossing its printed circle) |
| `…fields[].box` | `[x, y, w, h]` in template pixels; get them with `scripts/pick_boxes.py`. Optional for `mark` (the area around its option boxes) |
| `…fields[].options` | Allowed values for `choice` and `mark` fields |
| `…fields[].option_boxes` | `mark` only: one `[x, y, w, h]` per option, same order, around its printed circle |
| `…fields[].values` | Optional realistic values for `text` and `code` fields |
| `…fields[].min`, `max` | Optional range for `integer` (numbers) or `date` (`2024-01-01`) fields |
| `…fields[].cells` | `integer` and `code`: number of printed cells (a comb box), one character per cell |
| `…fields[].pattern` | `code` only: shape of synthetic codes, `A` = letter, `9` = digit, `*` = either, anything else as is (`AA-9999`). Default: `*` per cell |
| `…fields[].format` | `date` only: how people write it (default `%d/%m/%Y`). Truth is always ISO |

**Run config** (`configs/<run>.yaml`)

| Key | Meaning |
|-----|---------|
| `run_name` | Output folder name under `data/` |
| `form` | Path to the form folder |
| `teacher.backend`, `model` | Backend (`ollama`) and the exact tag from `ollama list` |
| `teacher.top_logprobs` | Alternatives saved per token (Ollama max: 20) |
| `teacher.think` | Reasoning mode; keep `false` for extraction |
| `image.max_pixels` | Larger images are downscaled to this before they reach the model (~1 token per 1024 px). It must stay the same for training, eval and the phone. An uncapped 12 MP photo is ~12k tokens |
| `student.model` | Hugging Face model id (Transformers / safetensors) |
| `student.method` | `lora` or `full` fine-tuning |
| `synth.num_photos` | Synthetic photos to generate per section |
| `synth.seed` | Base seed; each photo's seed is derived from it, the section id and the photo number |
| `synth.empty_rate` | Chance each field is left blank (truth `null`) |
| `synth.val_fraction` | Exact share of each section's photos marked `val` |
| `synth.fonts_dir`, `ink_colors` | Handwriting fonts and pen colours |
| `synth.augment.<defect>` | `p` (probability) and the range for each photo defect |
| `train.soft_label_weight` | Balance between matching the teacher's probabilities and the ground truth |
| `targets` | Thresholds that eval reports pass or fail against |
| `export.format` | `json`. Exporters are pluggable, so other output formats can be added |

## Data and privacy

Only synthetic data and templates are committed. `.gitignore` excludes `private/` and `data/` (real photos and stage
outputs), all photos (`*.jpg`, `*.jpeg`, `*.heic`, `*.png`, except `forms/*/template.png` and `forms/*/template-*.png`), model weights
(`*.safetensors`, `*.gguf`), MLflow runs (`mlruns/`, `mlartifacts/`), and every form and run config except the
`_template` ones, because a real form layout can be sensitive. Run `git add -n .` before committing to check what
would be included.

## Roadmap

- [x] Stage 1: synth
- [ ] Pipeline stages: teacher, dataset, train, eval, export
- [ ] Resolution sweep on `image.max_pixels`
- [ ] int8 / int4 quantization
- [ ] On-device test with LiteRT

## License

[MIT](LICENSE)
