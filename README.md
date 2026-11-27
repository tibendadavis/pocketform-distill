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

**Student smoke test:** `smoke_test.py <image> [--model <hf-id>] [--device auto|cuda|mps|cpu] [--max-new-tokens 512]`
runs the extraction prompt on one image with greedy decoding. It prints the device, load time, memory use, the output
with tokens/s, and the top-5 probabilities for the first 5 output tokens. An untrained student gives poor output, so the
test passes if it runs and prints probabilities. The dtype is bfloat16 on GPUs that support it, float16 on older NVIDIA
GPUs, and float32 on CPU.

**Teacher smoke test:** `teacher_smoke_test.py <image> --teacher <tag> [--student <hf-id>] [--top 20]` calls Ollama
with logprobs on and prints:

- the teacher's output and tokens/s
- the top alternatives for the first 5 tokens
- a **tokenizer check**: the share of teacher tokens that map to exactly one student token id. Ollama returns tokens
  as text and bytes rather than ids, and distillation compares the two models token by token, so the tokens must map
  back. Over 99% prints `PASS`, anything lower prints `CHECK`.
- the **"< 99% sure" count**: positions where the teacher's top choice had under 99% probability. It must be above 0,
  or the logprobs carry no soft-label signal.

## Stage 1: synth

```bash
uv run pocketform-distill synth --config configs/_template.yaml
```

Writes fake filled forms with known answers to `data/<run_name>/synth/`:

| File | Contents |
|---|---|
| `{id}.png` | the synthetic "phone photo" |
| `{id}.json` | truth keyed by field id; `null` = left empty |
| `manifest.jsonl` | one line per form: `id`, `split` (`train`/`val`), `seed`, and the `writer` and `defects` drawn from that seed |
| `run.json` | the config, form spec, fonts, template image hash and git commit it ran with |

How a form is made:

1. **Blank.** The form's `template_image`: a clean image of the empty form.
2. **Values.** Each field is left empty with probability `synth.empty_rate`, otherwise it gets a random value that
   respects its type: integers within `min`/`max` and skewed towards small counts, dates within `min`/`max`, choices
   from `options`, text from `values` if given.
3. **Handwriting.** One "writer" per form (one of the OFL handwriting fonts in `assets/fonts/handwriting/`, a pen
   colour, a slant and a neatness) writes each value with random size, offset and rotation inside its box. Comb
   integer fields (`digits: 3`) get one right-aligned digit per cell.
4. **Photo defects.** Each defect under `synth.augment` is applied with its own probability `p`: paper tint, uneven
   shading, rotation (±3°), perspective, blur, brightness/contrast, sensor noise and JPEG (quality 50-90). The
   comments in `configs/_template.yaml` say which phone-photo flaw each one imitates.

Form `i` uses seed `synth.seed + i`, and the same config, template and commit give byte-identical output on the same
machine. To trace a bad image, look up its line in `manifest.jsonl` (font, pen, and every defect with its values). Exactly
`round(num_forms * val_fraction)` forms are marked `val`. Rerunning the stage replaces its folder. Synth never reads
real photos, which are the test set only.

**Setting up a form:** every form needs `forms/<form>/template.png`, a flat, straight-on image of the **empty**
form. Best is an export of the form's print file (PDF -> PNG, ~1000-1600 px wide). Next best is a scanner-app shot
of an unfilled copy (iPhone Notes > Scan Documents, Google Drive > Scan, Microsoft Lens), which flattens and
straightens it. Don't use a filled-in phone photo: its handwriting would sit under the synthetic handwriting, it is
already distorted before synth adds its own defects, and real filled forms are kept for testing only.

**Boxes for a template:** a box is where the handwriting for one field goes, in template pixels. Run
`uv run python scripts/pick_boxes.py forms/<form>/template.png --form forms/<form> --write`. A window walks the
fields in `form.yaml` order. The bottom bar names the field and its label, and an orange dashed rectangle shows
where `form.yaml` currently puts it. For each field, click the top-left then the bottom-right corner of its writing
area. For a comb field, take all digit cells as one box. Enter keeps the orange box, u undoes, s skips, q finishes.
`--write` saves the boxes into `form.yaml` (previous version in `form.yaml.bak`); without it they are only printed.

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
forms/_template/                form spec template: form.yaml + template.png (fictional two-section form)
scripts/pick_boxes.py           click field boxes on a template image, print their coordinates
scripts/smoke_test.py           student smoke test
scripts/teacher_smoke_test.py   teacher smoke test
src/pocketform_distill/         the package (CLI: `pocketform-distill`)
  config.py                     typed run config + form spec (pydantic)
  synth.py, values.py,          stage 1: values, handwriting, phone-photo defects
  render.py, augment.py
tests/                          offline unit tests (no model downloads)
private/, data/                 real photos and stage outputs (git-ignored)
```

## Configuration

To use your own form, copy `forms/_template/` to `forms/<your-form>/` and `configs/_template.yaml` to
`configs/<your-run>.yaml`, replace `template.png` with your empty form, pick the boxes, then edit both. Your own form folders and run configs are git-ignored by default. To publish
one on purpose, add `!forms/<your-form>/` and `!configs/<your-run>.yaml` to `.gitignore`.

**Form spec** (`forms/<name>/form.yaml`)

| Key | Meaning |
|-----|---------|
| `name`, `description`, `version` | Identifies the form |
| `kind` | Layout family. Only `sectioned` exists today: a stack of titled sections of fields. Other kinds can be added later; later stages only see the flat list of fields |
| `template_image` | Required. Clean image of the empty form next to `form.yaml` (only `template.png` is committed) |
| `sections[].id`, `title` | A titled block of fields |
| `sections[].fields[].id` | Unique key used in outputs |
| `…fields[].label` | What the field means, e.g. as printed on the form |
| `…fields[].type` | `text` (any handwriting), `integer`, `date`, `choice` |
| `…fields[].box` | `[x, y, w, h]` in template pixels; get them with `scripts/pick_boxes.py` |
| `…fields[].options` | Allowed values for `choice` fields |
| `…fields[].values` | Optional realistic values for `text` fields |
| `…fields[].min`, `max` | Optional range for `integer` (numbers) or `date` (`2024-01-01`) fields |
| `…fields[].digits` | `integer` only: number of printed digit cells (a comb box), one digit per cell |
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
| `synth.num_forms` | Number of synthetic filled forms to generate |
| `synth.seed` | Base seed; form `i` uses `seed + i` |
| `synth.empty_rate` | Chance each field is left blank (truth `null`) |
| `synth.val_fraction` | Exact share of forms marked `val` |
| `synth.fonts_dir`, `ink_colors` | Handwriting fonts and pen colours |
| `synth.augment.<defect>` | `p` (probability) and the range for each photo defect |
| `train.soft_label_weight` | Balance between matching the teacher's probabilities and the ground truth |
| `targets` | Thresholds that eval reports pass or fail against |
| `export.format` | `json`. Exporters are pluggable, so other output formats can be added |

## Data and privacy

Only synthetic data and templates are committed. `.gitignore` excludes `private/` and `data/` (real photos and stage
outputs), all photos (`*.jpg`, `*.jpeg`, `*.heic`, `*.png`, except `forms/*/template.png`), model weights
(`*.safetensors`, `*.gguf`), MLflow runs (`mlruns/`, `mlartifacts/`), and every form and run config except the
`_template` ones, because a real form layout can be sensitive. Run `git add -n .` before committing to check what
would be included.

## Roadmap

- [x] Stage 1: synth
- [ ] Pipeline stages: teacher, dataset, train, eval, export
- [ ] Resolution sweep on `image.max_pixels`
- [ ] Shorter output format (field ids instead of labels)
- [ ] int8 / int4 quantization
- [ ] On-device test with LiteRT

## License

[MIT](LICENSE)
