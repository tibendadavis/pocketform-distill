# pocketform-distill

Teach a phone-sized vision-language model to read one paper form layout by distilling it from a larger model.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![CI](https://github.com/tibendadavis/pocketform-distill/actions/workflows/ci.yml/badge.svg)](https://github.com/tibendadavis/pocketform-distill/actions/workflows/ci.yml)

> **Status: early.** Setup and the student and teacher smoke tests work. The pipeline stages are not built yet.

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
configs/_template.yaml          run config template: one form + teacher + student + targets
forms/_template/form.yaml       form spec template (fictional form)
scripts/smoke_test.py           student smoke test
scripts/teacher_smoke_test.py   teacher smoke test
src/pocketform_distill/         the package (CLI: `pocketform-distill`), incl. teacher→student token mapping
tests/                          offline unit tests (no model downloads)
private/, data/                 real photos and stage outputs (git-ignored)
```

## Configuration

To use your own form, copy `forms/_template/` to `forms/<your-form>/` and `configs/_template.yaml` to
`configs/<your-run>.yaml`, then edit both. Your own form folders and run configs are git-ignored by default. To publish
one on purpose, add `!forms/<your-form>/` and `!configs/<your-run>.yaml` to `.gitignore`.

**Form spec** (`forms/<name>/form.yaml`)

| Key | Meaning |
|-----|---------|
| `name`, `description` | Identifies the form |
| `template_image` | Blank scan or photo of the form next to `form.yaml` (only `template.png` is committed) |
| `fields[].id` | Unique key used in outputs |
| `fields[].label` | Text as printed on the form |
| `fields[].type` | `text` (any handwriting), `integer`, `date`, `choice` |
| `fields[].options` | Allowed values for `choice` fields |
| `fields[].min`, `max` | Optional range for `integer` fields, used to generate realistic synthetic values |

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

- [ ] Pipeline stages: synth, teacher, dataset, train, eval, export
- [ ] Resolution sweep on `image.max_pixels`
- [ ] Shorter output format (field ids instead of labels)
- [ ] int8 / int4 quantization
- [ ] On-device test with LiteRT

## License

[MIT](LICENSE)
