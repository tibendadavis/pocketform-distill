"""Student smoke test. See README."""

import os

# Must be set before importing torch: runs ops without an MPS kernel on CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import time

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from pocketform_distill.config import load_form
from pocketform_distill.prompt import build_prompt


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--form", default="forms/_template", help="form folder; the prompt is built from its spec")
    parser.add_argument("--section", help="section the photo shows (default: the form's only section)")
    args = parser.parse_args()
    spec = load_form(args.form)
    prompt = build_prompt(spec, spec.section(args.section))
    image_path, model_id = args.image_path, args.model
    device = pick_device(args.device)
    dtype = pick_dtype(device)
    print(f"torch {torch.__version__} | device {device} | dtype {dtype}")

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype, device_map=device)
    model.eval()
    print(f"Loaded {model_id} in {time.time() - t0:.1f}s")
    print(f"Weights in memory: {model.get_memory_footprint() / 1e9:.1f} GB")
    if device == "cuda":
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    elif device == "mps":
        print(f"MPS memory allocated: {torch.mps.driver_allocated_memory() / 1e9:.1f} GB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    print(f"Prompt length (text + image tokens): {prompt_len}")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    elapsed = time.time() - t0

    new_tokens = out.sequences[0, prompt_len:]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    print(f"\n--- Model output ({len(new_tokens)} tokens, {elapsed:.1f}s, "
          f"{len(new_tokens) / elapsed:.1f} tok/s) ---\n{text}\n")

    print("--- Top-5 token probabilities for first 5 output tokens ---")
    for step, logits in enumerate(out.scores[:5]):
        probs = torch.softmax(logits[0].float(), dim=-1)
        top = torch.topk(probs, 5)
        pretty = ", ".join(
            f"{processor.decode([int(i)])!r}:{p:.2f}" for p, i in zip(top.values, top.indices)
        )
        print(f"  step {step}: {pretty}")


if __name__ == "__main__":
    main()
