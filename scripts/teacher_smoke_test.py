"""Teacher smoke test. See README."""

import argparse
import math
import time

import ollama
from transformers import AutoTokenizer

from pocketform_distill.token_mapping import to_student_id

PROMPT = (
    "This is a photo of a handwritten form. "
    "Extract every filled-in field as a JSON object mapping field label to value. "
    "Use null for empty fields. Return only the JSON."
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image_path")
    p.add_argument("--teacher", required=True, help="Ollama model tag, e.g. from `ollama list`")
    p.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    t0 = time.time()
    resp = ollama.chat(
        model=args.teacher,
        messages=[{"role": "user", "content": PROMPT, "images": [args.image_path]}],
        think=False,
        logprobs=True,
        top_logprobs=args.top,
        options={"temperature": 0},
    )
    elapsed = time.time() - t0
    print(f"--- Teacher output ({elapsed:.1f}s) ---\n{resp['message']['content']}\n")
    if resp.get("eval_count") and resp.get("eval_duration"):
        print(f"Speed: {resp['eval_count']} tokens, {resp['eval_count'] / (resp['eval_duration'] / 1e9):.1f} tok/s\n")

    logprobs = resp.get("logprobs") or []
    if not logprobs:
        raise SystemExit("No logprobs returned - update Ollama (>= 0.12.11) and check the model supports it.")

    print("--- Top-5 alternatives for first 5 tokens ---")
    for i, e in enumerate(logprobs[:5]):
        alts = ", ".join(f"{a['token']!r}:{math.exp(a['logprob']):.2f}" for a in (e.get("top_logprobs") or [])[:5])
        print(f"  {i}: chosen {e['token']!r} | {alts}")

    tok = AutoTokenizer.from_pretrained(args.student)
    total = mapped = 0
    unmapped = []
    for e in logprobs:
        for cand in [e, *(e.get("top_logprobs") or [])]:
            total += 1
            if to_student_id(cand, tok) is not None:
                mapped += 1
            elif len(unmapped) < 10:
                unmapped.append(cand["token"])
    pct = 100 * mapped / total
    print(f"\n--- Tokenizer check vs {args.student} ---")
    print(f"{mapped}/{total} teacher tokens map to a single student token ({pct:.1f}%)")
    if unmapped:
        print(f"Examples that didn't map: {unmapped}")
    print("PASS - Ollama route is viable" if pct > 99 else "CHECK - mapping gaps need investigating before building on this")

    spread = sum(1 for e in logprobs if e.get("top_logprobs") and math.exp(e["top_logprobs"][0]["logprob"]) < 0.99)
    print(f"{spread}/{len(logprobs)} positions where the teacher was < 99% sure (should be > 0)")


if __name__ == "__main__":
    main()
