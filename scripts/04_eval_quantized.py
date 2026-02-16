#!/usr/bin/env python3
"""
Evaluate WikiText-2 perplexity for a Qwen3-Coder-Next model (original or quantized dir).
Use after 03_gptq_from_recipe.py to measure PPL of the saved model.

Usage:
  python scripts/04_eval_quantized.py --model_path ./out_qwen3_bpp35 --seqlen 2048
  python scripts/04_eval_quantized.py --model_path Qwen/Qwen3-Coder-Next --seqlen 2048 --max_nsamples 0
"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def get_wikitext2_test(max_nsamples: int = 0):
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(data["text"])
    if max_nsamples and max_nsamples > 0:
        # approximate: take first max_nsamples * 1024 chars as one chunk
        text = text[: max_nsamples * 1024]
    return text


def eval_ppl(model, tokenizer, text: str, seqlen: int, device: str, max_nsamples: int = 0):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024 * 256)
    input_ids = enc.input_ids.to(device)
    if input_ids.shape[1] < seqlen + 1:
        print("Warning: text too short for seqlen")
        return float("nan")
    nll = 0.0
    n_tokens = 0
    stride = seqlen
    n_chunks = 0
    max_chunks = max_nsamples if max_nsamples > 0 else (input_ids.shape[1] - 1) // stride
    if max_nsamples > 0 and max_chunks > max_nsamples:
        max_chunks = max_nsamples
    for start in range(0, input_ids.shape[1] - seqlen - 1, stride):
        if n_chunks >= max_chunks and max_nsamples > 0:
            break
        chunk = input_ids[:, start : start + seqlen + 1]
        if chunk.shape[1] < seqlen + 1:
            break
        with torch.no_grad():
            logits = model(chunk[:, :-1]).logits
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = chunk[:, 1:].contiguous().view(-1)
            loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction="sum", ignore_index=tokenizer.pad_token_id or -100
            )
        nll += loss.item()
        n_tokens += (chunk[:, 1:] != (tokenizer.pad_token_id or -100)).sum().item()
        n_chunks += 1
    if n_tokens == 0:
        return float("nan")
    ppl = math.exp(nll / n_tokens)
    return ppl


def main():
    p = argparse.ArgumentParser(description="Eval WikiText-2 PPL for Qwen3-Coder-Next")
    p.add_argument("--model_path", type=str, required=True, help="HF model id or local dir (e.g. ./out_qwen3_bpp35)")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max_nsamples", type=int, default=0, help="0 = use full test set")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Loading tokenizer and model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=args.device if args.device != "cpu" else None,
        trust_remote_code=True,
    )
    if args.device == "cpu":
        model = model.to("cpu")
    model.eval()

    print("Loading WikiText-2 test ...")
    text = get_wikitext2_test(args.max_nsamples)
    device = args.device
    if hasattr(model, "device"):
        device = next(model.parameters()).device
    ppl = eval_ppl(model, tokenizer, text, args.seqlen, str(device), args.max_nsamples)
    print(f"WikiText-2 PPL: {ppl:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
