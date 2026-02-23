#!/usr/bin/env python3
"""
Evaluate WikiText-2 perplexity for a Qwen3-Coder-Next model (original, quantized dir, or native quant from recipe).

Usage:
  # Saved model (from 03 GPTQ or any HF path)
  python scripts/04_eval_quantized.py --model_path ./out_qwen3_bpp35 --seqlen 2048

  # Native quantization (no GPTQ): load model then apply recipe in memory
  python scripts/04_eval_quantized.py --model_path Qwen/Qwen3-Coder-Next --recipe_csv docs/bit_recipes/recipe.csv
  python scripts/04_eval_quantized.py --from_config --max_layers 2 --recipe_csv docs/bit_recipes_2layer/recipe.csv
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

from native_quant_utils import (
    load_recipe_expert_bits,
    apply_recipe_to_model,
    get_minimal_model_and_tokenizer,
)


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
            # Causal LM: logits[i] predicts chunk[i+1]; align length with labels (both seqlen)
            seq_len = logits.size(1)
            shift_logits = logits.contiguous().view(-1, logits.size(-1))
            shift_labels = chunk[:, 1 : 1 + seq_len].contiguous().view(-1)
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
    p.add_argument("--model_path", type=str, default=None, help="HF model id or local dir. Not used if --from_config.")
    p.add_argument("--from_config", action="store_true", help="Build minimal model from config (use with --max_layers).")
    p.add_argument("--max_layers", type=int, default=2, help="When --from_config: number of decoder layers.")
    p.add_argument("--recipe_csv", type=str, default=None, help="Apply native (symmetric) quant from recipe in memory before eval.")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max_nsamples", type=int, default=0, help="0 = use full test set")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--device_map", type=str, default=None)
    args = p.parse_args()

    if not args.from_config and not args.model_path:
        p.error("Provide --model_path or --from_config")

    use_device_map = args.device_map if args.device_map is not None else ("auto" if torch.cuda.is_available() else None)
    if use_device_map is None and args.device:
        use_device_map = args.device
    # For from_config we need a concrete device to .to(device); "auto" is not a torch device
    if use_device_map is None or use_device_map == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = use_device_map

    if args.from_config:
        print(f"Building minimal model from config ({args.max_layers} layers) ...")
        model, tokenizer = get_minimal_model_and_tokenizer(max_layers=args.max_layers)
        model = model.to(device)
    else:
        print(f"Loading tokenizer and model from {args.model_path} (device_map={use_device_map}) ...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            device_map=use_device_map,
            trust_remote_code=True,
        )
        if use_device_map is None:
            model = model.to("cpu")
    model.eval()

    if args.recipe_csv:
        recipe_path = Path(args.recipe_csv)
        if not recipe_path.is_absolute():
            recipe_path = ROOT / recipe_path
        print(f"Applying native quantization from recipe {recipe_path} ...")
        recipe = load_recipe_expert_bits(recipe_path)
        apply_recipe_to_model(model, recipe, device=str(device))
        print("  Native quant applied.")

    print("Loading WikiText-2 test ...")
    text = get_wikitext2_test(args.max_nsamples)
    device = next(model.parameters()).device
    eval_device = next(model.parameters()).device
    ppl = eval_ppl(model, tokenizer, text, args.seqlen, str(eval_device), args.max_nsamples)
    print(f"WikiText-2 PPL: {ppl:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
