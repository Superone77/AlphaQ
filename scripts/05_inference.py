#!/usr/bin/env python3
"""
Inference with Qwen3-Coder-Next: supports original HF model or quantized dir from 03_gptq_from_recipe.py.
Same API: from_pretrained(model_path).

Usage:
  python scripts/05_inference.py --model_path ./out_qwen3_bpp35 --prompt "Your prompt" --max_new_tokens 64
  python scripts/05_inference.py --model_path Qwen/Qwen3-Coder-Next --prompt "Hello" --max_new_tokens 64
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description="Inference with Qwen3-Coder-Next (original or GPTQ-roundtrip)")
    p.add_argument("--model_path", type=str, required=True, help="HuggingFace model id or local dir")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--prompt", type=str, default="The meaning of life is")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    print(f"Loading tokenizer and model from {args.model_path} (device_map={args.device_map}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    if args.device_map == "auto" and next(model.parameters()).device.type == "cuda":
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("Output:", text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
