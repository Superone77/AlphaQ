#!/usr/bin/env python3
"""
Inference with Qwen3-Coder-Next: original HF model, quantized dir, or minimal model from config.

Usage:
  python scripts/05_inference.py --model_path ./out_qwen3_bpp35 --prompt "Your prompt" --max_new_tokens 64
  python scripts/05_inference.py --model_path Qwen/Qwen3-Coder-Next --prompt "Hello" --max_new_tokens 64
  python scripts/05_inference.py --from_config --max_layers 2 --prompt "Hello"   # 2-layer original (no weights download)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from native_quant_utils import get_minimal_model_and_tokenizer


def main():
    p = argparse.ArgumentParser(description="Inference with Qwen3-Coder-Next (original or GPTQ-roundtrip)")
    p.add_argument("--model_path", type=str, default=None, help="HuggingFace model id or local dir. Not used if --from_config.")
    p.add_argument("--from_config", action="store_true", help="Build minimal model from config (e.g. 2-layer original).")
    p.add_argument("--max_layers", type=int, default=2, help="When --from_config: number of decoder layers.")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--prompt", type=str, default="The meaning of life is")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    if not args.from_config and not args.model_path:
        p.error("Provide --model_path or --from_config")

    if args.from_config:
        print(f"Building minimal model from config ({args.max_layers} layers) ...")
        model, tokenizer = get_minimal_model_and_tokenizer(max_layers=args.max_layers)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    else:
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
    model_dev = next(model.parameters()).device
    if model_dev.type == "cuda":
        inputs = {k: v.to(model_dev) for k, v in inputs.items()}

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
