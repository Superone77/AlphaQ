#!/usr/bin/env python3
"""
Run GPTQ on Qwen3-Coder-Next using an AlphaQ recipe CSV (name, bit_width).
Saves a quantize-then-dequantize model (same structure as original, float weights).
Requires GPU. Multi-GPU: use --device_map auto.

Usage:
  python scripts/03_gptq_from_recipe.py \\
    --model Qwen/Qwen3-Coder-Next \\
    --recipe_csv docs/bit_recipes/qwen3_coder_next_gamma10.0_bpp3.5.csv \\
    --output_dir ./out_qwen3_bpp35 \\
    --nsamples 128 --seqlen 2048 --device_map auto
  If OOM: use --calib_batch_size 16 (or 8) and/or --nsamples 64 --seqlen 1024
"""
import argparse
import csv
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from safetensors import safe_open
from gptq import GPTQ
from native_quant_utils import get_minimal_model_and_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_recipe(csv_path: str) -> dict:
    out = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip()
            bit = row.get("bit_width", "4").strip()
            if name and bit.isdigit():
                out[name] = int(bit)
    return out


def get_wikitext_calibration(nsamples: int, seqlen: int, tokenizer, device: str, seed: int = 42):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    n = trainenc.input_ids.shape[1]
    loader = []
    for _ in range(nsamples):
        i = random.randint(0, max(0, n - seqlen - 1))
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        loader.append((inp,))
    return loader


def run_gptq_qwen3_from_recipe(
    model,
    tokenizer,
    recipe: dict,
    trainloader,
    device_map: str,
    seqlen: int,
    blocksize: int = 128,
    percdamp: float = 0.01,
    groupsize: int = -1,
    actorder: bool = False,
    calib_batch_size: int = 32,
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    num_layers = len(layers)
    hidden_size = model.config.hidden_size

    logger.info("Building calibration inputs ...")
    all_input_ids = torch.cat([batch[0] for batch in trainloader], dim=0).cpu()
    # Use the model's own embed_tokens when it matches hidden_size (minimal/from_config model).
    # For real models load directly from safetensors to bypass slow accelerate offload hooks.
    _own_embed_w = model.model.embed_tokens.weight
    _index_file = Path(model.config._name_or_path) / "model.safetensors.index.json"
    if _own_embed_w.shape[1] == hidden_size:
        # Minimal model: embed weight already has the right hidden_size
        embed_weight = _own_embed_w.data.float().cpu()
        model_dtype = next(model.parameters()).dtype
    elif _index_file.exists():
        # Real model: load from safetensors to avoid accelerate offload overhead
        import json as _json
        with open(_index_file) as _f:
            _shard_file = _json.load(_f)["weight_map"]["model.embed_tokens.weight"]
        _shard_path = Path(model.config._name_or_path) / _shard_file
        with safe_open(str(_shard_path), framework="pt", device="cpu") as _st:
            embed_weight = _st.get_tensor("model.embed_tokens.weight").float()
        model_dtype = next((p for p in model.parameters() if p.device.type != "meta"), None)
        model_dtype = model_dtype.dtype if model_dtype is not None else torch.bfloat16
    else:
        embed_weight = _own_embed_w.data.float().cpu()
        model_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        all_embeds = nn.functional.embedding(all_input_ids, embed_weight).to(model_dtype)
    del embed_weight
    layer_inps = [all_embeds]
    nsamples = layer_inps[0].shape[0]
    seqlen_act = layer_inps[0].shape[1]
    first_dev = next(model.parameters()).device
    attention_mask = torch.ones(nsamples, seqlen_act, dtype=torch.bool, device="cpu")
    position_ids = torch.arange(seqlen_act, device="cpu").unsqueeze(0).expand(nsamples, -1)

    # Qwen3-Next decoder layers require position_embeddings (cos, sin) from RoPE
    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is not None:
        rotary_dev = next(rotary_emb.parameters()).device if len(list(rotary_emb.parameters())) > 0 else "cpu"
        with torch.no_grad():
            position_embeddings = rotary_emb(layer_inps[0].to(rotary_dev), position_ids.to(rotary_dev))
        position_embeddings = (position_embeddings[0].cpu(), position_embeddings[1].cpu())
    else:
        position_embeddings = None

    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        layer_dev = next(layer.parameters()).device
        full_inp = layer_inps[layer_idx]
        out_chunks = []
        for start in range(0, nsamples, calib_batch_size):
            end = min(start + calib_batch_size, nsamples)
            cur_inp = full_inp[start:end].to(layer_dev)
            attn = attention_mask[start:end].to(layer_dev)
            pos = position_ids[start:end].to(layer_dev)
            with torch.no_grad():
                if position_embeddings is not None:
                    cos = position_embeddings[0][start:end].to(layer_dev)
                    sin = position_embeddings[1][start:end].to(layer_dev)
                    out = layer(cur_inp, position_embeddings=(cos, sin), attention_mask=attn)
                else:
                    out = layer(cur_inp, position_embeddings=None, attention_mask=attn, position_ids=pos)
            out_chunks.append(out.cpu() if out.device.type != "cpu" else out.clone())
            del cur_inp, out
            if layer_dev.type == "cuda":
                torch.cuda.empty_cache()
        layer_inps.append(torch.cat(out_chunks, dim=0))
        if (layer_idx + 1) % 12 == 0:
            logger.info("  Calib layer %s/%s", layer_idx + 1, num_layers)

    def get_bit(name):
        return recipe.get(name)

    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        prefix = f"model.model.layers.{layer_idx}."
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "experts"):
            continue
        experts = layer.mlp.experts
        layer_dev = next(layers[layer_idx].parameters()).device
        mlp_inp = layer_inps[layer_idx].to(layer_dev)
        mlp_inp_flat = mlp_inp.reshape(-1, hidden_size).float()
        dev = next(layer.parameters()).device
        num_experts = len(experts)

        for e in range(num_experts):
            expert = experts[e]
            # Quantize gate_proj and up_proj
            for proj_name in ("gate_proj", "up_proj"):
                linear = getattr(expert, proj_name, None)
                if linear is None:
                    continue
                name = f"{prefix}mlp.experts.{e}.{proj_name}"
                bit = get_bit(name)
                if bit is None:
                    continue
                W = linear.weight.data.float().to(dev)
                wrapper = nn.Linear(W.shape[1], W.shape[0], bias=False, device=dev)
                wrapper.weight.data = W.clone()
                gptq = GPTQ(wrapper, logger, name, bit)
                gptq.quantizer.configure(bit, perchannel=True, sym=True, mse=False, pack=False)
                out_flat = mlp_inp_flat.to(dev) @ W.t()
                gptq.add_batch(mlp_inp_flat.to(dev), out_flat)
                gptq.fasterquant(blocksize=blocksize, percdamp=percdamp, groupsize=groupsize, actorder=actorder, name=name)
                linear.weight.data = wrapper.weight.data.to(linear.weight.dtype).to(linear.weight.device)
                gptq.free()
            # Quantize down_proj (needs gate/up activations as input)
            down_linear = getattr(expert, "down_proj", None)
            if down_linear is not None:
                name_d = f"{prefix}mlp.experts.{e}.down_proj"
                bit = get_bit(name_d)
                if bit is not None:
                    inp = mlp_inp_flat.to(dev)
                    W_gate = expert.gate_proj.weight.data.float().to(dev)
                    W_up = expert.up_proj.weight.data.float().to(dev)
                    gate_out = torch.nn.functional.silu(inp @ W_gate.t())
                    up_out = inp @ W_up.t()
                    mid = gate_out * up_out
                    W_d = down_linear.weight.data.float().to(dev)
                    wrapper = nn.Linear(W_d.shape[1], W_d.shape[0], bias=False, device=dev)
                    wrapper.weight.data = W_d.clone()
                    gptq = GPTQ(wrapper, logger, name_d, bit)
                    gptq.quantizer.configure(bit, perchannel=True, sym=True, mse=False, pack=False)
                    out_flat = mid @ W_d.t()
                    gptq.add_batch(mid, out_flat)
                    gptq.fasterquant(blocksize=blocksize, percdamp=percdamp, groupsize=groupsize, actorder=actorder, name=name_d)
                    down_linear.weight.data = wrapper.weight.data.to(down_linear.weight.dtype).to(down_linear.weight.device)
                    gptq.free()

        if (layer_idx + 1) % 6 == 0:
            logger.info("  GPTQ experts layer %s/%s", layer_idx + 1, num_layers)
        if layer_idx + 1 < num_layers:
            layer = layer.to(dev)
            next_chunks = []
            for start in range(0, nsamples, calib_batch_size):
                end = min(start + calib_batch_size, nsamples)
                cur = layer_inps[layer_idx][start:end].to(dev)
                attn = attention_mask[start:end].to(dev)
                with torch.no_grad():
                    if position_embeddings is not None:
                        cos = position_embeddings[0][start:end].to(dev)
                        sin = position_embeddings[1][start:end].to(dev)
                        next_inp = layer(cur, position_embeddings=(cos, sin), attention_mask=attn)
                    else:
                        next_inp = layer(cur, position_embeddings=None, attention_mask=attn, position_ids=position_ids[start:end].to(dev))
                next_chunks.append(next_inp.cpu() if next_inp.device.type != "cpu" else next_inp.clone())
                del cur, next_inp
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            layer_inps[layer_idx + 1] = torch.cat(next_chunks, dim=0)
        layer_inps[layer_idx] = None  # free CPU RAM
        layers[layer_idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    return model


def main():
    p = argparse.ArgumentParser(description="Qwen3-Coder-Next GPTQ from AlphaQ recipe CSV")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-Coder-Next")
    p.add_argument("--from_config", action="store_true",
                   help="Build minimal model from config instead of loading real weights (for 2-layer testing)")
    p.add_argument("--max_layers", type=int, default=2, help="Number of layers when --from_config")
    p.add_argument("--recipe_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--calib_batch_size", type=int, default=32,
                   help="Process this many samples per layer at a time to reduce GPU memory (default 32)")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--blocksize", type=int, default=128)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--groupsize", type=int, default=-1)
    p.add_argument("--act_order", action="store_true")
    args = p.parse_args()

    recipe_path = Path(args.recipe_csv)
    if not recipe_path.is_absolute():
        recipe_path = ROOT / recipe_path
    if not recipe_path.exists():
        logger.error("Recipe not found: %s", recipe_path)
        return 1
    recipe = load_recipe(str(recipe_path))
    logger.info("Loaded recipe: %s entries from %s", len(recipe), recipe_path)

    if args.from_config:
        logger.info("Building minimal model from config (%s layers) ...", args.max_layers)
        model, tokenizer = get_minimal_model_and_tokenizer(max_layers=args.max_layers, model=args.model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    else:
        logger.info("Loading model: %s (device_map=%s)", args.model, args.device_map)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
            device_map=args.device_map,
            trust_remote_code=True,
        )

    trainloader = get_wikitext_calibration(args.nsamples, args.seqlen, tokenizer, str(next(model.parameters()).device))
    logger.info("Calibration: %s samples, seqlen=%s", len(trainloader), args.seqlen)

    run_gptq_qwen3_from_recipe(
        model,
        tokenizer,
        recipe,
        trainloader,
        args.device_map,
        args.seqlen,
        blocksize=args.blocksize,
        percdamp=args.percdamp,
        groupsize=args.groupsize,
        actorder=args.act_order,
        calib_batch_size=args.calib_batch_size,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving quantize-then-dequantize model ...")
    # Truncate layer_types in config to match actual num_hidden_layers (avoids cache IndexError on reload)
    if getattr(model.config, "layer_types", None) is not None:
        model.config.layer_types = model.config.layer_types[:model.config.num_hidden_layers]
    # Patch broken deepspeed import in accelerate's extract_model_from_parallel before calling save_pretrained
    import accelerate.utils.other as _auo
    _orig_extract = _auo.extract_model_from_parallel
    _auo.extract_model_from_parallel = lambda m, **kw: m
    try:
        model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="10GB")
    finally:
        _auo.extract_model_from_parallel = _orig_extract
    tokenizer.save_pretrained(out_dir)
    logger.info("Done. Load with AutoModelForCausalLM.from_pretrained(%s)", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
