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

from gptq import GPTQ

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
    embeds = []
    for batch in trainloader:
        inp = batch[0]
        embeds.append(model.model.embed_tokens(inp))
    layer_inps = [torch.cat(embeds, dim=0)]
    nsamples = layer_inps[0].shape[0]
    seqlen_act = layer_inps[0].shape[1]
    first_dev = next(model.parameters()).device
    attention_mask = torch.ones(nsamples, seqlen_act, dtype=torch.long, device=first_dev)
    position_ids = torch.arange(seqlen_act, device=first_dev).unsqueeze(0).expand(nsamples, -1)

    # Qwen3-Next decoder layers require position_embeddings (cos, sin) from RoPE
    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is not None:
        with torch.no_grad():
            position_embeddings = rotary_emb(layer_inps[0].to(first_dev), position_ids)
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
                    out = layer(cur_inp, (cos, sin), attention_mask=attn)[0]
                else:
                    out = layer(cur_inp, attention_mask=attn, position_ids=pos)[0]
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
        mlp_inp_flat = mlp_inp.reshape(-1, hidden_size)
        gate_up = experts.gate_up_proj
        down = experts.down_proj
        inter = gate_up.shape[1] // 2
        dev = next(layer.parameters()).device

        for e in range(gate_up.shape[0]):
            name_gate = f"{prefix}mlp.experts.gate_up_proj.expert_{e}.gate"
            name_up = f"{prefix}mlp.experts.gate_up_proj.expert_{e}.up"
            name_gu = f"{prefix}mlp.experts.gate_up_proj.expert_{e}"
            bit_gate = get_bit(name_gate)
            bit_up = get_bit(name_up)
            bit_whole = get_bit(name_gu)
            if bit_gate is not None and bit_up is not None:
                W_full = gate_up.data[e].float().clone().to(dev)
                W_gate = W_full[:inter]
                W_up = W_full[inter:]
                parts = []
                for part_name, W_part, bit in [(name_gate, W_gate, bit_gate), (name_up, W_up, bit_up)]:
                    wrapper = nn.Linear(W_part.shape[1], W_part.shape[0], bias=False, device=dev)
                    wrapper.weight.data = W_part.t().clone()
                    gptq = GPTQ(wrapper, logger, part_name, bit)
                    gptq.quantizer.configure(bit, perchannel=True, sym=True, mse=False, pack=False)
                    out_flat = mlp_inp_flat @ W_part.t()
                    gptq.add_batch(mlp_inp_flat, out_flat)
                    gptq.fasterquant(blocksize=blocksize, percdamp=percdamp, groupsize=groupsize, actorder=actorder, name=part_name)
                    q_w = wrapper.weight.data.t().clone()
                    parts.append(q_w)
                    gptq.free()
                experts.gate_up_proj.data[e] = torch.cat(parts, dim=0).to(experts.gate_up_proj.dtype).to(experts.gate_up_proj.device)
            elif bit_whole is not None:
                W = gate_up.data[e].float().clone().to(dev)
                wrapper = nn.Linear(gate_up.shape[2], gate_up.shape[1], bias=False, device=dev)
                wrapper.weight.data = W.t().clone()
                gptq = GPTQ(wrapper, logger, name_gu, bit_whole)
                gptq.quantizer.configure(bit_whole, perchannel=True, sym=True, mse=False, pack=False)
                out_flat = mlp_inp_flat @ W.t()
                gptq.add_batch(mlp_inp_flat, out_flat)
                gptq.fasterquant(blocksize=blocksize, percdamp=percdamp, groupsize=groupsize, actorder=actorder, name=name_gu)
                q_w = wrapper.weight.data.t().clone()
                experts.gate_up_proj.data[e] = q_w.to(experts.gate_up_proj.dtype).to(experts.gate_up_proj.device)
                gptq.free()

        for e in range(down.shape[0]):
            name_d = f"{prefix}mlp.experts.down_proj.expert_{e}"
            bit = get_bit(name_d)
            if bit is None:
                continue
            W_gu = experts.gate_up_proj.data[e].float().to(dev)
            out1 = mlp_inp_flat @ W_gu.t()
            gate_part = out1[:, :inter]
            up_part = out1[:, inter:]
            mid = torch.nn.functional.silu(up_part) * gate_part
            W_d = down.data[e].float().clone().to(dev)
            wrapper = nn.Linear(inter, down.shape[2], bias=False, device=dev)
            wrapper.weight.data = W_d.t().clone()
            gptq = GPTQ(wrapper, logger, name_d, bit)
            gptq.quantizer.configure(bit, perchannel=True, sym=True, mse=False, pack=False)
            out_flat = mid @ W_d.t()
            gptq.add_batch(mid, out_flat)
            gptq.fasterquant(blocksize=blocksize, percdamp=percdamp, groupsize=groupsize, actorder=actorder, name=name_d)
            q_w = wrapper.weight.data.t().clone()
            experts.down_proj.data[e] = q_w.to(experts.down_proj.dtype).to(experts.down_proj.device)
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
                        next_inp = layer(cur, (cos, sin), attention_mask=attn)[0]
                    else:
                        next_inp = layer(cur, attention_mask=attn, position_ids=position_ids[start:end].to(dev))[0]
                next_chunks.append(next_inp.cpu() if next_inp.device.type != "cpu" else next_inp.clone())
                del cur, next_inp
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            layer_inps[layer_idx + 1] = torch.cat(next_chunks, dim=0)
        layers[layer_idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    return model


def main():
    p = argparse.ArgumentParser(description="Qwen3-Coder-Next GPTQ from AlphaQ recipe CSV")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-Coder-Next")
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
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    logger.info("Done. Load with AutoModelForCausalLM.from_pretrained(%s)", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
