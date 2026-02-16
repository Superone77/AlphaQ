#!/usr/bin/env python3
"""
Compute AlphaQ alpha and variance for Qwen3-Coder-Next (all layers or first N).
Outputs CSV with columns: name, layer, module_type, alpha, variance.
Used as input for 02_precision_solve.py.

Usage:
  python scripts/01_compute_alpha.py --output_csv docs/alpha_qwen3_full.csv
  python scripts/01_compute_alpha.py --output_csv docs/alpha_qwen3_full.csv --max_layers 2  # quick test
"""
import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

# Repo root = parent of scripts/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ALPHA_MODE", "FARMS")

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from alphaq.utils_alpha import alpha_hill_from_weight


def get_minimal_config(num_layers=48, num_experts=512):
    config = AutoConfig.from_pretrained("Qwen/Qwen3-Coder-Next", trust_remote_code=True)
    config.num_hidden_layers = num_layers
    config.num_experts = num_experts
    config.hidden_size = 128
    config.intermediate_size = 256
    config.moe_intermediate_size = 64
    config.shared_expert_intermediate_size = 64
    config.num_attention_heads = 2
    config.num_key_value_heads = 1
    config.linear_num_key_heads = 2
    config.linear_num_value_heads = 2
    config.vocab_size = 1000
    return config


def compute_alpha_and_variance(weight: torch.Tensor, use_farms: bool = True):
    w = weight.detach().to(dtype=torch.float32, device="cpu")
    if w.ndim > 2:
        w = w.reshape(w.shape[0], -1)
    if min(w.shape) < 2:
        return float("nan"), float("nan")
    try:
        alpha, _, _ = alpha_hill_from_weight(w, use_farms=use_farms)
        var = float(torch.var(w, unbiased=False).item())
        return float(alpha), var
    except Exception:
        return float("nan"), float("nan")


def collect_alpha_first_n_layers(model, n_layers: int):
    results = []
    layers = model.model.layers
    for i in range(min(n_layers, len(layers))):
        layer = layers[i]
        prefix = f"model.model.layers.{i}."
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                weight = getattr(module, "weight", None)
                if weight is None:
                    continue
                full_name = prefix + name
                alpha, var = compute_alpha_and_variance(weight)
                results.append({"name": full_name, "alpha": alpha, "variance": var})
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            gate = layer.mlp.gate
            w = getattr(gate, "weight", None)
            if w is not None and isinstance(w, torch.Tensor) and w.ndim >= 2:
                full_name = f"{prefix}mlp.gate.weight"
                alpha, var = compute_alpha_and_variance(w)
                results.append({"name": full_name, "alpha": alpha, "variance": var})
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            experts = layer.mlp.experts
            if hasattr(experts, "gate_up_proj") and experts.gate_up_proj is not None:
                gate_up = experts.gate_up_proj
                inter = gate_up.shape[1] // 2
                for e in range(gate_up.shape[0]):
                    w = gate_up[e]
                    w_gate = w[:inter]
                    w_up = w[inter:]
                    alpha_g, var_g = compute_alpha_and_variance(w_gate)
                    results.append({"name": f"{prefix}mlp.experts.gate_up_proj.expert_{e}.gate", "alpha": alpha_g, "variance": var_g})
                    alpha_u, var_u = compute_alpha_and_variance(w_up)
                    results.append({"name": f"{prefix}mlp.experts.gate_up_proj.expert_{e}.up", "alpha": alpha_u, "variance": var_u})
            if hasattr(experts, "down_proj") and experts.down_proj is not None:
                for e in range(experts.down_proj.shape[0]):
                    w = experts.down_proj[e]
                    full_name = f"{prefix}mlp.experts.down_proj.expert_{e}"
                    alpha, var = compute_alpha_and_variance(w)
                    results.append({"name": full_name, "alpha": alpha, "variance": var})
    return results


def get_module_type(name: str) -> str:
    if "linear_attn." in name:
        return "linear_attn"
    if "self_attn." in name:
        return "self_attn"
    if ".mlp.gate.weight" in name:
        return "mlp.gate"
    if "mlp.experts.gate_up_proj" in name:
        if name.endswith(".gate"):
            return "mlp.experts.gate_up_proj.gate"
        if name.endswith(".up"):
            return "mlp.experts.gate_up_proj.up"
        return "mlp.experts.gate_up_proj"
    if "mlp.experts.down_proj" in name:
        return "mlp.experts.down_proj"
    if "mlp.shared_expert." in name:
        return "mlp.shared_expert"
    return "other"


def extract_layer_id(name: str) -> int:
    m = re.match(r"model\.model\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def write_csv(results: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "layer", "module_type", "alpha", "variance"])
        for r in results:
            w.writerow([
                r["name"],
                extract_layer_id(r["name"]),
                get_module_type(r["name"]),
                r["alpha"] if math.isfinite(r["alpha"]) else "",
                r["variance"] if math.isfinite(r["variance"]) else "",
            ])
    print(f"Wrote {output_path} ({len(results)} rows)")


def main():
    p = argparse.ArgumentParser(description="Compute AlphaQ alpha/variance for Qwen3-Coder-Next")
    p.add_argument("--output_csv", type=str, default="docs/alpha_qwen3_full.csv", help="Output CSV path")
    p.add_argument("--max_layers", type=int, default=48, help="Max decoder layers to process (default 48)")
    p.add_argument("--num_experts", type=int, default=512, help="Number of experts (default 512)")
    args = p.parse_args()

    output_path = Path(args.output_csv)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    num_layers = args.max_layers
    num_experts = args.num_experts
    print(f"Building Qwen3-Coder-Next from config ({num_layers} layers, {num_experts} experts) ...")
    config = get_minimal_config(num_layers=num_layers, num_experts=num_experts)
    model = AutoModelForCausalLM.from_config(config)
    model = model.to("cpu")
    model.eval()

    print(f"Computing alpha for {num_layers} layers ...")
    results = collect_alpha_first_n_layers(model, n_layers=num_layers)
    print(f"Got {len(results)} entries.")
    write_csv(results, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
