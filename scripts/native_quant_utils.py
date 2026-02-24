# Helpers for native (symmetric per-tensor) quantization from recipe CSV — no GPTQ.
# Used by 04 and 05 when running with --recipe_csv (e.g. when SKIP_GPTQ=1).
import csv
import re
from pathlib import Path
from typing import Dict, Tuple

import torch


def load_recipe_expert_bits(csv_path: Path) -> Dict[Tuple[int, str, int], int]:
    """
    Load recipe CSV (name, bit_width); return dict (layer_idx, proj_key, expert_idx) -> bit_width.
    proj_key: 'gate_up_proj', 'gate_up_proj_gate', 'gate_up_proj_up', or 'down_proj'.
    """
    out = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip()
            bit = row.get("bit_width", "4").strip()
            if not name or not bit.isdigit():
                continue
            bit = int(bit)
            m = re.match(
                r"model\.model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)\.expert_(\d+)(?:\.(gate|up))?",
                name,
            )
            if m:
                layer_idx = int(m.group(1))
                proj = m.group(2)
                expert_idx = int(m.group(3))
                half = m.group(4)
                if proj == "gate_up_proj" and half:
                    proj_key = f"gate_up_proj_{half}"
                else:
                    proj_key = proj
                out[(layer_idx, proj_key, expert_idx)] = bit
    return out


def quantize_symmetric(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-tensor quantization to `bits`. Returns dequantized float."""
    w = w.float()
    if bits <= 0:
        return w
    if bits == 1:
        scale = w.abs().mean().clamp(min=1e-12)
        q = (w >= 0).to(w.dtype) * 2 - 1
        return (q * scale).to(w.device)
    maxq = float(2 ** (bits - 1) - 1)
    scale = w.abs().max() / maxq
    scale = scale.clamp(min=1e-12)
    q = torch.round(w / scale).clamp(-maxq, maxq)
    return (q.float() * scale).to(w.device)


def apply_recipe_to_model(model, recipe: Dict[Tuple[int, str, int], int], device: str = "cpu"):
    """Apply recipe (native symmetric quant) to MoE experts in-place. Supports gate/up separate."""
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "experts"):
            continue
        experts = layer.mlp.experts
        gate_up = getattr(experts, "gate_up_proj", None)
        if gate_up is not None:
            inter = gate_up.shape[1] // 2
            for e in range(gate_up.shape[0]):
                bit_gate = recipe.get((layer_idx, "gate_up_proj_gate", e))
                bit_up = recipe.get((layer_idx, "gate_up_proj_up", e))
                bit_whole = recipe.get((layer_idx, "gate_up_proj", e))
                if bit_gate is not None and bit_up is not None:
                    w = gate_up.data[e].float().to(device)
                    gate_up.data[e] = torch.cat([
                        quantize_symmetric(w[:inter], bit_gate),
                        quantize_symmetric(w[inter:], bit_up),
                    ], dim=0).to(gate_up.dtype).to(gate_up.device)
                elif bit_whole is not None:
                    w = gate_up.data[e].float().to(device)
                    gate_up.data[e] = quantize_symmetric(w, bit_whole).to(gate_up.dtype).to(gate_up.device)
        down = getattr(experts, "down_proj", None)
        if down is not None:
            for e in range(down.shape[0]):
                bit = recipe.get((layer_idx, "down_proj", e))
                if bit is None:
                    continue
                w = down.data[e].float().to(device)
                down.data[e] = quantize_symmetric(w, bit).to(down.dtype).to(down.device)
    return model


def _patch_qwen3_cache_for_minimal_layers():
    """
    When num_hidden_layers is small (e.g. 2), config.layer_types may have no 'full_attention',
    so Qwen3NextDynamicCache.transformer_layers is empty and get_seq_length() hits
    IndexError on self.transformer_layers[0]. Patch: if transformer_layers is empty, return 0.
    """
    try:
        from transformers.models import qwen3_next
        mod = qwen3_next.modeling_qwen3_next
        cache_cls = getattr(mod, "Qwen3NextDynamicCache", None)
        if cache_cls is None:
            return
        _orig = cache_cls.get_seq_length

        def _get_seq_length(self, layer_idx=0):
            if not self.transformer_layers:
                return 0
            return _orig(self, layer_idx)

        cache_cls.get_seq_length = _get_seq_length
    except Exception:
        pass


def get_minimal_model_and_tokenizer(
    max_layers: int = 2,
    num_experts: int = 512,
    model: str = "Qwen/Qwen3-Coder-Next",
):
    """Build minimal Qwen3-Coder-Next from config (no weights download). Returns (model, tokenizer)."""
    _patch_qwen3_cache_for_minimal_layers()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    config.num_hidden_layers = max_layers
    # Truncate layer_types so cache init matches num_hidden_layers (avoids IndexError in has_previous_state)
    if getattr(config, "layer_types", None) is not None and len(config.layer_types) > max_layers:
        config.layer_types = config.layer_types[:max_layers]
    config.num_experts = num_experts
    config.hidden_size = 128
    config.intermediate_size = 256
    config.moe_intermediate_size = 64
    config.shared_expert_intermediate_size = 64
    config.num_attention_heads = 2
    config.num_key_value_heads = 1
    config.linear_num_key_heads = 2
    config.linear_num_value_heads = 2
    # Keep vocab_size from pretrained so tokenizer and embed_tokens match
    minimal_model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return minimal_model, tokenizer
