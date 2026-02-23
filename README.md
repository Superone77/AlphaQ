# AlphaQ — Mixed-Precision Quantization for Qwen3-Coder-Next

Based on the **AlphaQ** method from [Mixture-Compressor-MoE](https://github.com/Aaronhuang-778/Mixture-Compressor-MoE), this repository provides a full pipeline for **Qwen3-Coder-Next**: **precision solving (bit-width allocation) → GPTQ quantization → quantized model evaluation**, with standalone scripts and clear usage instructions.

## Overview

| Feature | Description | Script |
|---------|-------------|--------|
| **AlphaQ** | Calibration-free bit allocation: compute per-layer importance from weight spectra (PL Alpha Hill / FARMS), then solve MILP to minimize weighted quantization error under a bit budget | see below |
| **Precision solving** | Read alpha/variance CSV, solve MILP to get per-layer/per-expert bit assignment (recipe CSV) | `scripts/02_precision_solve.py` |
| **GPTQ** | Run block-wise GPTQ on Qwen3-Coder-Next MoE expert weights according to recipe CSV; save a quantize-then-dequantize float model | `scripts/03_gptq_from_recipe.py` |
| **Quantized model evaluation** | Evaluate PPL on WikiText-2 for mixed-precision or GPTQ models | `scripts/04_eval_quantized.py` |
| **Inference** | Run generative inference on the original model or the GPTQ output directory | `scripts/05_inference.py` |

Alpha values (input to precision solving) are computed by `scripts/01_compute_alpha.py`, which builds a minimal config-based Qwen3-Coder-Next and computes alpha/variance per layer, writing a CSV.

## Setup

```bash
# Python 3.10+ recommended
cd AlphaQ
pip install -r requirements.txt
```

Dependencies include: `torch`, `transformers`, `datasets`, `pulp`, `numpy`, `scipy`. For inference and evaluation only `torch` and `transformers` are required (+ `datasets` for PPL).

### Multi-GPU (model parallelism)

GPTQ (03), evaluation (04), and inference (05) support **multi-GPU via HuggingFace `device_map="auto"`**: the model is sharded across visible GPUs and each layer runs on its assigned device.

- **03_gptq_from_recipe.py**: `--device_map auto` (default). Calibration and quantization run per layer on that layer’s device; no code changes needed.
- **04_eval_quantized.py**: use `--device_map auto` for multi-GPU (default when CUDA is available). Override with `--device_map cuda:0` for a single GPU.
- **05_inference.py**: `--device_map auto` (default). Inputs are moved to the model’s first device before `generate()`.

This is **model parallelism** (one model spread across GPUs), not data parallelism.

## One-command full pipeline

Two scripts run the full flow (01 → 02 → 03 → 04 → 05) from the repo root:

| Script | Model | Use case |
|--------|--------|----------|
| `scripts/run_full_pipeline_2layer.sh` | 2 layers only | Fast test (01–02 on CPU; 03–05 need GPU). Recipe covers 2 layers; 03 loads full model and quantizes only those layers’ experts. |
| `scripts/run_full_pipeline_full.sh` | Full 48 layers | Full run. 01 uses minimal config (48 layers); 03 quantizes all layers per recipe. |

```bash
# From AlphaQ repo root (GPU required for steps 3–5)
./scripts/run_full_pipeline_2layer.sh

# Full model (longer; 01 takes a while for 48 layers)
./scripts/run_full_pipeline_full.sh
```

Override defaults via env vars, e.g. `BPP=4.0 GAMMA=10.0 ./scripts/run_full_pipeline_full.sh`.

## Pipeline and Scripts

### 1. Compute Alpha (01_compute_alpha.py)

Computes **alpha** and **variance** for each layer of Qwen3-Coder-Next (including MoE expert gate/up/down slices) and writes a CSV for the precision solver.

```bash
# Default: full 48 layers (minimal config model, no full weights)
python scripts/01_compute_alpha.py --output_csv docs/alpha_qwen3_full.csv

# Quick test: first 2 layers only
python scripts/01_compute_alpha.py --output_csv docs/alpha_qwen3_full.csv --max_layers 2
```

- **Input**: HuggingFace model name or local path (`--model`, default `Qwen/Qwen3-Coder-Next`). The script uses a minimal config for speed.
- **Output**: CSV with columns `name, layer, module_type, alpha, variance`. `name` matches recipe naming (e.g. `model.model.layers.0.mlp.experts.gate_up_proj.expert_0.gate`).

### 2. Precision solving (02_precision_solve.py)

Reads the alpha CSV and solves MILP for per-layer/per-expert **bit_width** under a given **average bit budget (bpp)** and **gamma**, producing a recipe CSV.

```bash
# Using the alpha CSV from step 01
python scripts/02_precision_solve.py \
  --alpha_csv docs/alpha_qwen3_full.csv \
  --output_dir docs/bit_recipes \
  --bpp 3.5 \
  --gamma 10.0 \
  --candidate_bits 2,3,4
```

- **Input**: `--alpha_csv` (output of step 01).
- **Output**: `output_dir/qwen3_coder_next_gamma{X}_bpp{Y}.csv` with columns `name, bit_width`, ready for steps 03 and 04.
- **Arguments**: `--bpp` average bits per parameter; `--gamma` sensitivity exponent; `--candidate_bits` allowed bit widths (e.g. `2,3,4`).

### 3. GPTQ quantization (03_gptq_from_recipe.py)

Runs **GPTQ** on Qwen3-Coder-Next MoE expert weights according to the recipe CSV (block-wise, per-recipe bit width), and saves a quantize-then-dequantize model (same structure as the original, float weights) loadable with `from_pretrained`.

```bash
# Requires GPU; use device_map=auto for multi-GPU
python scripts/03_gptq_from_recipe.py \
  --model Qwen/Qwen3-Coder-Next \
  --recipe_csv docs/bit_recipes/qwen3_coder_next_gamma10.0_bpp3.5.csv \
  --output_dir ./out_qwen3_bpp35 \
  --nsamples 128 \
  --seqlen 2048 \
  --device_map auto
```

- **Input**: `--model`, `--recipe_csv` (from step 02 or a CSV under `qwen3_bit_recipes/`).
- **Output**: Full HuggingFace model under `output_dir` (config + safetensors + tokenizer), ready for step 04 evaluation and step 05 inference.

### 4. Quantized model evaluation (04_eval_quantized.py)

Computes **perplexity (PPL)** on **WikiText-2** for:

- The original HuggingFace model (or local path), or
- The quantize-then-dequantize directory saved in step 03.

```bash
# Evaluate GPTQ output directory (multi-GPU by default when CUDA available)
python scripts/04_eval_quantized.py --model_path ./out_qwen3_bpp35 --seqlen 2048

# Single GPU or explicit device map
python scripts/04_eval_quantized.py --model_path Qwen/Qwen3-Coder-Next --device_map cuda:0
```

- **Input**: `--model_path` (HuggingFace id or local directory). Optional `--device_map` (default: `auto` for multi-GPU when CUDA is available).
- **Output**: WikiText-2 PPL printed to the console; use `--max_nsamples 0` to use the full test set.

### 5. Inference (05_inference.py)

Runs text generation on the **original model** or the **quantized model directory** from step 03. Same API for both (both use `from_pretrained`).

```bash
# Using GPTQ output directory
python scripts/05_inference.py --model_path ./out_qwen3_bpp35 --prompt "Your prompt" --max_new_tokens 64

# Using original model
python scripts/05_inference.py --model_path Qwen/Qwen3-Coder-Next --prompt "Your prompt" --max_new_tokens 64
```

---

## Repository structure

```
AlphaQ/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── scripts/
│   ├── 01_compute_alpha.py       # Compute alpha/variance → CSV
│   ├── 02_precision_solve.py     # Precision solving (MILP) → recipe CSV
│   ├── 03_gptq_from_recipe.py    # GPTQ from recipe → save model
│   ├── 04_eval_quantized.py     # WikiText-2 PPL evaluation
│   ├── 05_inference.py          # Generative inference
│   ├── run_full_pipeline_2layer.sh   # Full pipeline, 2 layers only
│   └── run_full_pipeline_full.sh     # Full pipeline, 48 layers
├── alphaq/                   # AlphaQ core (alpha computation, etc.)
│   ├── __init__.py
│   └── utils_alpha.py
├── gptq.py                   # GPTQ implementation (from MC-MoE)
├── utils/                    # Quantizers and helpers
│   ├── __init__.py
│   ├── quantizer.py
│   ├── quantizer_moe.py
│   └── reconstruct.py
└── qwen3_bit_recipes/        # Optional: place recipe CSVs here
    └── README.txt
```

## References

- **AlphaQ** and **Mixture-Compressor-MoE**: [Aaronhuang-778/Mixture-Compressor-MoE](https://github.com/Aaronhuang-778/Mixture-Compressor-MoE) (ICLR 2025).
- **Qwen3-Coder-Next**: [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next).

The AlphaQ method, precision solver, and scripts in this repo are based on the AlphaQ implementation and Qwen3-Coder-Next adaptation in the Mixture-Compressor-MoE repository.
